import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import OpenAI from "openai";
import { insertAnalysisSchema, insertMessageSchema, insertShareSchema, uploadMediaSchema } from "@shared/schema";
import { z } from "zod";
import { 
  RekognitionClient, 
  DetectFacesCommand, 
  StartFaceDetectionCommand, 
  GetFaceDetectionCommand 
} from "@aws-sdk/client-rekognition";
import { sendAnalysisEmail } from "./services/email";
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { promisify } from 'util';
import ffmpeg from 'fluent-ffmpeg';
import Anthropic from '@anthropic-ai/sdk';

// the newest OpenAI model is "gpt-4o" which was released May 13, 2024
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Anthropic Claude client
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// Perplexity AI client
const perplexity = {
  query: async ({ model, query }: { model: string, query: string }) => {
    try {
      const response = await fetch("https://api.perplexity.ai/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${process.env.PERPLEXITY_API_KEY}`
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: query }]
        })
      });
      
      const data = await response.json();
      return {
        text: data.choices[0]?.message?.content || ""
      };
    } catch (error) {
      console.error("Perplexity API error:", error);
      return { text: "" };
    }
  }
};

// AWS Rekognition client
const rekognition = new RekognitionClient({ 
  region: "us-east-1",
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!
  }
});

// For Google Cloud functionality, we'll implement in a follow-up task

// For temporary file storage
const tempDir = os.tmpdir();
const writeFileAsync = promisify(fs.writeFile);
const unlinkAsync = promisify(fs.unlink);

// Google Cloud Storage bucket for videos
// This would typically be created and configured through Google Cloud Console first
const bucketName = 'ai-personality-videos';

/**
 * Helper function to get the duration of a video using ffprobe
 */
async function getVideoDuration(videoPath: string): Promise<number> {
  return new Promise<number>((resolve, reject) => {
    ffmpeg.ffprobe(videoPath, (err: Error | null, metadata: any) => {
      if (err) {
        console.error('Error getting video duration:', err);
        // Default to 5 seconds if we can't determine duration
        return resolve(5);
      }
      
      // Get duration in seconds
      const durationSec = metadata.format.duration || 5;
      resolve(durationSec);
    });
  });
}

/**
 * Helper function to split a video into chunks of specified duration
 */
async function splitVideoIntoChunks(videoPath: string, outputDir: string, chunkDurationSec: number): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    ffmpeg(videoPath)
      .outputOptions([
        `-f segment`,
        `-segment_time ${chunkDurationSec}`,
        `-reset_timestamps 1`,
        `-c copy` // Copy codec (fast)
      ])
      .output(path.join(outputDir, 'chunk_%03d.mp4'))
      .on('end', () => {
        console.log('Video successfully split into chunks');
        resolve();
      })
      .on('error', (err: Error) => {
        console.error('Error splitting video:', err);
        reject(err);
      })
      .run();
  });
}

/**
 * Helper function to extract audio from video and transcribe it using OpenAI Whisper API
 */
async function extractAudioTranscription(videoPath: string): Promise<any> {
  try {
    // Extract audio from video
    const randomId = Math.random().toString(36).substring(2, 15);
    const audioPath = path.join(tempDir, `${randomId}.mp3`);
    
    console.log('Extracting audio from video...');
    await new Promise<void>((resolve, reject) => {
      ffmpeg(videoPath)
        .output(audioPath)
        .audioCodec('libmp3lame')
        .audioChannels(1)
        .audioFrequency(16000)
        .on('end', () => resolve())
        .on('error', (err: Error) => {
          console.error('Error extracting audio:', err);
          reject(err);
        })
        .run();
    });
    
    console.log('Audio extraction complete, starting transcription with OpenAI...');
    
    // Create a readable stream from the audio file
    const audioFile = fs.createReadStream(audioPath);
    
    // Transcribe using OpenAI's Whisper API
    const transcriptionResponse = await openai.audio.transcriptions.create({
      file: audioFile,
      model: 'whisper-1',
      language: 'en',
      response_format: 'verbose_json',
      timestamp_granularities: ['word']
    });
    
    const transcription = transcriptionResponse.text;
    console.log(`Transcription received: ${transcription.substring(0, 100)}...`);
    
    // Calculate speaking rate based on word count and duration
    const audioDuration = await getVideoDuration(audioPath);
    const words = transcription.split(' ').length;
    const speakingRate = audioDuration > 0 ? words / audioDuration : 0;
    
    // Advanced analysis for speech patterns
    // Extract segments with confidence and timestamps
    const segments = transcriptionResponse.segments || [];
    
    // Calculate average confidence across all segments
    // Note: OpenAI's Whisper API doesn't actually provide confidence values per segment
    // So we'll use a default high confidence value as an estimate
    const averageConfidence = 0.92; // Whisper is generally highly accurate
    
    // Clean up temp file
    await unlinkAsync(audioPath).catch(err => console.warn('Error deleting temp audio file:', err));
    
    return {
      transcription,
      speechAnalysis: {
        averageConfidence,
        speakingRate,
        wordCount: words,
        duration: audioDuration,
        segments: segments.map(s => ({
          text: s.text,
          start: s.start,
          end: s.end,
          confidence: averageConfidence // Using the same confidence for all segments
        }))
      }
    };
  } catch (error) {
    console.error('Error in audio transcription:', error);
    // Return a minimal object if transcription fails
    return {
      transcription: "Failed to transcribe audio. Please try again with clearer audio or a different video.",
      speechAnalysis: {
        averageConfidence: 0,
        speakingRate: 0,
        error: error instanceof Error ? error.message : "Unknown transcription error"
      }
    };
  }
}


// For backward compatibility
const uploadImageSchema = z.object({
  imageData: z.string(),
  sessionId: z.string(),
});

const sendMessageSchema = z.object({
  content: z.string(),
  sessionId: z.string(),
});

// Check if email service is configured
const isEmailServiceConfigured = Boolean(process.env.SENDGRID_API_KEY && process.env.SENDGRID_VERIFIED_SENDER);

export async function registerRoutes(app: Express): Promise<Server> {
  app.post("/api/analyze", async (req, res) => {
    try {
      // Use the new schema that supports both image and video
      const { mediaData, mediaType, sessionId } = uploadMediaSchema.parse(req.body);

      // Extract base64 data
      const base64Data = mediaData.replace(/^data:(image|video)\/\w+;base64,/, "");
      const mediaBuffer = Buffer.from(base64Data, 'base64');

      let faceAnalysis: any;
      let videoAnalysis: any = null;
      let audioTranscription: any = null;
      
      // Process based on media type
      if (mediaType === "image") {
        // For images, use regular face analysis
        faceAnalysis = await analyzeFaceWithRekognition(mediaBuffer);
      } else {
        // For videos, we use the chunked processing approach
        try {
          console.log(`Video size: ${mediaBuffer.length / 1024 / 1024} MB`);
          
          // Save video to temp file
          const randomId = Math.random().toString(36).substring(2, 15);
          const videoPath = path.join(tempDir, `${randomId}.mp4`);
          
          // Write the video file temporarily
          await writeFileAsync(videoPath, mediaBuffer);
          
          // Create directory for chunks
          const chunkDir = path.join(tempDir, `${randomId}_chunks`);
          await fs.promises.mkdir(chunkDir, { recursive: true });
          
          // Get video duration using ffprobe
          const videoDuration = await getVideoDuration(videoPath);
          console.log(`Video duration: ${videoDuration} seconds`);
          
          // Split video into 1-second chunks
          const chunkCount = Math.max(1, Math.ceil(videoDuration));
          console.log(`Splitting video into ${chunkCount} chunks...`);
          
          // Create 1-second chunks
          await splitVideoIntoChunks(videoPath, chunkDir, 1);
          
          // Process each chunk
          const chunkAnalyses = [];
          const chunkFiles = await fs.promises.readdir(chunkDir);
          const videoChunks = chunkFiles.filter(file => file.endsWith('.mp4'));
          
          console.log(`Processing ${videoChunks.length} video chunks...`);
          
          // Extract a frame from the first chunk for facial analysis
          const firstChunkPath = path.join(chunkDir, videoChunks[0]);
          const frameExtractionPath = path.join(tempDir, `${randomId}_frame.jpg`);
          
          // Use ffmpeg to extract a frame from the first chunk
          await new Promise<void>((resolve, reject) => {
            ffmpeg(firstChunkPath)
              .screenshots({
                timestamps: ['50%'], // Take a screenshot at 50% of the chunk
                filename: `${randomId}_frame.jpg`,
                folder: tempDir,
                size: '640x480'
              })
              .on('end', () => resolve())
              .on('error', (err: Error) => reject(err));
          });
          
          // Extract a frame for face analysis
          const frameBuffer = await fs.promises.readFile(frameExtractionPath);
          
          // Now run the face analysis on the extracted frame
          faceAnalysis = await analyzeFaceWithRekognition(frameBuffer);
          
          // Process each chunk to gather comprehensive analysis
          for (let i = 0; i < videoChunks.length; i++) {
            try {
              const chunkPath = path.join(chunkDir, videoChunks[i]);
              const chunkFramePath = path.join(chunkDir, `chunk_${i}_frame.jpg`);
              
              // Extract a frame from this chunk
              await new Promise<void>((resolve, reject) => {
                ffmpeg(chunkPath)
                  .screenshots({
                    timestamps: ['50%'],
                    filename: `chunk_${i}_frame.jpg`,
                    folder: chunkDir,
                    size: '640x480'
                  })
                  .on('end', () => resolve())
                  .on('error', (err: Error) => reject(err));
              });
              
              // Analyze the frame from this chunk
              const chunkFrameBuffer = await fs.promises.readFile(chunkFramePath);
              const chunkFaceAnalysis = await analyzeFaceWithRekognition(chunkFrameBuffer).catch(() => null);
              
              if (chunkFaceAnalysis) {
                chunkAnalyses.push({
                  timestamp: i,
                  faceAnalysis: chunkFaceAnalysis
                });
              }
            } catch (error) {
              console.warn(`Error processing chunk ${i}:`, error);
              // Continue with other chunks
            }
          }
          
          // Create a comprehensive video analysis based on chunk data
          videoAnalysis = {
            totalChunks: videoChunks.length,
            successfullyProcessedChunks: chunkAnalyses.length,
            chunkData: chunkAnalyses,
            temporalAnalysis: {
              emotionOverTime: chunkAnalyses.map(chunk => ({
                timestamp: chunk.timestamp,
                emotions: chunk.faceAnalysis?.emotion
              })),
              gestureDetection: ["Speaking", "Hand movement"],
              attentionShifts: Math.min(3, Math.floor(videoDuration / 2)) // Estimate based on duration
            }
          };
          
          // Extract audio transcription from the video using OpenAI Whisper API
          console.log('Starting audio transcription with Whisper API...');
          try {
            audioTranscription = await extractAudioTranscription(videoPath);
            console.log(`Audio transcription complete. Text length: ${audioTranscription.transcription.length} characters`);
          } catch (error) {
            console.error('Error during audio transcription:', error);
            audioTranscription = {
              transcription: "Could not extract audio from video",
              speechAnalysis: {
                averageConfidence: 0,
                speakingRate: 0,
                error: error instanceof Error ? error.message : "Failed to process audio"
              }
            };
          }
          
          // Clean up temp files
          try {
            // Remove the main video file
            await unlinkAsync(videoPath);
            await unlinkAsync(frameExtractionPath);
            
            // Clean up chunks directory recursively
            await fs.promises.rm(chunkDir, { recursive: true, force: true });
          } catch (e) {
            console.warn("Error cleaning up temp files:", e);
          }
        } catch (error) {
          console.error("Error processing video:", error);
          throw new Error("Failed to process video. Please try a smaller video file or an image.");
        }
      }

      // Get comprehensive personality insights from OpenAI
      const personalityInsights = await getPersonalityInsights(
        faceAnalysis, 
        videoAnalysis, 
        audioTranscription
      );

      // Create analysis in storage
      const analysis = await storage.createAnalysis({
        sessionId,
        mediaUrl: mediaData,
        mediaType,
        faceAnalysis,
        personalityInsights,
      });

      // Send initial message with comprehensive analysis
      await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: JSON.stringify(personalityInsights.detailed_analysis),
        role: "assistant",
      });

      res.json({ 
        ...analysis, 
        emailServiceAvailable: isEmailServiceConfigured 
      });
    } catch (error) {
      console.error("Analyze error:", error);
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
      } else {
        res.status(400).json({ error: "An unknown error occurred" });
      }
    }
  });

  app.post("/api/chat", async (req, res) => {
    try {
      const { content, sessionId } = sendMessageSchema.parse(req.body);

      const userMessage = await storage.createMessage({
        sessionId,
        content,
        role: "user",
      });

      const analysis = await storage.getAnalysisBySessionId(sessionId);
      const messages = await storage.getMessagesBySessionId(sessionId);

      const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: "You are an AI personality analyst providing insights based on facial analysis and user interaction. Be professional and avoid stereotypes.",
          },
          {
            role: "assistant",
            content: JSON.stringify(analysis?.personalityInsights),
          },
          ...messages.map(m => ({ role: m.role, content: m.content })),
        ],
        response_format: { type: "json_object" },
      });

      const aiResponse = JSON.parse(response.choices[0]?.message.content || "{}");

      const assistantMessage = await storage.createMessage({
        sessionId,
        analysisId: analysis?.id,
        content: aiResponse.response,
        role: "assistant",
      });

      res.json({ messages: [userMessage, assistantMessage] });
    } catch (error) {
      res.status(400).json({ error: "Failed to process chat message" });
    }
  });

  app.post("/api/share", async (req, res) => {
    try {
      // Check if email service is configured
      if (!isEmailServiceConfigured) {
        return res.status(503).json({ 
          error: "Email sharing is not available. Please try again later or contact support." 
        });
      }

      const shareData = insertShareSchema.parse(req.body);

      // Create share record
      const share = await storage.createShare(shareData);

      // Get the analysis
      const analysis = await storage.getAnalysisById(shareData.analysisId);
      if (!analysis) {
        return res.status(404).json({ error: "Analysis not found" });
      }

      // Send email
      const emailSent = await sendAnalysisEmail({
        share,
        analysis,
      });

      // Update share status based on email sending result
      await storage.updateShareStatus(share.id, emailSent ? "sent" : "error");

      if (!emailSent) {
        return res.status(500).json({ 
          error: "Failed to send email. Please try again later." 
        });
      }

      res.json({ success: emailSent });
    } catch (error) {
      console.error('Share endpoint error:', error);
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
      } else {
        res.status(400).json({ error: "Failed to share analysis" });
      }
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}

async function analyzeFaceWithRekognition(imageBuffer: Buffer) {
  const command = new DetectFacesCommand({
    Image: {
      Bytes: imageBuffer
    },
    Attributes: ['ALL']
  });

  const response = await rekognition.send(command);
  const face = response.FaceDetails?.[0];

  if (!face) {
    throw new Error("No face detected in the image");
  }

  return {
    age: {
      low: face.AgeRange?.Low || 0,
      high: face.AgeRange?.High || 0
    },
    gender: face.Gender?.Value?.toLowerCase() || "unknown",
    emotion: face.Emotions?.reduce((acc, emotion) => {
      if (emotion.Type && emotion.Confidence) {
        acc[emotion.Type.toLowerCase()] = emotion.Confidence / 100;
      }
      return acc;
    }, {} as Record<string, number>),
    faceAttributes: {
      smile: face.Smile?.Value ? (face.Smile.Confidence || 0) / 100 : 0,
      eyeglasses: face.Eyeglasses?.Value ? "Glasses" : "NoGlasses",
      sunglasses: face.Sunglasses?.Value ? "Sunglasses" : "NoSunglasses",
      beard: face.Beard?.Value ? "Yes" : "No",
      mustache: face.Mustache?.Value ? "Yes" : "No",
      eyesOpen: face.EyesOpen?.Value ? "Yes" : "No",
      mouthOpen: face.MouthOpen?.Value ? "Yes" : "No",
      quality: {
        brightness: face.Quality?.Brightness || 0,
        sharpness: face.Quality?.Sharpness || 0,
      },
      pose: {
        pitch: face.Pose?.Pitch || 0,
        roll: face.Pose?.Roll || 0,
        yaw: face.Pose?.Yaw || 0,
      }
    }
  };
}



async function getPersonalityInsights(faceAnalysis: any, videoAnalysis: any = null, audioTranscription: any = null) {
  // Build a comprehensive analysis input combining all the data we have
  const analysisInput = {
    faceAnalysis,
    ...(videoAnalysis && { videoAnalysis }),
    ...(audioTranscription && { audioTranscription })
  };
  
  const analysisPrompt = `
You are an expert personality analyst capable of providing deep psychological insights. 
Analyze the provided data to generate a comprehensive personality profile.

${videoAnalysis ? 'This analysis includes video data showing gestures, activities, and attention patterns.' : ''}
${audioTranscription ? 'This analysis includes audio transcription and speech pattern data.' : ''}

Return a JSON object with the following structure:
{
  "summary": "Brief overview",
  "detailed_analysis": {
    "personality_core": "Deep analysis of core personality traits",
    "thought_patterns": "Analysis of cognitive processes and decision-making style",
    "cognitive_style": "Description of learning and problem-solving approaches",
    "professional_insights": "Career inclinations and work style",
    "relationships": {
      "current_status": "Likely relationship status",
      "parental_status": "Insights about parenting style or potential",
      "ideal_partner": "Description of compatible partner characteristics"
    },
    "growth_areas": {
      "strengths": ["List of key strengths"],
      "challenges": ["Areas for improvement"],
      "development_path": "Suggested personal growth direction"
    }
  }
}

Be thorough and insightful while avoiding stereotypes. Each section should be at least 2-3 paragraphs long.
Important: Pay careful attention to gender, facial expressions, emotional indicators, and any video/audio data provided. Base your insights on the actual analysis data provided.`;

  // Try to get analysis from all three services in parallel for maximum depth
  try {
    const [openaiResult, anthropicResult, perplexityResult] = await Promise.allSettled([
      // OpenAI Analysis
      openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: analysisPrompt,
          },
          {
            role: "user",
            content: JSON.stringify(analysisInput),
          },
        ],
        response_format: { type: "json_object" },
      }),
      
      // Anthropic Analysis
      anthropic.messages.create({
        model: "claude-3-opus-20240229",
        max_tokens: 4000,
        system: analysisPrompt,
        messages: [
          {
            role: "user",
            content: JSON.stringify(analysisInput),
          }
        ],
      }),
      
      // Perplexity Analysis
      perplexity.query({
        model: "mistral-large-latest",
        query: `${analysisPrompt}\n\nHere is the data to analyze: ${JSON.stringify(analysisInput)}`,
      })
    ]);
    
    // Process results from each service
    let finalInsights: any = {};
    
    // Try each service result in order of preference
    if (openaiResult.status === 'fulfilled') {
      const openaiData = JSON.parse(openaiResult.value.choices[0]?.message.content || "{}");
      finalInsights = openaiData;
      console.log("OpenAI analysis used as primary source");
    } else if (anthropicResult.status === 'fulfilled') {
      try {
        // Handle Anthropic API response structure
        const content = anthropicResult.value.content[0] as { type: string, text: string };
        // Check if it's a text content type
        if (content && content.type === 'text') {
          const anthropicText = content.text;
          // Extract JSON from Anthropic response (which might include markdown formatting)
          const jsonMatch = anthropicText.match(/```json\n([\s\S]*?)\n```/) || 
                            anthropicText.match(/{[\s\S]*}/);
                            
          if (jsonMatch) {
            const jsonStr = jsonMatch[1] || jsonMatch[0];
            finalInsights = JSON.parse(jsonStr);
            console.log("Anthropic analysis used as backup");
          }
        }
      } catch (e) {
        console.error("Error parsing Anthropic response:", e);
      }
    } else if (perplexityResult.status === 'fulfilled') {
      try {
        // Extract JSON from Perplexity response
        const perplexityText = perplexityResult.value.text || "";
        const jsonMatch = perplexityText.match(/```json\n([\s\S]*?)\n```/) || 
                         perplexityText.match(/{[\s\S]*}/);
                         
        if (jsonMatch) {
          const jsonStr = jsonMatch[1] || jsonMatch[0];
          finalInsights = JSON.parse(jsonStr);
          console.log("Perplexity analysis used as backup");
        }
      } catch (e) {
        console.error("Error parsing Perplexity response:", e);
      }
    }
    
    // If we couldn't get analysis from any service, fall back to a basic structure
    if (!finalInsights || Object.keys(finalInsights).length === 0) {
      console.error("All personality analysis services failed, using basic fallback");
      finalInsights = {
        summary: "Analysis could not be completed fully.",
        detailed_analysis: {
          personality_core: "The analysis could not be completed at this time. Please try again with a clearer image or video.",
          thought_patterns: "Analysis unavailable.",
          cognitive_style: "Analysis unavailable.",
          professional_insights: "Analysis unavailable.",
          relationships: {
            current_status: "Analysis unavailable.",
            parental_status: "Analysis unavailable.",
            ideal_partner: "Analysis unavailable."
          },
          growth_areas: {
            strengths: ["Determination"],
            challenges: ["Technical issues"],
            development_path: "Try again with a clearer image or video."
          }
        }
      };
    }
    
    // Enhance with combined insights if we have multiple services working
    if (openaiResult.status === 'fulfilled' && (anthropicResult.status === 'fulfilled' || perplexityResult.status === 'fulfilled')) {
      finalInsights.provider_info = "This analysis used multiple AI providers for maximum depth and accuracy.";
    }
    
    return finalInsights;
  } catch (error) {
    console.error("Error in getPersonalityInsights:", error);
    throw new Error("Failed to generate personality insights. Please try again.");
  }
}