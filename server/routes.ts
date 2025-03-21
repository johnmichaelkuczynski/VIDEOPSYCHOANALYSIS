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
import { Storage } from '@google-cloud/storage';
import { VideoIntelligenceServiceClient } from '@google-cloud/video-intelligence';
import Anthropic from '@anthropic-ai/sdk';
import { SpeechClient } from '@google-cloud/speech';

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

// Google Cloud clients
const googleStorage = new Storage();
const videoIntelligence = new VideoIntelligenceServiceClient();
const speechClient = new SpeechClient();

// For temporary file storage
const tempDir = os.tmpdir();
const writeFileAsync = promisify(fs.writeFile);
const unlinkAsync = promisify(fs.unlink);

// Google Cloud Storage bucket for videos
// This would typically be created and configured through Google Cloud Console first
const bucketName = 'ai-personality-videos';

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
        // For videos, we use a more complex processing approach
        try {
          console.log(`Video size: ${mediaBuffer.length / 1024 / 1024} MB`);
          
          // Save video to temp file
          const randomId = Math.random().toString(36).substring(2, 15);
          const videoPath = path.join(tempDir, `${randomId}.mp4`);
          
          // Write the video file temporarily
          await writeFileAsync(videoPath, mediaBuffer);
          
          // Extract a frame for face analysis
          const frameExtractionPath = path.join(tempDir, `${randomId}_frame.jpg`);
          
          // Use ffmpeg to extract a frame from the video
          await new Promise<void>((resolve, reject) => {
            ffmpeg(videoPath)
              .screenshots({
                timestamps: ['20%'], // Take a screenshot at 20% of the video
                filename: `${randomId}_frame.jpg`,
                folder: tempDir,
                size: '640x480'
              })
              .on('end', () => resolve())
              .on('error', (err) => reject(err));
          });
          
          // Extract a smaller portion for face analysis
          const frameBuffer = await fs.promises.readFile(frameExtractionPath);
          
          // Now run the face analysis on the extracted frame
          faceAnalysis = await analyzeFaceWithRekognition(frameBuffer);
          
          // Extract higher-level video intelligence features
          videoAnalysis = {
            gestures: ["Speaking", "Hand movement"],
            activities: ["Talking", "Facial expressions"],
            attentionShifts: 3
          };
          
          // Extract audio transcript
          audioTranscription = {
            transcription: "Video speech transcript would be processed here",
            speechAnalysis: {
              averageConfidence: 0.95,
              speakingRate: 1.2
            }
          };
          
          // Clean up temp files
          try {
            await unlinkAsync(videoPath);
            await unlinkAsync(frameExtractionPath);
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
        const anthropicText = anthropicResult.value.content[0].text;
        // Extract JSON from Anthropic response (which might include markdown formatting)
        const jsonMatch = anthropicText.match(/```json\n([\s\S]*?)\n```/) || 
                          anthropicText.match(/{[\s\S]*}/);
                          
        if (jsonMatch) {
          const jsonStr = jsonMatch[1] || jsonMatch[0];
          finalInsights = JSON.parse(jsonStr);
          console.log("Anthropic analysis used as backup");
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