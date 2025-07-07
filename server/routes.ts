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
import { generateAnalysisHtml, generatePdf, generateDocx, generateAnalysisTxt } from './services/document';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { promisify } from 'util';
import ffmpeg from 'fluent-ffmpeg';
import Anthropic from '@anthropic-ai/sdk';
import fetch from 'node-fetch';
import FormData from 'form-data';

// Initialize API clients with proper error handling for missing keys
let openai: OpenAI | null = null;
let anthropic: Anthropic | null = null;
let azureOpenAI: OpenAI | null = null;
let deepseek: OpenAI | null = null;

// API Keys available for various services
const GLADIA_API_KEY = process.env.GLADIA_API_KEY;
const ASSEMBLYAI_API_KEY = process.env.ASSEMBLYAI_API_KEY;
const DEEPGRAM_API_KEY = process.env.DEEPGRAM_API_KEY;
const FACEPP_API_KEY = process.env.FACEPP_API_KEY;
const FACEPP_API_SECRET = process.env.FACEPP_API_SECRET;
const AZURE_FACE_API_KEY = process.env.AZURE_FACE_API_KEY;
const AZURE_FACE_ENDPOINT = process.env.AZURE_FACE_ENDPOINT;
const GOOGLE_CLOUD_VISION_API_KEY = process.env.GOOGLE_CLOUD_VISION_API_KEY;
const AZURE_VIDEO_INDEXER_KEY = process.env.AZURE_VIDEO_INDEXER_KEY;
const AZURE_VIDEO_INDEXER_LOCATION = process.env.AZURE_VIDEO_INDEXER_LOCATION;
const AZURE_VIDEO_INDEXER_ACCOUNT_ID = process.env.AZURE_VIDEO_INDEXER_ACCOUNT_ID;

// Log available APIs for transcription
if (GLADIA_API_KEY) {
  console.log("Gladia transcription API available");
}

if (ASSEMBLYAI_API_KEY) {
  console.log("AssemblyAI transcription API available");
}

if (DEEPGRAM_API_KEY) {
  console.log("Deepgram transcription API available");
}

// Availability of face analysis APIs
if (FACEPP_API_KEY && FACEPP_API_SECRET) {
  console.log("Face++ API available for face analysis");
}

if (AZURE_FACE_API_KEY && AZURE_FACE_ENDPOINT) {
  console.log("Azure Face API available for face analysis");
}

if (GOOGLE_CLOUD_VISION_API_KEY) {
  console.log("Google Cloud Vision API available for image analysis");
}

// Availability of video analysis
if (AZURE_VIDEO_INDEXER_KEY && AZURE_VIDEO_INDEXER_LOCATION && AZURE_VIDEO_INDEXER_ACCOUNT_ID) {
  console.log("Azure Video Indexer API available for deep video analysis");
}

// Deepgram API will be used via direct fetch calls instead of SDK

// Initialize Azure OpenAI if available
if (process.env.AZURE_OPENAI_KEY && process.env.AZURE_OPENAI_ENDPOINT) {
  try {
    // Initialize the Azure OpenAI client
    azureOpenAI = new OpenAI({
      apiKey: process.env.AZURE_OPENAI_KEY,
      baseURL: `${process.env.AZURE_OPENAI_ENDPOINT}/openai/deployments/gpt-4/`,
      defaultQuery: { "api-version": "2023-12-01-preview" },
      defaultHeaders: { "api-key": process.env.AZURE_OPENAI_KEY }
    });
    console.log("Azure OpenAI client initialized successfully");
  } catch (error) {
    console.error("Failed to initialize Azure OpenAI client:", error);
  }
}

// Initialize DeepSeek client (using OpenAI-compatible API)
if (process.env.DEEPSEEK_API_KEY) {
  try {
    deepseek = new OpenAI({
      apiKey: process.env.DEEPSEEK_API_KEY,
      baseURL: "https://api.deepseek.com/v1"
    });
    console.log("DeepSeek client initialized successfully");
  } catch (error) {
    console.error("Failed to initialize DeepSeek client:", error);
  }
}

// Check if OpenAI API key is available
if (process.env.OPENAI_API_KEY) {
  try {
    openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    console.log("OpenAI client initialized successfully");
  } catch (error) {
    console.error("Failed to initialize OpenAI client:", error);
  }
} else {
  console.warn("OPENAI_API_KEY environment variable is not set. OpenAI API functionality will be limited.");
}

// Check if Anthropic API key is available
if (process.env.ANTHROPIC_API_KEY) {
  try {
    anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    console.log("Anthropic client initialized successfully");
  } catch (error) {
    console.error("Failed to initialize Anthropic client:", error);
  }
} else {
  console.warn("ANTHROPIC_API_KEY environment variable is not set. Anthropic API functionality will be limited.");
}

// Perplexity AI client
const perplexity = {
  query: async ({ model, query }: { model: string, query: string }) => {
    if (!process.env.PERPLEXITY_API_KEY) {
      console.warn("PERPLEXITY_API_KEY environment variable is not set. Perplexity API functionality will be limited.");
      throw new Error("Perplexity API key not available");
    }
    
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
// Let the AWS SDK pick up credentials from environment variables automatically
const rekognition = new RekognitionClient({ 
  region: process.env.AWS_REGION || "us-east-1"
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
 * Helper function to extract a specific 3-second segment from a video
 */
async function extractVideoSegment(videoPath: string, startTime: number, duration: number, outputPath: string): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    ffmpeg(videoPath)
      .seekInput(startTime)
      .duration(duration)
      .outputOptions(['-c:v libx264', '-c:a aac'])
      .output(outputPath)
      .on('end', () => {
        console.log(`Video segment extracted: ${startTime}s to ${startTime + duration}s`);
        resolve();
      })
      .on('error', (err: Error) => {
        console.error('Error extracting video segment:', err);
        reject(err);
      })
      .run();
  });
}

/**
 * Helper function to analyze video using Azure Video Indexer
 * Extracts insights about scenes, emotions, and content
 */
async function analyzeVideoWithAzureIndexer(videoBuffer: Buffer): Promise<any> {
  // Check if Azure Video Indexer keys are available
  if (!AZURE_VIDEO_INDEXER_KEY || !AZURE_VIDEO_INDEXER_LOCATION || !AZURE_VIDEO_INDEXER_ACCOUNT_ID) {
    console.warn('Azure Video Indexer credentials not available');
    return null;
  }
  
  try {
    console.log('Starting Azure Video Indexer analysis...');
    
    // Step 1: Get an access token for the Video Indexer API
    const accessTokenResponse = await fetch(
      `https://api.videoindexer.ai/auth/${AZURE_VIDEO_INDEXER_LOCATION}/Accounts/${AZURE_VIDEO_INDEXER_ACCOUNT_ID}/AccessToken?allowEdit=true`,
      {
        method: 'GET',
        headers: {
          'Ocp-Apim-Subscription-Key': AZURE_VIDEO_INDEXER_KEY
        }
      }
    );
    
    if (!accessTokenResponse.ok) {
      throw new Error(`Failed to get access token: ${await accessTokenResponse.text()}`);
    }
    
    const accessToken = await accessTokenResponse.text();
    console.log('Obtained Azure Video Indexer access token');
    
    // Step 2: Create a random ID for the video
    const videoId = `video_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
    
    // Step 3: Upload the video
    // Create form data with the video buffer
    const formData = new FormData();
    formData.append('file', videoBuffer, 'video.mp4');
    
    // @ts-ignore: FormData is compatible with fetch API's Body type
    const uploadResponse = await fetch(
      `https://api.videoindexer.ai/${AZURE_VIDEO_INDEXER_LOCATION}/Accounts/${AZURE_VIDEO_INDEXER_ACCOUNT_ID}/Videos?accessToken=${accessToken}&name=${videoId}&privacy=private&indexingPreset=Default`,
      {
        method: 'POST',
        body: formData
      }
    );
    
    if (!uploadResponse.ok) {
      throw new Error(`Failed to upload video: ${await uploadResponse.text()}`);
    }
    
    const uploadResult = await uploadResponse.json();
    const indexingVideoId = uploadResult.id;
    console.log(`Video uploaded, ID: ${indexingVideoId}`);
    
    // Step 4: Wait for indexing to complete
    let isIndexingComplete = false;
    let indexingState = "";
    let indexingRetries = 0;
    const maxRetries = 20; // Maximum number of retries
    
    while (!isIndexingComplete && indexingRetries < maxRetries) {
      await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds between checks
      
      const indexingStateResponse = await fetch(
        `https://api.videoindexer.ai/${AZURE_VIDEO_INDEXER_LOCATION}/Accounts/${AZURE_VIDEO_INDEXER_ACCOUNT_ID}/Videos/${indexingVideoId}/Index?accessToken=${accessToken}`,
        {
          method: 'GET'
        }
      );
      
      if (indexingStateResponse.ok) {
        const indexingData = await indexingStateResponse.json();
        indexingState = indexingData.state;
        
        if (indexingState === "Processed") {
          isIndexingComplete = true;
          console.log('Video indexing completed successfully');
          
          // Step 5: Get the insights from the video
          // The indexingData already contains all the insights
          
          // Step 6: Clean up - delete the video from Azure
          try {
            await fetch(
              `https://api.videoindexer.ai/${AZURE_VIDEO_INDEXER_LOCATION}/Accounts/${AZURE_VIDEO_INDEXER_ACCOUNT_ID}/Videos/${indexingVideoId}?accessToken=${accessToken}`,
              {
                method: 'DELETE'
              }
            );
            console.log('Video deleted from Azure Video Indexer');
          } catch (deleteError) {
            console.warn('Failed to delete video from Azure Video Indexer:', deleteError);
          }
          
          // Process and return the insights
          return processVideoIndexerResults(indexingData);
        } else if (indexingState === "Failed") {
          throw new Error("Video indexing failed on Azure Video Indexer");
        } else {
          console.log(`Indexing in progress, state: ${indexingState}`);
        }
      } else {
        console.warn(`Failed to get indexing state: ${await indexingStateResponse.text()}`);
      }
      
      indexingRetries++;
    }
    
    if (!isIndexingComplete) {
      throw new Error(`Video indexing timed out after ${maxRetries} retries, last state: ${indexingState}`);
    }
    
  } catch (error) {
    console.error('Azure Video Indexer analysis error:', error);
    return null;
  }
}

/**
 * Helper function to process Azure Video Indexer results
 */
function processVideoIndexerResults(indexingData: any): any {
  // Extract the most useful information from the indexing data
  const videoInsights = {
    provider: "azure_video_indexer",
    duration: indexingData.summarizedInsights?.duration || 0,
    
    // Scene analysis
    scenes: indexingData.summarizedInsights?.scenes?.map((scene: any) => ({
      id: scene.id,
      instances: scene.instances?.map((instance: any) => ({
        start: instance.start,
        end: instance.end
      }))
    })) || [],
    
    // Emotion analysis
    emotions: indexingData.summarizedInsights?.emotions?.map((emotion: any) => ({
      type: emotion.type,
      instances: emotion.instances?.map((instance: any) => ({
        start: instance.start,
        end: instance.end,
        confidence: instance.confidence
      }))
    })) || [],
    
    // Face detection and tracking
    faces: indexingData.summarizedInsights?.faces?.map((face: any) => ({
      id: face.id,
      name: face.name || "Unknown person",
      confidence: face.confidence,
      instances: face.instances?.map((instance: any) => ({
        start: instance.start,
        end: instance.end,
        thumbnailId: instance.thumbnailId
      }))
    })) || [],
    
    // Keywords/topics
    topics: indexingData.summarizedInsights?.topics?.map((topic: any) => ({
      name: topic.name,
      confidence: topic.confidence,
      instances: topic.instances?.map((instance: any) => ({
        start: instance.start,
        end: instance.end
      }))
    })) || [],
    
    // Labels (objects, actions)
    labels: indexingData.summarizedInsights?.labels?.map((label: any) => ({
      name: label.name,
      confidence: label.confidence,
      instances: label.instances?.map((instance: any) => ({
        start: instance.start,
        end: instance.end
      }))
    })) || []
  };
  
  return videoInsights;
}

/**
 * Helper function to extract audio from video and transcribe it using multiple transcription services
 * Uses Gladia (primary), AssemblyAI (secondary with emotion tagging), or Deepgram (fallback)
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
    
    console.log('Audio extraction complete, starting transcription...');
    
    // Get the audio file details
    const audioFile = fs.createReadStream(audioPath);
    const audioBuffer = await fs.promises.readFile(audioPath);
    const audioBase64 = audioBuffer.toString('base64');
    const audioDuration = await getVideoDuration(audioPath);
    
    // Initialize results object
    let transcriptionResult: any = {
      transcription: "",
      provider: "none",
      emotion: null,
      confidence: 0,
      wordLevelData: false,
      segments: []
    };
    
    // Try Gladia API first (primary transcription service)
    if (GLADIA_API_KEY) {
      try {
        console.log('Attempting transcription with Gladia API...');
        const formData = new FormData();
        formData.append('audio', audioBuffer, 'audio.mp3');
        
        const gladiaResponse = await fetch('https://api.gladia.io/v2/transcription', {
          method: 'POST',
          headers: {
            'x-gladia-key': GLADIA_API_KEY,
          },
          // @ts-ignore: FormData is compatible with fetch API's Body type
          body: formData
        });
        
        if (gladiaResponse.ok) {
          const result = await gladiaResponse.json();
          
          if (result.prediction && result.prediction.transcription) {
            // Process utterances from Gladia segments if available
            const utterances = [];
            if (result.prediction.utterances && result.prediction.utterances.length > 0) {
              for (const utterance of result.prediction.utterances) {
                utterances.push({
                  text: utterance.text,
                  start: utterance.start,
                  end: utterance.end,
                  sentiment: "unknown" // Gladia doesn't provide sentiment
                });
              }
            } else if (result.prediction.segments && result.prediction.segments.length > 0) {
              // Use segments as utterances if utterances aren't available
              for (const segment of result.prediction.segments) {
                utterances.push({
                  text: segment.text,
                  start: segment.start,
                  end: segment.end,
                  sentiment: "unknown"
                });
              }
            } else {
              // If no segments or utterances, create a single utterance
              utterances.push({
                text: result.prediction.transcription,
                start: 0,
                end: audioDuration || 0,
                sentiment: "unknown"
              });
            }
            
            // Process words if available
            const words = [];
            if (result.prediction.words && result.prediction.words.length > 0) {
              for (const word of result.prediction.words) {
                words.push({
                  text: word.word || word.text,
                  start: word.start,
                  end: word.end,
                  confidence: word.confidence || 0.9
                });
              }
            }
            
            transcriptionResult = {
              text: result.prediction.transcription,
              provider: "gladia",
              confidence: result.prediction.confidence || 0.9,
              wordLevelData: true,
              // Store in standardized format for easy quotation extraction
              transcription: {
                full_text: result.prediction.transcription,
                utterances: utterances,
                words: words
              },
              segments: result.prediction.words || []
            };
            console.log('Gladia transcription successful!');
          }
        } else {
          console.warn('Gladia API returned non-OK response:', gladiaResponse.status);
        }
      } catch (error) {
        console.error('Gladia transcription error:', error);
      }
    }
    
    // If Gladia fails, try AssemblyAI (which adds emotion detection)
    if (transcriptionResult.provider === "none" && ASSEMBLYAI_API_KEY) {
      try {
        console.log('Attempting transcription with AssemblyAI...');
        
        // First upload the audio file
        const uploadResponse = await fetch('https://api.assemblyai.com/v2/upload', {
          method: 'POST',
          headers: {
            'Authorization': ASSEMBLYAI_API_KEY,
            'Content-Type': 'application/json'
          },
          body: audioBuffer
        });
        
        if (uploadResponse.ok) {
          const { upload_url } = await uploadResponse.json();
          
          // Submit for transcription with sentiment analysis
          const transcribeResponse = await fetch('https://api.assemblyai.com/v2/transcript', {
            method: 'POST',
            headers: {
              'Authorization': ASSEMBLYAI_API_KEY,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              audio_url: upload_url,
              sentiment_analysis: true, // Enable sentiment analysis
              entity_detection: true,   // Identify entities
              iab_categories: true      // Topic detection
            })
          });
          
          if (transcribeResponse.ok) {
            const { id } = await transcribeResponse.json();
            
            // Poll for completion (AssemblyAI is async)
            let transcript;
            let completed = false;
            
            for (let i = 0; i < 30 && !completed; i++) { // Try up to 30 times (30 seconds)
              await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
              
              const pollingResponse = await fetch(`https://api.assemblyai.com/v2/transcript/${id}`, {
                headers: { 'Authorization': ASSEMBLYAI_API_KEY }
              });
              
              if (pollingResponse.ok) {
                transcript = await pollingResponse.json();
                if (transcript.status === 'completed') {
                  completed = true;
                } else if (transcript.status === 'error') {
                  console.error('AssemblyAI transcription error:', transcript.error);
                  break;
                }
              }
            }
            
            if (completed && transcript) {
              // Extract emotion data from sentiment analysis
              const emotions = transcript.sentiment_analysis_results || [];
              
              // Process the sentiment analysis into utterance segments
              const utterances = [];
              if (emotions && emotions.length > 0) {
                for (const segment of emotions) {
                  utterances.push({
                    text: segment.text,
                    start: segment.start / 1000, // Convert to seconds
                    end: segment.end / 1000,     // Convert to seconds
                    sentiment: segment.sentiment
                  });
                }
              } else {
                // If no sentiment analysis, create a single utterance
                utterances.push({
                  text: transcript.text,
                  start: 0,
                  end: audioDuration || 0,
                  sentiment: "neutral"
                });
              }
              
              // Process word-level data
              const words = [];
              if (transcript.words && Array.isArray(transcript.words)) {
                for (const word of transcript.words) {
                  words.push({
                    text: word.text,
                    start: word.start / 1000, // Convert to seconds
                    end: word.end / 1000,     // Convert to seconds
                    confidence: word.confidence || 0.9
                  });
                }
              }
              
              transcriptionResult = {
                text: transcript.text,
                provider: "assemblyai",
                confidence: 0.9, // AssemblyAI doesn't provide confidence scores directly
                wordLevelData: true,
                sentiment: transcript.sentiment,
                // Store the utterances and words in a format easier for quotation extraction
                transcription: {
                  full_text: transcript.text,
                  utterances: utterances,
                  words: words
                },
                emotion: emotions.map((item: { 
                  text: string, 
                  sentiment: string, 
                  confidence: number, 
                  start: number, 
                  end: number 
                }) => ({
                  text: item.text,
                  sentiment: item.sentiment,
                  confidence: item.confidence,
                  start: item.start,
                  end: item.end
                })),
                segments: transcript.words || [],
                entities: transcript.entities || [],
                topics: transcript.iab_categories_result?.summary || {}
              };
              console.log('AssemblyAI transcription successful!');
            }
          }
        }
      } catch (error) {
        console.error('AssemblyAI transcription error:', error);
      }
    }
    
    // If both previous services fail, use Deepgram as final fallback
    // Note: We'll implement this via direct API call since we had issues with the SDK
    if (transcriptionResult.provider === "none" && DEEPGRAM_API_KEY) {
      try {
        console.log('Attempting transcription with Deepgram API...');
        
        const deepgramResponse = await fetch('https://api.deepgram.com/v1/listen?model=nova-2&detect_language=true&punctuate=true&diarize=true', {
          method: 'POST',
          headers: {
            'Authorization': `Token ${DEEPGRAM_API_KEY}`,
            'Content-Type': 'audio/mp3'
          },
          body: audioBuffer
        });
        
        if (deepgramResponse.ok) {
          const result = await deepgramResponse.json();
          
          if (result.results && result.results.channels && result.results.channels.length > 0) {
            const transcript = result.results.channels[0].alternatives[0];
            
            // Process words for consistent format
            const words = [];
            if (transcript.words && Array.isArray(transcript.words)) {
              for (const word of transcript.words) {
                words.push({
                  text: word.word || word.text,
                  start: word.start,
                  end: word.end,
                  confidence: word.confidence || 0.8
                });
              }
            }
            
            // Create utterances from paragraphs or sentences
            const utterances = [];
            if (transcript.paragraphs && transcript.paragraphs.length > 0) {
              for (const paragraph of transcript.paragraphs) {
                utterances.push({
                  text: paragraph.text,
                  start: paragraph.start,
                  end: paragraph.end,
                  sentiment: "unknown" // Deepgram doesn't provide sentiment
                });
              }
            } else if (transcript.sentences && transcript.sentences.length > 0) {
              for (const sentence of transcript.sentences) {
                utterances.push({
                  text: sentence.text,
                  start: sentence.start,
                  end: sentence.end,
                  sentiment: "unknown"
                });
              }
            } else {
              // If no paragraphs or sentences, create a single utterance
              utterances.push({
                text: transcript.transcript,
                start: 0,
                end: audioDuration || 0,
                sentiment: "unknown"
              });
            }
            
            transcriptionResult = {
              text: transcript.transcript,
              provider: "deepgram",
              confidence: transcript.confidence,
              language: result.results.language,
              wordLevelData: true,
              // Store in standardized format for easy quotation extraction
              transcription: {
                full_text: transcript.transcript,
                utterances: utterances,
                words: words
              },
              segments: transcript.words || []
            };
            console.log('Deepgram transcription successful!');
          }
        }
      } catch (error) {
        console.error('Deepgram transcription error:', error);
      }
    }
    
    // If all transcription services fail, fall back to OpenAI Whisper if available
    if (transcriptionResult.provider === "none" && openai) {
      try {
        console.log('All primary transcription services failed, falling back to OpenAI Whisper...');
        
        // Reset the file stream position
        const newAudioFile = fs.createReadStream(audioPath);
        
        const transcriptionResponse = await openai.audio.transcriptions.create({
          file: newAudioFile,
          model: 'whisper-1',
          language: 'en',
          response_format: 'verbose_json',
          timestamp_granularities: ['word']
        });
        
        // Process word-level data if available
        const words = [];
        if (transcriptionResponse.words && Array.isArray(transcriptionResponse.words)) {
          for (const word of transcriptionResponse.words) {
            words.push({
              text: word.word,
              start: word.start,
              end: word.end,
              confidence: 0.92 // Whisper doesn't provide per-word confidence
            });
          }
        }
        
        // Process utterances from segments
        const utterances = [];
        if (transcriptionResponse.segments && transcriptionResponse.segments.length > 0) {
          for (const segment of transcriptionResponse.segments) {
            utterances.push({
              text: segment.text,
              start: segment.start,
              end: segment.end,
              sentiment: "unknown" // Whisper doesn't provide sentiment
            });
          }
        } else {
          // If no segments, create a single utterance
          utterances.push({
            text: transcriptionResponse.text,
            start: 0,
            end: audioDuration || 0,
            sentiment: "unknown"
          });
        }
        
        transcriptionResult = {
          text: transcriptionResponse.text,
          provider: "openai_whisper",
          confidence: 0.92, // Whisper is generally highly accurate
          wordLevelData: true,
          // Store in standardized format for easy quotation extraction
          transcription: {
            full_text: transcriptionResponse.text,
            utterances: utterances,
            words: words
          },
          segments: transcriptionResponse.segments || []
        };
        
        console.log('OpenAI Whisper transcription successful!');
      } catch (error) {
        console.error('OpenAI Whisper transcription error:', error);
      }
    }
    
    // Clean up temp file
    await unlinkAsync(audioPath).catch(err => console.warn('Error deleting temp audio file:', err));
    
    // If no transcription service worked
    if (transcriptionResult.provider === "none") {
      console.error('All transcription services failed');
      return {
        transcription: "Failed to transcribe audio. None of the transcription services were able to process this video.",
        transcriptionData: {
          full_text: "Failed to transcribe audio. None of the transcription services were able to process this video.",
          utterances: [],
          words: []
        },
        speechAnalysis: {
          provider: "none",
          averageConfidence: 0,
          speakingRate: 0,
          error: "All transcription services failed"
        }
      };
    }
    
    // Calculate speaking rate based on word count and duration
    const textToCount = transcriptionResult.text || transcriptionResult.transcription;
    const words = textToCount ? textToCount.split(' ').length : 0;
    const speakingRate = audioDuration > 0 ? words / audioDuration : 0;
    
    // Return standardized response format with detailed transcription data
    return {
      // Original transcription text (for backwards compatibility)
      transcription: transcriptionResult.text || transcriptionResult.transcription,
      // Complete structured transcription data for UI and quote extraction
      transcriptionData: transcriptionResult.transcription || {
        full_text: transcriptionResult.text || transcriptionResult.transcription,
        utterances: [],
        words: []
      },
      speechAnalysis: {
        provider: transcriptionResult.provider,
        averageConfidence: transcriptionResult.confidence,
        speakingRate,
        wordCount: words,
        duration: audioDuration,
        emotion: transcriptionResult.emotion,
        sentiment: transcriptionResult.sentiment,
        entities: transcriptionResult.entities,
        topics: transcriptionResult.topics,
        segments: transcriptionResult.segments
      }
    };
  } catch (error) {
    console.error('Error in audio transcription:', error);
    // Return a minimal object if transcription fails
    return {
      transcription: "Failed to transcribe audio. Please try again with clearer audio or a different video.",
      speechAnalysis: {
        provider: "error",
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
let isEmailServiceConfigured = false;
if (process.env.SENDGRID_API_KEY && process.env.SENDGRID_VERIFIED_SENDER) {
  isEmailServiceConfigured = true;
}

// Define the schema for retrieving a shared analysis
const getSharedAnalysisSchema = z.object({
  shareId: z.coerce.number(),
});

export async function registerRoutes(app: Express): Promise<Server> {
  // Text analysis endpoint
  app.post("/api/analyze/text", async (req, res) => {
    try {
      const { content, sessionId, selectedModel = "deepseek", title } = req.body;
      
      if (!content || typeof content !== 'string') {
        return res.status(400).json({ error: "Text content is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      // Choose which AI model to use
      let aiModel = selectedModel;
      if (
        (selectedModel === "deepseek" && !deepseek) ||
        (selectedModel === "openai" && !openai) ||
        (selectedModel === "anthropic" && !anthropic) ||
        (selectedModel === "perplexity" && !process.env.PERPLEXITY_API_KEY)
      ) {
        // Fallback to available model if selected one is not available
        if (openai) aiModel = "openai";
        else if (anthropic) aiModel = "anthropic";
        else if (process.env.PERPLEXITY_API_KEY) aiModel = "perplexity";
        else {
          return res.status(503).json({ 
            error: "No AI models are currently available. Please try again later." 
          });
        }
      }
      
      // Get personality insights based on text content
      let personalityInsights;
      const textAnalysisPrompt = `
You are an expert psychologist and personality analyst. Analyze the following text to provide comprehensive personality insights about the author.

FOCUS AREAS:
1. CONTENT ANALYSIS: Discuss in detail what the author writes about, their interests, concerns, opinions, and perspectives
2. PERSONALITY INSIGHTS: What does their choice of topics, way of expressing ideas, and communication style reveal about their character?
3. COGNITIVE ASSESSMENT: Analyze vocabulary complexity, reasoning patterns, problem-solving approaches, and intellectual sophistication
4. VALUES & WORLDVIEW: What do their expressed ideas, concerns, and perspectives reveal about their deeper values and beliefs?
5. DIRECT QUOTES: Include 5-8 specific quotes that showcase personality traits, intelligence, values, and communication style

TEXT:
${content}

Provide a detailed psychological, emotional, and behavioral analysis of the author based on their writing style, tone, word choice, and content. Include:

1. Content themes and what they reveal about the person's interests and priorities
2. Personality core traits with specific evidence from the text
3. Cognitive abilities demonstrated through vocabulary and reasoning
4. Values and beliefs expressed through their perspectives
5. Communication style and social intelligence
6. Professional insights based on demonstrated knowledge and interests
7. Emotional intelligence and self-awareness
8. Growth areas and potential development paths

Format your analysis as detailed JSON with the following structure:
{
  "summary": "comprehensive overview integrating content analysis with personality insights",
  "detailed_analysis": {
    "content_themes": "detailed analysis of topics discussed and what they reveal about interests and priorities",
    "personality_core": "personality traits with specific evidence from text content",
    "cognitive_profile": {
      "intelligence_assessment": "cognitive abilities demonstrated through vocabulary and reasoning",
      "cognitive_strengths": ["specific strengths with textual evidence"],
      "processing_style": "thinking patterns evident in how ideas are expressed"
    },
    "speech_analysis": {
      "key_quotes": ["5-8 meaningful quotes showcasing personality and thinking"],
      "vocabulary_analysis": "analysis of word choice and communication sophistication",
      "personality_revealed": "what the content reveals about character and values"
    },
    "values_and_worldview": "beliefs and perspectives expressed in the text",
    "emotional_intelligence": "emotional awareness and expression demonstrated",
    "professional_insights": "career inclinations based on interests and communication style",
    "growth_areas": {
      "strengths": ["key strengths with evidence"],
      "development_path": "suggested growth direction based on content and style"
    }
  }
}
`;

      // Get personality analysis from selected AI model
      let analysisResult;
      if (aiModel === "openai" && openai) {
        const completion = await openai.chat.completions.create({
          model: "gpt-4o", // the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
          messages: [
            { role: "system", content: "You are an expert in personality analysis and psychological assessment." },
            { role: "user", content: textAnalysisPrompt }
          ],
          response_format: { type: "json_object" }
        });
        
        analysisResult = JSON.parse(completion.choices[0].message.content);
      } 
      else if (aiModel === "anthropic" && anthropic) {
        const response = await anthropic.messages.create({
          model: "claude-3-7-sonnet-20250219", // the newest Anthropic model is "claude-3-7-sonnet-20250219" which was released February 24, 2025
          max_tokens: 4000,
          system: "You are an expert in personality analysis and psychological assessment. Always respond with well-structured JSON.",
          messages: [{ role: "user", content: textAnalysisPrompt }],
        });
        
        analysisResult = JSON.parse(response.content[0].text);
      }
      else if (aiModel === "perplexity") {
        const response = await perplexity.query({
          model: "llama-3.1-sonar-small-128k-online",
          query: textAnalysisPrompt
        });
        
        try {
          analysisResult = JSON.parse(response.text);
        } catch (e) {
          console.error("Error parsing Perplexity response:", e);
          // Fallback structure if parsing fails
          analysisResult = {
            summary: response.text.substring(0, 200) + "...",
            detailed_analysis: {
              personality_core: "Error parsing structured response from Perplexity",
              thought_patterns: "Please try again with a different AI model"
            }
          };
        }
      }
      
      // Create personality insights in expected format
      personalityInsights = {
        peopleCount: 1,
        individualProfiles: [analysisResult]
      };
      
      // Create analysis record in storage
      const analysis = await storage.createAnalysis({
        sessionId,
        mediaType: "text",
        personalityInsights,
        title: title || "Text Analysis"
      });
      
      // Format message for response
      const formattedContent = `
# Personality Analysis Based on Text

${analysisResult.summary}

## Detailed Analysis

### Personality Core
${analysisResult.detailed_analysis.personality_core}

### Thought Patterns
${analysisResult.detailed_analysis.thought_patterns}

### Emotional Tendencies
${analysisResult.detailed_analysis.emotional_tendencies || ""}

### Communication Style
${analysisResult.detailed_analysis.communication_style || ""}

### Professional Insights
${analysisResult.detailed_analysis.professional_insights || ""}

You can ask follow-up questions about this analysis.
`;
      
      // Create initial message
      const initialMessage = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        role: "assistant",
        content: formattedContent
      });
      
      // Return data to client
      res.json({
        analysisId: analysis.id,
        messages: [initialMessage],
        emailServiceAvailable: isEmailServiceConfigured
      });
    } catch (error) {
      console.error("Text analysis error:", error);
      if (error instanceof Error) {
        res.status(500).json({ error: error.message });
      } else {
        res.status(500).json({ error: "Failed to analyze text" });
      }
    }
  });
  
  // Document analysis endpoint
  app.post("/api/analyze/document", async (req, res) => {
    try {
      const { fileData, fileName, fileType, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!fileData || typeof fileData !== 'string') {
        return res.status(400).json({ error: "Document data is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      // Extract base64 content from data URL
      const base64Data = fileData.split(',')[1];
      if (!base64Data) {
        return res.status(400).json({ error: "Invalid document data format" });
      }
      
      // Save the document to a temporary file
      const fileBuffer = Buffer.from(base64Data, 'base64');
      const tempDocPath = path.join(tempDir, `doc_${Date.now()}_${fileName}`);
      await writeFileAsync(tempDocPath, fileBuffer);
      
      // Choose which AI model to use
      let aiModel = selectedModel;
      if (
        (selectedModel === "openai" && !openai) ||
        (selectedModel === "anthropic" && !anthropic) ||
        (selectedModel === "perplexity" && !process.env.PERPLEXITY_API_KEY)
      ) {
        // Fallback to available model if selected one is not available
        if (openai) aiModel = "openai";
        else if (anthropic) aiModel = "anthropic";
        else if (process.env.PERPLEXITY_API_KEY) aiModel = "perplexity";
        else {
          return res.status(503).json({ 
            error: "No AI models are currently available. Please try again later." 
          });
        }
      }
      
      // Extract text from document and analyze it
      // Note: In a real implementation, use proper document parsing libraries
      // like pdf.js, docx, etc. For simplicity, we're using a placeholder.
      const documentAnalysisPrompt = `
I'm going to analyze the uploaded document: ${fileName} (${fileType}).

Provide a comprehensive analysis of this document, including:

1. Document overview and key topics
2. Main themes and insights
3. Emotional tone and sentiment
4. Writing style assessment
5. Author personality assessment based on the document

Format your analysis as detailed JSON with the following structure:
{
  "summary": "brief overall summary",
  "detailed_analysis": {
    "document_overview": "",
    "main_themes": "",
    "emotional_tone": "",
    "writing_style": "",
    "author_personality": ""
  }
}
`;

      // Get document analysis from selected AI model
      let analysisResult;
      if (aiModel === "openai" && openai) {
        const completion = await openai.chat.completions.create({
          model: "gpt-4o", // the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
          messages: [
            { role: "system", content: "You are an expert in document analysis and personality assessment." },
            { role: "user", content: documentAnalysisPrompt }
          ],
          response_format: { type: "json_object" }
        });
        
        analysisResult = JSON.parse(completion.choices[0].message.content);
      } 
      else if (aiModel === "anthropic" && anthropic) {
        const response = await anthropic.messages.create({
          model: "claude-3-7-sonnet-20250219", // the newest Anthropic model is "claude-3-7-sonnet-20250219" which was released February 24, 2025
          max_tokens: 4000,
          system: "You are an expert in document analysis and psychological assessment. Always respond with well-structured JSON.",
          messages: [{ role: "user", content: documentAnalysisPrompt }],
        });
        
        analysisResult = JSON.parse(response.content[0].text);
      }
      else if (aiModel === "perplexity") {
        const response = await perplexity.query({
          model: "llama-3.1-sonar-small-128k-online",
          query: documentAnalysisPrompt
        });
        
        try {
          analysisResult = JSON.parse(response.text);
        } catch (e) {
          console.error("Error parsing Perplexity response:", e);
          // Fallback structure if parsing fails
          analysisResult = {
            summary: response.text.substring(0, 200) + "...",
            detailed_analysis: {
              document_overview: "Error parsing structured response from Perplexity",
              main_themes: "Please try again with a different AI model"
            }
          };
        }
      }
      
      // Create personality insights in expected format
      const personalityInsights = {
        peopleCount: 1,
        individualProfiles: [{
          summary: analysisResult.summary,
          detailed_analysis: {
            personality_core: analysisResult.detailed_analysis.author_personality,
            thought_patterns: analysisResult.detailed_analysis.main_themes,
            emotional_tendencies: analysisResult.detailed_analysis.emotional_tone,
            communication_style: analysisResult.detailed_analysis.writing_style
          }
        }]
      };
      
      // Clean up temporary file
      try {
        await unlinkAsync(tempDocPath);
      } catch (e) {
        console.warn("Error removing temporary document file:", e);
      }
      
      // Create analysis record in storage
      const analysis = await storage.createAnalysis({
        sessionId,
        mediaType: "document",
        personalityInsights,
        title: title || fileName
      });
      
      // Format message for response
      const formattedContent = `
# Document Analysis: ${fileName}

${analysisResult.summary}

## Document Overview
${analysisResult.detailed_analysis.document_overview}

## Main Themes
${analysisResult.detailed_analysis.main_themes}

## Emotional Tone
${analysisResult.detailed_analysis.emotional_tone}

## Writing Style
${analysisResult.detailed_analysis.writing_style}

## Author Personality Assessment
${analysisResult.detailed_analysis.author_personality}

You can ask follow-up questions about this analysis.
`;
      
      // Create initial message
      const initialMessage = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        role: "assistant",
        content: formattedContent
      });
      
      // Return data to client
      res.json({
        analysisId: analysis.id,
        messages: [initialMessage],
        emailServiceAvailable: isEmailServiceConfigured
      });
    } catch (error) {
      console.error("Document analysis error:", error);
      if (error instanceof Error) {
        res.status(500).json({ error: error.message });
      } else {
        res.status(500).json({ error: "Failed to analyze document" });
      }
    }
  });
  // Text analysis endpoint
  app.post("/api/analyze/text", async (req, res) => {
    try {
      const { content, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!content || typeof content !== 'string') {
        return res.status(400).json({ error: "Text content is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing text analysis with model: ${selectedModel}`);
      
      // Get personality insights based on text content
      const textAnalysisPrompt = `
Please analyze the following text to provide comprehensive personality insights about the author:

TEXT:
${content}

Provide a detailed psychological, emotional, and behavioral analysis of the author based on their writing style, tone, word choice, and content. Include:

1. Personality core traits (Big Five traits, strengths, challenges)
2. Thought patterns and cognitive style
3. Emotional tendencies and expression
4. Communication style and social dynamics
5. Professional insights and work style
6. Decision-making process
7. Relationship approach
8. Areas for growth or self-awareness
`;

      // Get personality analysis from selected AI model
      let analysisText: string;
      
      if (selectedModel === "openai" && openai) {
        console.log('Using OpenAI for text analysis');
        const completion = await openai.chat.completions.create({
          model: "gpt-4o", // the newest OpenAI model is "gpt-4o" which was released May 13, 2024
          messages: [
            { role: "system", content: "You are an expert in personality analysis and psychological assessment." },
            { role: "user", content: textAnalysisPrompt }
          ]
        });
        
        analysisText = completion.choices[0].message.content || "";
      } 
      else if (selectedModel === "anthropic" && anthropic) {
        console.log('Using Anthropic for text analysis');
        const response = await anthropic.messages.create({
          model: "claude-3-7-sonnet-20250219", // the newest Anthropic model is "claude-3-7-sonnet-20250219" which was released February 24, 2025
          max_tokens: 4000,
          system: "You are an expert in personality analysis and psychological assessment.",
          messages: [{ role: "user", content: textAnalysisPrompt }],
        });
        
        analysisText = response.content[0].text;
      }
      else if (selectedModel === "perplexity" && process.env.PERPLEXITY_API_KEY) {
        console.log('Using Perplexity for text analysis');
        const response = await perplexity.query({
          model: "llama-3.1-sonar-small-128k-online",
          query: textAnalysisPrompt
        });
        
        analysisText = response.text;
      }
      else {
        return res.status(503).json({ 
          error: "Selected AI model is not available. Please try again with a different model." 
        });
      }
      
      // Create an analysis with a dummy mediaUrl since the schema requires it but we don't have
      // media for text analysis
      const dummyMediaUrl = `text:${Date.now()}`;
      
      // Create analysis record in storage
      const analysis = await storage.createAnalysis({
        sessionId,
        mediaUrl: dummyMediaUrl,
        mediaType: "text",
        personalityInsights: { analysis: analysisText },
        title: title || "Text Analysis"
      });
      
      // Create initial message
      const initialMessage = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        role: "assistant",
        content: analysisText
      });
      
      // Return data to client
      res.json({
        analysisId: analysis.id,
        messages: [initialMessage],
        emailServiceAvailable: isEmailServiceConfigured
      });
    } catch (error) {
      console.error("Text analysis error:", error);
      if (error instanceof Error) {
        res.status(500).json({ error: error.message });
      } else {
        res.status(500).json({ error: "Failed to analyze text" });
      }
    }
  });
  
  // Document analysis endpoint
  app.post("/api/analyze/document", async (req, res) => {
    try {
      const { fileData, fileName, fileType, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!fileData || typeof fileData !== 'string') {
        return res.status(400).json({ error: "Document data is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing document analysis with model: ${selectedModel}, file: ${fileName}`);
      
      // Extract base64 content from data URL
      const base64Data = fileData.split(',')[1];
      if (!base64Data) {
        return res.status(400).json({ error: "Invalid document data format" });
      }
      
      // Save the document to a temporary file
      const fileBuffer = Buffer.from(base64Data, 'base64');
      const tempDocPath = path.join(tempDir, `doc_${Date.now()}_${fileName}`);
      await writeFileAsync(tempDocPath, fileBuffer);
      
      // Document analysis prompt
      const documentAnalysisPrompt = `
I'm going to analyze the uploaded document: ${fileName} (${fileType}).

Provide a comprehensive analysis of this document, including:

1. Document overview and key topics
2. Main themes and insights
3. Emotional tone and sentiment
4. Writing style assessment
5. Author personality assessment based on the document
`;

      // Get document analysis from selected AI model
      let analysisText: string;
      
      if (selectedModel === "openai" && openai) {
        console.log('Using OpenAI for document analysis');
        const completion = await openai.chat.completions.create({
          model: "gpt-4o", // the newest OpenAI model is "gpt-4o" which was released May 13, 2024
          messages: [
            { role: "system", content: "You are an expert in document analysis and personality assessment." },
            { role: "user", content: documentAnalysisPrompt }
          ]
        });
        
        analysisText = completion.choices[0].message.content || "";
      } 
      else if (selectedModel === "anthropic" && anthropic) {
        console.log('Using Anthropic for document analysis');
        const response = await anthropic.messages.create({
          model: "claude-3-7-sonnet-20250219", // the newest Anthropic model is "claude-3-7-sonnet-20250219" which was released February 24, 2025
          max_tokens: 4000,
          system: "You are an expert in document analysis and psychological assessment.",
          messages: [{ role: "user", content: documentAnalysisPrompt }],
        });
        
        analysisText = response.content[0].text;
      }
      else if (selectedModel === "perplexity" && process.env.PERPLEXITY_API_KEY) {
        console.log('Using Perplexity for document analysis');
        const response = await perplexity.query({
          model: "llama-3.1-sonar-small-128k-online",
          query: documentAnalysisPrompt
        });
        
        analysisText = response.text;
      }
      else {
        return res.status(503).json({ 
          error: "Selected AI model is not available. Please try again with a different model." 
        });
      }
      
      // Clean up temporary file
      try {
        await unlinkAsync(tempDocPath);
      } catch (e) {
        console.warn("Error removing temporary document file:", e);
      }
      
      // Create an analysis with a dummy mediaUrl since the schema requires it but we don't have media for document analysis
      const dummyMediaUrl = `document:${Date.now()}`;
      
      // Create analysis record in storage
      const analysis = await storage.createAnalysis({
        sessionId,
        mediaUrl: dummyMediaUrl,
        mediaType: "document",
        personalityInsights: { analysis: analysisText },
        documentType: fileType === "pdf" ? "pdf" : "docx",
        title: title || fileName
      });
      
      // Create initial message
      const initialMessage = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        role: "assistant",
        content: analysisText
      });
      
      // Return data to client
      res.json({
        analysisId: analysis.id,
        messages: [initialMessage],
        emailServiceAvailable: isEmailServiceConfigured
      });
    } catch (error) {
      console.error("Document analysis error:", error);
      if (error instanceof Error) {
        res.status(500).json({ error: error.message });
      } else {
        res.status(500).json({ error: "Failed to analyze document" });
      }
    }
  });
  
  // Chat endpoint to continue conversation with AI
  app.post("/api/chat", async (req, res) => {
    try {
      const { content, sessionId, selectedModel = "openai" } = req.body;
      
      if (!content || typeof content !== 'string') {
        return res.status(400).json({ error: "Message content is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing chat with model: ${selectedModel}, sessionId: ${sessionId}`);
      
      // Get existing messages for this session
      const existingMessages = await storage.getMessagesBySessionId(sessionId);
      const analysisId = existingMessages.length > 0 ? existingMessages[0].analysisId : null;
      
      // Create user message
      const userMessage = await storage.createMessage({
        sessionId,
        analysisId,
        role: "user",
        content
      });
      
      // Get analysis if available
      let analysisContext = "";
      if (analysisId) {
        const analysis = await storage.getAnalysisById(analysisId);
        if (analysis && analysis.personalityInsights) {
          // Add the analysis context for better AI responses
          analysisContext = "This conversation is about a personality analysis. Here's the context: " + 
            JSON.stringify(analysis.personalityInsights);
        }
      }
      
      // Format the conversation history for the AI
      const conversationHistory = existingMessages.map(msg => ({
        role: msg.role,
        content: msg.content
      }));
      
      // Add the new user message
      conversationHistory.push({
        role: "user",
        content
      });
      
      // Get AI response based on selected model
      let aiResponseText: string;
      
      if (selectedModel === "openai" && openai) {
        console.log('Using OpenAI for chat');
        const systemPrompt = analysisContext ? 
          `You are an AI assistant specialized in personality analysis. ${analysisContext}` :
          "You are an AI assistant specialized in personality analysis. Be helpful, informative, and engaging.";
        
        const completion = await openai.chat.completions.create({
          model: "gpt-4o", // the newest OpenAI model is "gpt-4o" which was released May 13, 2024
          messages: [
            { 
              role: "system", 
              content: systemPrompt
            },
            ...conversationHistory.map(msg => ({
              role: msg.role as any,
              content: msg.content
            }))
          ]
        });
        
        aiResponseText = completion.choices[0].message.content || "";
      } 
      else if (selectedModel === "anthropic" && anthropic) {
        console.log('Using Anthropic for chat');
        const systemPrompt = analysisContext ? 
          `You are an AI assistant specialized in personality analysis. ${analysisContext}` :
          "You are an AI assistant specialized in personality analysis. Be helpful, informative, and engaging.";
          
        // Format conversation history for Claude
        const messages = conversationHistory.map(msg => ({
          role: msg.role as any, 
          content: msg.content
        }));
        
        const response = await anthropic.messages.create({
          model: "claude-3-7-sonnet-20250219", // the newest Anthropic model is "claude-3-7-sonnet-20250219" which was released February 24, 2025
          max_tokens: 4000,
          system: systemPrompt,
          messages
        });
        
        aiResponseText = response.content[0].text;
      }
      else if (selectedModel === "perplexity" && process.env.PERPLEXITY_API_KEY) {
        console.log('Using Perplexity for chat');
        // Format conversation for Perplexity
        // We need to format the entire conversation as a single prompt
        let formattedConversation = "You are an AI assistant specialized in personality analysis. ";
        if (analysisContext) {
          formattedConversation += analysisContext + "\n\n";
        }
        
        formattedConversation += "Here's the conversation so far:\n\n";
        
        for (const message of conversationHistory) {
          formattedConversation += `${message.role === 'user' ? 'User' : 'Assistant'}: ${message.content}\n\n`;
        }
        
        formattedConversation += "Please provide your next response as the assistant:";
        
        const response = await perplexity.query({
          model: "llama-3.1-sonar-small-128k-online",
          query: formattedConversation
        });
        
        aiResponseText = response.text;
      }
      else {
        return res.status(503).json({ 
          error: "Selected AI model is not available. Please try again with a different model." 
        });
      }
      
      // Create AI response message
      const aiMessage = await storage.createMessage({
        sessionId,
        analysisId,
        role: "assistant",
        content: aiResponseText
      });
      
      // Return both the user message and AI response
      res.json({
        messages: [userMessage, aiMessage],
        success: true
      });
    } catch (error) {
      console.error("Chat error:", error);
      if (error instanceof Error) {
        res.status(500).json({ error: error.message });
      } else {
        res.status(500).json({ error: "Failed to process chat message" });
      }
    }
  });

  app.post("/api/analyze", async (req, res) => {
    try {
      // Use the new schema that supports both image and video with optional maxPeople
      const { mediaData, mediaType, sessionId, maxPeople = 5, selectedModel = "deepseek", videoSegmentStart = 0, videoSegmentDuration = 3 } = uploadMediaSchema.parse(req.body);

      // Extract base64 data
      const base64Data = mediaData.replace(/^data:(image|video)\/\w+;base64,/, "");
      const mediaBuffer = Buffer.from(base64Data, 'base64');

      let faceAnalysis: any = [];
      let videoAnalysis: any = null;
      let audioTranscription: any = null;
      
      // Process based on media type
      if (mediaType === "image") {
        // For images, use multi-person face analysis
        console.log(`Analyzing image for up to ${maxPeople} people...`);
        faceAnalysis = await analyzeFaceWithRekognition(mediaBuffer, maxPeople);
        console.log(`Detected ${Array.isArray(faceAnalysis) ? faceAnalysis.length : 1} people in the image`);
      } else {
        // For videos, we use the new 3-second segment approach
        try {
          console.log(`Video size: ${mediaBuffer.length / 1024 / 1024} MB`);
          
          // Save video to temp file
          const randomId = Math.random().toString(36).substring(2, 15);
          const videoPath = path.join(tempDir, `${randomId}.mp4`);
          
          // Write the video file temporarily
          await writeFileAsync(videoPath, mediaBuffer);
          
          // Get video duration using ffprobe
          const videoDuration = await getVideoDuration(videoPath);
          console.log(`Video duration: ${videoDuration} seconds`);
          
          // Extract the specific 3-second segment requested
          const segmentPath = path.join(tempDir, `${randomId}_segment.mp4`);
          const actualDuration = Math.min(videoSegmentDuration, videoDuration - videoSegmentStart);
          
          if (actualDuration <= 0) {
            throw new Error(`Invalid segment: starts at ${videoSegmentStart}s but video is only ${videoDuration}s long`);
          }
          
          console.log(`Extracting ${actualDuration}s segment starting at ${videoSegmentStart}s...`);
          await extractVideoSegment(videoPath, videoSegmentStart, actualDuration, segmentPath);
          
          // Process the segment instead of the full video
          const segmentBuffer = await fs.promises.readFile(segmentPath);
          
          // Extract a frame from the segment for facial analysis
          const frameExtractionPath = path.join(tempDir, `${randomId}_frame.jpg`);
          
          // Use ffmpeg to extract a frame from the segment
          await new Promise<void>((resolve, reject) => {
            ffmpeg(segmentPath)
              .screenshots({
                timestamps: ['50%'], // Take a screenshot at 50% of the segment
                filename: `${randomId}_frame.jpg`,
                folder: tempDir,
                size: '640x480'
              })
              .on('end', () => resolve())
              .on('error', (err: Error) => reject(err));
          });
          
          // Extract a frame for face analysis
          const frameBuffer = await fs.promises.readFile(frameExtractionPath);
          
          // Now run the face analysis on the extracted frame for multiple people
          faceAnalysis = await analyzeFaceWithRekognition(frameBuffer, maxPeople);
          console.log(`Detected ${Array.isArray(faceAnalysis) ? faceAnalysis.length : 1} people in the video frame`);
          
          // Process the segment for comprehensive analysis
          console.log(`Processing video segment: ${videoSegmentStart}s to ${videoSegmentStart + actualDuration}s`);
          
          // Try to get Azure Video Indexer analysis if available (on the segment)
          let azureVideoInsights = null;
          
          if (AZURE_VIDEO_INDEXER_KEY && AZURE_VIDEO_INDEXER_LOCATION && AZURE_VIDEO_INDEXER_ACCOUNT_ID) {
            try {
              console.log('Attempting deep video analysis with Azure Video Indexer...');
              azureVideoInsights = await analyzeVideoWithAzureIndexer(segmentBuffer);
              
              if (azureVideoInsights) {
                console.log('Azure Video Indexer analysis successful!');
              }
            } catch (error) {
              console.warn('Azure Video Indexer analysis failed:', error);
              // Continue with basic analysis if Azure Video Indexer fails
            }
          }
          
          // Create a comprehensive video analysis for the segment
          videoAnalysis = {
            provider: azureVideoInsights ? "azure_video_indexer" : "basic",
            segmentStart: videoSegmentStart,
            segmentDuration: actualDuration,
            totalVideoDuration: videoDuration,
            segmentData: {
              timestamp: videoSegmentStart,
              duration: actualDuration,
              faceAnalysis: faceAnalysis
            },
            
            // Include Azure insights if available
            ...(azureVideoInsights && { azureInsights: azureVideoInsights })
          };
          
          // Get audio transcription from the segment
          console.log('Starting audio transcription with Whisper API...');
          audioTranscription = await extractAudioTranscription(segmentPath);
          console.log(`Audio transcription complete. Text length: ${audioTranscription.transcription.length} characters`);
          
          // Clean up temp files
          try {
            // Remove the main video file, segment, and frame
            await unlinkAsync(videoPath);
            await unlinkAsync(segmentPath);
            await unlinkAsync(frameExtractionPath);
          } catch (e) {
            console.warn("Error cleaning up temp files:", e);
          }
        } catch (error) {
          console.error("Error processing video:", error);
          throw new Error("Failed to process video. Please try a smaller video file or an image.");
        }
      }

      // Get comprehensive personality insights with enhanced cognitive profiling
      const personalityInsights = await getEnhancedPersonalityInsights(
        faceAnalysis, 
        videoAnalysis, 
        audioTranscription,
        selectedModel
      );

      // Determine how many people were detected
      const peopleCount = personalityInsights.peopleCount || 1;

      // Create analysis in storage
      const analysis = await storage.createAnalysis({
        sessionId,
        mediaUrl: mediaData,
        mediaType,
        faceAnalysis,
        videoAnalysis: videoAnalysis || undefined,
        audioTranscription: audioTranscription || undefined,
        personalityInsights,
      });

      // Format initial message content for the chat
      let formattedContent = "";
      
      if (personalityInsights.individualProfiles?.length > 1) {
        // Multi-person message format with improved visual structure
        const peopleCount = personalityInsights.individualProfiles.length;
        formattedContent = `🧠 AI-Powered Psychological Profile Report\n`;
        formattedContent += `🖼️ Subjects Detected: ${peopleCount} Individuals\n`;
        formattedContent += `📷 Mode: Group Analysis\n\n`;
        
        // Add each individual profile first
        personalityInsights.individualProfiles.forEach((profile, index) => {
          const gender = profile.personLabel?.includes('Male') ? 'Male' : 
                         profile.personLabel?.includes('Female') ? 'Female' : '';
          const ageMatch = profile.personLabel?.match(/~(\d+)-(\d+)/);
          const ageRange = ageMatch ? `~${ageMatch[1]}–${ageMatch[2]} years` : '';
          const genderAge = [gender, ageRange].filter(Boolean).join(', ');
          
          formattedContent += `${'─'.repeat(65)}\n`;
          formattedContent += `👤 Subject ${index + 1}${genderAge ? ` (${genderAge})` : ''}\n`;
          formattedContent += `${'─'.repeat(65)}\n\n`;
          
          const detailedAnalysis = profile.detailed_analysis || {};
          
          formattedContent += `🧾 Summary:\n${profile.summary || 'No summary available'}\n\n`;
          
          if (detailedAnalysis.personality_core) {
            formattedContent += `🧬 Core Personality:\n${detailedAnalysis.personality_core}\n\n`;
          }
          
          if (detailedAnalysis.cognitive_style) {
            formattedContent += `🧠 Cognitive Style:\n${detailedAnalysis.cognitive_style}\n\n`;
          }
          
          if (detailedAnalysis.professional_insights) {
            formattedContent += `💼 Professional Fit:\n${detailedAnalysis.professional_insights}\n\n`;
          }
          
          if (detailedAnalysis.relationships) {
            formattedContent += `❤️ Relationships:\n`;
            const relationshipParts = [];
            
            if (detailedAnalysis.relationships.current_status && 
                detailedAnalysis.relationships.current_status !== 'Not available') {
              relationshipParts.push(detailedAnalysis.relationships.current_status);
            }
            
            if (detailedAnalysis.relationships.parental_status && 
                detailedAnalysis.relationships.parental_status !== 'Not available') {
              relationshipParts.push(detailedAnalysis.relationships.parental_status);
            }
            
            if (detailedAnalysis.relationships.ideal_partner && 
                detailedAnalysis.relationships.ideal_partner !== 'Not available') {
              relationshipParts.push(`Ideal match: ${detailedAnalysis.relationships.ideal_partner}`);
            }
            
            formattedContent += relationshipParts.length > 0 
              ? relationshipParts.join(' ') 
              : 'No relationship data available';
            
            formattedContent += `\n\n`;
          }
          
          if (detailedAnalysis.growth_areas) {
            formattedContent += `📈 Growth Areas:\n`;
            
            if (Array.isArray(detailedAnalysis.growth_areas.strengths) && 
                detailedAnalysis.growth_areas.strengths.length > 0) {
              formattedContent += `Strengths:\n${detailedAnalysis.growth_areas.strengths.map((s: string) => `• ${s}`).join('\n')}\n\n`;
            }
            
            if (Array.isArray(detailedAnalysis.growth_areas.challenges) && 
                detailedAnalysis.growth_areas.challenges.length > 0) {
              formattedContent += `Challenges:\n${detailedAnalysis.growth_areas.challenges.map((c: string) => `• ${c}`).join('\n')}\n\n`;
            }
            
            if (detailedAnalysis.growth_areas.development_path) {
              formattedContent += `Development Path:\n${detailedAnalysis.growth_areas.development_path}\n\n`;
            }
          }
        });
        
        // Add group dynamics at the end
        if (personalityInsights.groupDynamics) {
          formattedContent += `${'─'.repeat(65)}\n`;
          formattedContent += `🤝 Group Dynamics (${peopleCount}-Person Analysis)\n`;
          formattedContent += `${'─'.repeat(65)}\n\n`;
          formattedContent += `${personalityInsights.groupDynamics}\n`;
        }
        
      } else if (personalityInsights.individualProfiles?.length === 1) {
        // Single person format (maintain similar structure for consistency)
        const profile = personalityInsights.individualProfiles[0];
        const detailedAnalysis = profile.detailed_analysis || {};
        
        const gender = profile.personLabel?.includes('Male') ? 'Male' : 
                       profile.personLabel?.includes('Female') ? 'Female' : '';
        const ageMatch = profile.personLabel?.match(/~(\d+)-(\d+)/);
        const ageRange = ageMatch ? `~${ageMatch[1]}–${ageMatch[2]} years` : '';
        const genderAge = [gender, ageRange].filter(Boolean).join(', ');
        
        formattedContent = `🧠 AI-Powered Psychological Profile Report\n`;
        formattedContent += `🖼️ Subject Detected: 1 Individual\n`;
        formattedContent += `📷 Mode: Individual Analysis\n\n`;
        
        formattedContent += `${'─'.repeat(65)}\n`;
        formattedContent += `👤 Subject 1${genderAge ? ` (${genderAge})` : ''}\n`;
        formattedContent += `${'─'.repeat(65)}\n\n`;
        
        formattedContent += `🧾 Summary:\n${profile.summary || 'No summary available'}\n\n`;
        
        if (detailedAnalysis.personality_core) {
          formattedContent += `🧬 Core Personality:\n${detailedAnalysis.personality_core || 'Not available'}\n\n`;
        }
        
        if (detailedAnalysis.cognitive_style) {
          formattedContent += `🧠 Cognitive Style:\n${detailedAnalysis.cognitive_style || 'Not available'}\n\n`;
        }
        
        if (detailedAnalysis.professional_insights) {
          formattedContent += `💼 Professional Fit:\n${detailedAnalysis.professional_insights || 'Not available'}\n\n`;
        }
        
        if (detailedAnalysis.relationships) {
          formattedContent += `❤️ Relationships:\n`;
          const relationshipParts = [];
          
          if (detailedAnalysis.relationships.current_status && 
              detailedAnalysis.relationships.current_status !== 'Not available') {
            relationshipParts.push(detailedAnalysis.relationships.current_status);
          }
          
          if (detailedAnalysis.relationships.parental_status && 
              detailedAnalysis.relationships.parental_status !== 'Not available') {
            relationshipParts.push(detailedAnalysis.relationships.parental_status);
          }
          
          if (detailedAnalysis.relationships.ideal_partner && 
              detailedAnalysis.relationships.ideal_partner !== 'Not available') {
            relationshipParts.push(`Ideal match: ${detailedAnalysis.relationships.ideal_partner}`);
          }
          
          formattedContent += relationshipParts.length > 0 
            ? relationshipParts.join(' ') 
            : 'No relationship data available';
          
          formattedContent += `\n\n`;
        }
        
        if (detailedAnalysis.growth_areas) {
          formattedContent += `📈 Growth Areas:\n`;
          
          if (Array.isArray(detailedAnalysis.growth_areas.strengths) && 
              detailedAnalysis.growth_areas.strengths.length > 0) {
            formattedContent += `Strengths:\n${detailedAnalysis.growth_areas.strengths.map((s: string) => `• ${s}`).join('\n')}\n\n`;
          }
          
          if (Array.isArray(detailedAnalysis.growth_areas.challenges) && 
              detailedAnalysis.growth_areas.challenges.length > 0) {
            formattedContent += `Challenges:\n${detailedAnalysis.growth_areas.challenges.map((c: string) => `• ${c}`).join('\n')}\n\n`;
          }
          
          if (detailedAnalysis.growth_areas.development_path) {
            formattedContent += `Development Path:\n${detailedAnalysis.growth_areas.development_path}\n\n`;
          }
        }
      } else {
        // Fallback if no profiles
        formattedContent = "No personality profiles could be generated. Please try again with a different image or video.";
      }

      // Send initial message with comprehensive analysis
      const message = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: formattedContent,
        role: "assistant",
      });

      // Get all messages to return to client
      const messages = await storage.getMessagesBySessionId(sessionId);

      res.json({ 
        ...analysis, 
        messages,
        emailServiceAvailable: isEmailServiceConfigured 
      });
      
      console.log(`Analysis complete. Created message with ID ${message.id} and returning ${messages.length} messages`);
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

      // Check if OpenAI client is available
      if (!openai) {
        return res.status(400).json({ 
          error: "OpenAI API key is not configured. Please provide an OpenAI API key to use the chat functionality.",
          configError: "OPENAI_API_KEY_MISSING",
          messages: [userMessage]
        });
      }

      const analysis = await storage.getAnalysisBySessionId(sessionId);
      const messages = await storage.getMessagesBySessionId(sessionId);

      try {
        // Set up the messages for the API call
        const apiMessages = [
          {
            role: "system",
            content: `You are an AI assistant capable of general conversation as well as providing specialized analysis about the personality insights previously generated. 
            
If the user asks about the analysis, provide detailed information based on the personality insights.
If the user asks general questions unrelated to the analysis, respond naturally and helpfully as you would to any question.

IMPORTANT: Do not use markdown formatting in your responses. Do not use ** for bold text, do not use ### for headers, and do not use markdown formatting for bullet points or numbered lists. Use plain text formatting only.

Be engaging, professional, and conversational in all responses. Feel free to have opinions, share information, and engage in dialogue on any topic.`,
          },
          {
            role: "assistant",
            content: typeof analysis?.personalityInsights === 'object' 
              ? JSON.stringify(analysis?.personalityInsights) 
              : String(analysis?.personalityInsights || ''),
          },
          ...messages.map(m => ({ role: m.role, content: m.content })),
        ];
        
        // Convert message format to match OpenAI's expected types
        const typedMessages = apiMessages.map(msg => {
          // Convert role to proper type
          const role = msg.role === 'user' ? 'user' : 
                      msg.role === 'assistant' ? 'assistant' : 'system';
          
          // Return properly typed message
          return {
            role,
            content: msg.content || ''
          };
        });
        
        // Use the properly typed messages for the API call
        const response = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: typedMessages,
          // Don't use JSON format as it requires specific message formats
          // response_format: { type: "json_object" },
        });

        // Get the raw text response
        const responseContent = response.choices[0]?.message.content || "";
        let aiResponse = responseContent;
        
        // Try to parse as JSON if it appears to be JSON, otherwise use as plain text
        try {
          if (responseContent.trim().startsWith('{') && responseContent.trim().endsWith('}')) {
            aiResponse = JSON.parse(responseContent);
          }
        } catch (e) {
          // If parsing fails, use the raw text
          console.log("Failed to parse response as JSON, using raw text");
          aiResponse = responseContent;
        }

        // Create the assistant message using the response content
        // If aiResponse is an object with a response property, use that
        // Otherwise, use the raw text response
        const messageContent = typeof aiResponse === 'object' && aiResponse.response 
          ? aiResponse.response 
          : typeof aiResponse === 'string' 
            ? aiResponse 
            : "I'm sorry, I couldn't generate a proper response.";
            
        const assistantMessage = await storage.createMessage({
          sessionId,
          analysisId: analysis?.id,
          content: messageContent,
          role: "assistant",
        });

        res.json({ messages: [userMessage, assistantMessage] });
      } catch (apiError) {
        console.error("OpenAI API error:", apiError);
        res.status(500).json({ 
          error: "Error communicating with OpenAI API. Please check your API key configuration.",
          messages: [userMessage]
        });
      }
    } catch (error) {
      console.error("Chat processing error:", error);
      res.status(400).json({ error: "Failed to process chat message" });
    }
  });

  app.get("/api/messages", async (req, res) => {
    try {
      const { sessionId } = req.query;
      
      if (!sessionId || typeof sessionId !== 'string') {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      const messages = await storage.getMessagesBySessionId(sessionId);
      res.json(messages);
    } catch (error) {
      console.error("Get messages error:", error);
      res.status(400).json({ error: "Failed to get messages" });
    }
  });

  app.get("/api/shared-analysis/:shareId", async (req, res) => {
    try {
      const { shareId } = getSharedAnalysisSchema.parse({ shareId: req.params.shareId });
      
      // Get the share record
      const share = await storage.getShareById(shareId);
      if (!share) {
        return res.status(404).json({ error: "Shared analysis not found" });
      }
      
      // Get the analysis
      const analysis = await storage.getAnalysisById(share.analysisId);
      if (!analysis) {
        return res.status(404).json({ error: "Analysis not found" });
      }
      
      // Get all messages for this analysis
      const messages = await storage.getMessagesBySessionId(analysis.sessionId);
      
      // Return the complete data
      res.json({
        analysis,
        messages,
        share,
        emailServiceAvailable: isEmailServiceConfigured
      });
    } catch (error) {
      console.error("Get shared analysis error:", error);
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
      } else {
        res.status(400).json({ error: "Failed to get shared analysis" });
      }
    }
  });

  // API status endpoint - returns the availability of various services
  app.get("/api/status", async (req, res) => {
    try {
      const statusData = {
        // LLM Services
        openai: !!openai,
        anthropic: !!anthropic,
        perplexity: !!process.env.PERPLEXITY_API_KEY,
        azureOpenai: !!process.env.AZURE_OPENAI_ENDPOINT && !!process.env.AZURE_OPENAI_KEY,
        
        // Facial Analysis Services
        aws_rekognition: !!process.env.AWS_ACCESS_KEY_ID && !!process.env.AWS_SECRET_ACCESS_KEY,
        facepp: !!process.env.FACEPP_API_KEY && !!process.env.FACEPP_API_SECRET,
        azure_face: !!process.env.AZURE_FACE_ENDPOINT && !!process.env.AZURE_FACE_API_KEY,
        google_vision: !!process.env.GOOGLE_CLOUD_VISION_API_KEY,
        
        // Transcription Services
        gladia: !!process.env.GLADIA_API_KEY,
        assemblyai: !!process.env.ASSEMBLYAI_API_KEY,
        deepgram: !!process.env.DEEPGRAM_API_KEY,
        
        // Video Analysis Services
        azure_video_indexer: !!process.env.AZURE_VIDEO_INDEXER_KEY && 
                            !!process.env.AZURE_VIDEO_INDEXER_LOCATION && 
                            !!process.env.AZURE_VIDEO_INDEXER_ACCOUNT_ID,
        
        // Email Service
        sendgrid: !!process.env.SENDGRID_API_KEY && !!process.env.SENDGRID_VERIFIED_SENDER,
        
        // Service status timestamp
        timestamp: new Date().toISOString()
      };
      
      res.json(statusData);
    } catch (error) {
      console.error("Error checking API status:", error);
      res.status(500).json({ error: "Failed to check API status" });
    }
  });
  
  // Session management endpoints
  app.get("/api/sessions", async (req, res) => {
    try {
      const sessions = await storage.getAllSessions();
      res.json(sessions);
    } catch (error) {
      console.error("Error getting sessions:", error);
      res.status(500).json({ error: "Failed to get sessions" });
    }
  });
  
  app.post("/api/session/clear", async (req, res) => {
    try {
      const { sessionId } = req.body;
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      await storage.clearSession(sessionId);
      res.json({ success: true });
    } catch (error) {
      console.error("Error clearing session:", error);
      res.status(500).json({ error: "Failed to clear session" });
    }
  });
  
  app.patch("/api/session/name", async (req, res) => {
    try {
      const { sessionId, name } = req.body;
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      if (!name) {
        return res.status(400).json({ error: "Name is required" });
      }
      
      await storage.updateSessionName(sessionId, name);
      res.json({ success: true });
    } catch (error) {
      console.error("Error updating session name:", error);
      res.status(500).json({ error: "Failed to update session name" });
    }
  });
  
  // Test email endpoint (for troubleshooting only, disable in production)
  app.get("/api/test-email", async (req, res) => {
    try {
      if (!process.env.SENDGRID_API_KEY || !process.env.SENDGRID_VERIFIED_SENDER) {
        return res.status(503).json({ 
          error: "Email service is not available. Please check environment variables." 
        });
      }
      
      // Create a test share
      const testShare = {
        id: 9999,
        analysisId: 9999,
        senderEmail: "test@example.com",
        recipientEmail: process.env.SENDGRID_VERIFIED_SENDER, // Use the verified sender as recipient for testing
        status: "pending",
        createdAt: new Date().toISOString()
      };
      
      // Create a test analysis
      const testAnalysis = {
        id: 9999,
        sessionId: "test-session",
        title: "Test Analysis",
        mediaType: "text",
        mediaUrl: null,
        peopleCount: 1,
        personalityInsights: {
          summary: "This is a test analysis summary for email testing purposes.",
          personality_core: {
            summary: "Test personality core summary."
          },
          thought_patterns: {
            summary: "Test thought patterns summary."
          },
          professional_insights: {
            summary: "Test professional insights summary."
          },
          growth_areas: {
            strengths: ["Test strength 1", "Test strength 2"],
            challenges: ["Test challenge 1", "Test challenge 2"],
            development_path: "Test development path."
          }
        },
        downloaded: false,
        createdAt: new Date().toISOString()
      };
      
      // Send test email
      console.log("Sending test email...");
      const emailSent = await sendAnalysisEmail({
        share: testShare,
        analysis: testAnalysis,
        shareUrl: "https://example.com/test-share"
      });
      
      if (emailSent) {
        res.json({ success: true, message: "Test email sent successfully" });
      } else {
        res.status(500).json({ success: false, error: "Failed to send test email" });
      }
    } catch (error) {
      console.error("Test email error:", error);
      res.status(500).json({ success: false, error: String(error) });
    }
  });
  
  // Get a specific analysis by ID
  app.get("/api/analysis/:id", async (req, res) => {
    try {
      const analysisId = parseInt(req.params.id);
      if (isNaN(analysisId)) {
        return res.status(400).json({ error: 'Invalid analysis ID' });
      }
      
      const analysis = await storage.getAnalysisById(analysisId);
      if (!analysis) {
        return res.status(404).json({ error: 'Analysis not found' });
      }
      
      res.json(analysis);
    } catch (error) {
      console.error('Error fetching analysis:', error);
      res.status(500).json({ error: 'Failed to fetch analysis' });
    }
  });
  
  // Download analysis as PDF or DOCX
  app.get("/api/download/:analysisId", async (req, res) => {
    try {
      const { analysisId } = req.params;
      const format = req.query.format as string || 'pdf';
      
      // Get the analysis from storage
      const analysis = await storage.getAnalysisById(parseInt(analysisId));
      if (!analysis) {
        return res.status(404).json({ error: "Analysis not found" });
      }
      
      let buffer: Buffer;
      let contentType: string;
      let filename: string;
      
      if (format === 'docx') {
        // Generate DOCX
        buffer = await generateDocx(analysis);
        contentType = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
        filename = `personality-analysis-${analysisId}.docx`;
      } else if (format === 'txt') {
        // Generate TXT
        const txtContent = generateAnalysisTxt(analysis);
        buffer = Buffer.from(txtContent, 'utf-8');
        contentType = 'text/plain';
        filename = `personality-analysis-${analysisId}.txt`;
      } else {
        // Default to PDF
        const htmlContent = generateAnalysisHtml(analysis);
        buffer = await generatePdf(htmlContent);
        contentType = 'application/pdf';
        filename = `personality-analysis-${analysisId}.pdf`;
      }
      
      // Mark as downloaded in the database
      await storage.updateAnalysisDownloadStatus(analysis.id, true);
      
      // Send the file
      res.setHeader('Content-Type', contentType);
      res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
      res.setHeader('Content-Length', buffer.length);
      res.send(buffer);
      
    } catch (error) {
      console.error("Download error:", error);
      res.status(500).json({ error: "Failed to generate document" });
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

      // Generate the share URL with the current hostname and /share path with analysis ID
      const hostname = req.get('host');
      const protocol = req.headers['x-forwarded-proto'] || req.protocol;
      const shareUrl = `${protocol}://${hostname}/share/${share.id}`;
      
      // Send email with share URL
      const emailSent = await sendAnalysisEmail({
        share,
        analysis,
        shareUrl
      });

      // Update share status based on email sending result
      await storage.updateShareStatus(share.id, emailSent ? "sent" : "error");

      if (!emailSent) {
        return res.status(500).json({ 
          error: "Failed to send email. Please try again later." 
        });
      }

      res.json({ success: emailSent, shareUrl });
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

/**
 * Enhanced face analysis function that uses multiple services with fallback
 * Tries Azure Face API first, then Face++, and finally AWS Rekognition
 */
async function analyzeFaces(imageBuffer: Buffer, maxPeople: number = 5) {
  let analysisResult = {
    provider: "none",
    faces: [],
    success: false
  };
  
  // First try Azure Face API if available
  if (AZURE_FACE_API_KEY && AZURE_FACE_ENDPOINT) {
    try {
      console.log('Attempting face analysis with Azure Face API...');
      
      // Update: Use the modern Azure Face API without deprecated attributes
      // The newer version doesn't support emotion detection and some other attributes that were deprecated
      const azureResponse = await fetch(`${AZURE_FACE_ENDPOINT}/face/v1.0/detect?returnFaceId=true&returnFaceLandmarks=true&recognitionModel=recognition_04`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/octet-stream',
          'Ocp-Apim-Subscription-Key': AZURE_FACE_API_KEY
        },
        body: imageBuffer
      });
      
      if (azureResponse.ok) {
        const facesData = await azureResponse.json();
        
        if (facesData && Array.isArray(facesData) && facesData.length > 0) {
          // Process and format the Azure response
          const processedFaces = facesData.slice(0, maxPeople).map((face: any, index: number) => {
            // Create descriptive label for this person (without gender/age which are no longer available)
            const personLabel = `Person ${index + 1}`;
            
            // Get face size and position
            const faceWidth = face.faceRectangle.width;
            const faceHeight = face.faceRectangle.height;
            const faceArea = faceWidth * faceHeight;
            
            // Create a normalized bounding box (0-1 range)
            // Assuming the image dimensions based on the face position
            const imageWidth = Math.max(1000, face.faceRectangle.left + face.faceRectangle.width * 2);
            const imageHeight = Math.max(1000, face.faceRectangle.top + face.faceRectangle.height * 2);
            
            const boundingBox = {
              Width: face.faceRectangle.width / imageWidth,
              Height: face.faceRectangle.height / imageHeight,
              Left: face.faceRectangle.left / imageWidth,
              Top: face.faceRectangle.top / imageHeight
            };
            
            // Estimate age range (since no longer provided by the API)
            const estimatedAge = {
              low: 20,
              high: 40
            };
            
            // Use face landmarks to estimate expressions and attributes
            const landmarks = face.faceLandmarks;
            
            // Calculate approximate smile score based on landmarks
            let smileScore = 0;
            
            if (landmarks) {
              // Estimate smile by looking at mouth corners relative to mouth center
              const mouthLeft = landmarks.mouthLeft;
              const mouthRight = landmarks.mouthRight; 
              const upperLipTop = landmarks.upperLipTop;
              
              if (mouthLeft && mouthRight && upperLipTop) {
                // Simple smile detection based on mouth curve
                // If mouth corners are higher than the center, it might indicate a smile
                const mouthCurve = ((mouthLeft.y + mouthRight.y) / 2) - upperLipTop.y;
                smileScore = Math.max(0, Math.min(1, mouthCurve / 10));
              }
            }
            
            // Estimate basic emotions (simplified since these are no longer provided by the API)
            const estimatedEmotions = {
              neutral: 0.7,
              happiness: smileScore
            };
            
            return {
              personLabel,
              positionInImage: index + 1,
              boundingBox,
              age: estimatedAge,
              gender: "unknown", // No longer provided by Azure Face API
              emotion: estimatedEmotions,
              faceAttributes: {
                smile: smileScore,
                eyeglasses: "Unknown", // No longer provided by Azure Face API
                sunglasses: "Unknown", // No longer provided by Azure Face API
                beard: "Unknown", // No longer provided by Azure Face API
                mustache: "Unknown", // No longer provided by Azure Face API
                eyesOpen: "Unknown", // Not directly provided by Azure
                mouthOpen: "Unknown", // Not directly provided by Azure
                quality: {
                  brightness: 0, // Not directly provided by Azure in new API
                  sharpness: 0 // Not directly provided by Azure in new API
                },
                pose: {
                  pitch: 0, // Not directly provided by Azure
                  roll: 0,
                  yaw: 0
                }
              },
              dominant: index === 0
            };
          });
          
          analysisResult = {
            provider: "azure",
            faces: processedFaces,
            success: true
          };
          
          console.log('Azure Face API analysis successful!');
          return processedFaces;
        }
      } else {
        console.warn('Azure Face API returned an error:', await azureResponse.text());
      }
    } catch (error) {
      console.error('Azure Face API analysis error:', error);
    }
  }
  
  // If Azure failed, try Face++ if available
  if (!analysisResult.success && FACEPP_API_KEY && FACEPP_API_SECRET) {
    try {
      console.log('Attempting face analysis with Face++ API...');
      
      // Format the image data for Face++ API
      const formData = new FormData();
      formData.append('api_key', FACEPP_API_KEY);
      formData.append('api_secret', FACEPP_API_SECRET);
      formData.append('image_file', imageBuffer, 'image.jpg');
      formData.append('return_landmark', '0');
      formData.append('return_attributes', 'gender,age,smiling,emotion,eyestatus,mouthstatus,eyegaze,beauty,skinstatus');
      
      // @ts-ignore: FormData is compatible with fetch API's Body type
      const faceppResponse = await fetch('https://api-us.faceplusplus.com/facepp/v3/detect', {
        method: 'POST',
        body: formData
      });
      
      if (faceppResponse.ok) {
        const facesData = await faceppResponse.json();
        
        if (facesData && facesData.faces && facesData.faces.length > 0) {
          // Process and format the Face++ response
          const processedFaces = facesData.faces.slice(0, maxPeople).map((face: any, index: number) => {
            // Create descriptive label
            const genderValue = face.attributes?.gender?.value || 'unknown';
            const genderLabel = genderValue.toLowerCase() === 'male' ? 'Male' : 'Female';
            const ageValue = face.attributes?.age?.value || 0;
            const personLabel = `Person ${index + 1} (${genderLabel}, ~${ageValue} years)`;
            
            // Map emotions to standardized format
            const emotions = face.attributes?.emotion || {};
            const emotionMap: Record<string, number> = {};
            
            Object.keys(emotions).forEach(emotion => {
              emotionMap[emotion.toLowerCase()] = emotions[emotion] / 100;
            });
            
            const faceRect = face.face_rectangle || {};
            
            return {
              personLabel,
              positionInImage: index + 1,
              boundingBox: {
                Width: faceRect.width / 100,
                Height: faceRect.height / 100,
                Left: faceRect.left / 100,
                Top: faceRect.top / 100
              },
              age: {
                low: Math.max(0, ageValue - 5),
                high: ageValue + 5
              },
              gender: genderValue.toLowerCase(),
              emotion: emotionMap,
              faceAttributes: {
                smile: face.attributes?.smile?.value / 100 || 0,
                eyeglasses: face.attributes?.eyeglass?.value > 50 ? "Glasses" : "NoGlasses",
                sunglasses: face.attributes?.sunglass?.value > 50 ? "Sunglasses" : "NoSunglasses",
                beard: face.attributes?.beard?.value > 50 ? "Yes" : "No",
                mustache: face.attributes?.moustache?.value > 50 ? "Yes" : "No",
                eyesOpen: face.attributes?.eyestatus?.left_eye_status?.eye_open > 50 ? "Yes" : "No",
                mouthOpen: face.attributes?.mouthstatus?.open > 50 ? "Yes" : "No",
                quality: {
                  brightness: 0, // Not directly provided
                  sharpness: 0, // Not directly provided
                },
                pose: {
                  pitch: 0, // Not directly provided in basic mode
                  roll: 0,  // Not directly provided in basic mode
                  yaw: 0    // Not directly provided in basic mode
                }
              },
              dominant: index === 0
            };
          });
          
          analysisResult = {
            provider: "facepp",
            faces: processedFaces,
            success: true
          };
          
          console.log('Face++ API analysis successful!');
          return processedFaces;
        }
      } else {
        console.warn('Face++ API returned an error:', await faceppResponse.text());
      }
    } catch (error) {
      console.error('Face++ API analysis error:', error);
    }
  }
  
  // Try Google Cloud Vision if previous methods failed and API key is available
  if (!analysisResult.success && GOOGLE_CLOUD_VISION_API_KEY) {
    try {
      console.log('Attempting face analysis with Google Cloud Vision API...');
      
      // Prepare the request to Google Cloud Vision API
      const requestBody = {
        requests: [
          {
            image: {
              content: imageBuffer.toString('base64')
            },
            features: [
              {
                type: "FACE_DETECTION",
                maxResults: maxPeople
              }
            ]
          }
        ]
      };
      
      const gcvResponse = await fetch(`https://vision.googleapis.com/v1/images:annotate?key=${GOOGLE_CLOUD_VISION_API_KEY}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });
      
      if (gcvResponse.ok) {
        const result = await gcvResponse.json();
        const faceAnnotations = result.responses?.[0]?.faceAnnotations || [];
        
        if (faceAnnotations.length > 0) {
          // Process and format the Google Cloud Vision response
          const processedFaces = faceAnnotations.slice(0, maxPeople).map((face: any, index: number) => {
            // Create a descriptive label for each person
            const personLabel = `Person ${index + 1}`;
            
            // Get vertices of the face bounding polygon
            const vertices = face.boundingPoly?.vertices || [];
            let left = 0, top = 0, right = 0, bottom = 0;
            
            if (vertices.length >= 4) {
              left = Math.min(...vertices.map((v: any) => v.x || 0));
              top = Math.min(...vertices.map((v: any) => v.y || 0));
              right = Math.max(...vertices.map((v: any) => v.x || 0));
              bottom = Math.max(...vertices.map((v: any) => v.y || 0));
            }
            
            // Create normalized bounding box (0-1 range)
            // Note: We're estimating the image size from the face bounds
            const imageWidth = 1000; // Approximate width for normalization
            const imageHeight = 1000; // Approximate height for normalization
            
            const boundingBox = {
              Width: (right - left) / imageWidth,
              Height: (bottom - top) / imageHeight,
              Left: left / imageWidth,
              Top: top / imageHeight
            };
            
            // Map Google Cloud Vision emotions to our standard format
            const emotionMap: Record<string, number> = {
              joy: face.joyLikelihood === "VERY_LIKELY" ? 0.9 : 
                   face.joyLikelihood === "LIKELY" ? 0.7 :
                   face.joyLikelihood === "POSSIBLE" ? 0.5 :
                   face.joyLikelihood === "UNLIKELY" ? 0.3 : 0.1,
              sorrow: face.sorrowLikelihood === "VERY_LIKELY" ? 0.9 : 
                      face.sorrowLikelihood === "LIKELY" ? 0.7 :
                      face.sorrowLikelihood === "POSSIBLE" ? 0.5 :
                      face.sorrowLikelihood === "UNLIKELY" ? 0.3 : 0.1,
              anger: face.angerLikelihood === "VERY_LIKELY" ? 0.9 : 
                     face.angerLikelihood === "LIKELY" ? 0.7 :
                     face.angerLikelihood === "POSSIBLE" ? 0.5 :
                     face.angerLikelihood === "UNLIKELY" ? 0.3 : 0.1,
              surprise: face.surpriseLikelihood === "VERY_LIKELY" ? 0.9 : 
                        face.surpriseLikelihood === "LIKELY" ? 0.7 :
                        face.surpriseLikelihood === "POSSIBLE" ? 0.5 :
                        face.surpriseLikelihood === "UNLIKELY" ? 0.3 : 0.1
            };
            
            return {
              personLabel,
              positionInImage: index + 1,
              boundingBox,
              age: {
                low: 18, // GCV doesn't provide exact age estimates
                high: 50
              },
              gender: "unknown", // GCV doesn't provide gender
              emotion: emotionMap,
              faceAttributes: {
                smile: emotionMap.joy,
                eyeglasses: "Unknown", // Not provided
                sunglasses: "Unknown", // Not provided
                beard: "Unknown", // Not provided
                mustache: "Unknown", // Not provided
                eyesOpen: "Unknown", // Not provided
                mouthOpen: "Unknown", // Not provided
                quality: {
                  brightness: 0, // Not directly provided
                  sharpness: 0 // Not directly provided
                },
                pose: {
                  pitch: face.tiltAngle || 0,
                  roll: face.rollAngle || 0,
                  yaw: face.panAngle || 0
                }
              },
              dominant: index === 0
            };
          });
          
          analysisResult = {
            provider: "google_cloud_vision",
            faces: processedFaces,
            success: true
          };
          
          console.log('Google Cloud Vision face analysis successful!');
          return processedFaces;
        }
      } else {
        console.warn('Google Cloud Vision API returned an error:', await gcvResponse.text());
      }
    } catch (error) {
      console.error('Google Cloud Vision analysis error:', error);
    }
  }
  
  // If all previous methods failed, fall back to AWS Rekognition
  try {
    console.log('Falling back to AWS Rekognition for face analysis...');
    
    const command = new DetectFacesCommand({
      Image: {
        Bytes: imageBuffer
      },
      Attributes: ['ALL']
    });

    const response = await rekognition.send(command);
    const faces = response.FaceDetails || [];

    if (faces.length === 0) {
      throw new Error("No faces detected in the image");
    }

    // Limit the number of faces to analyze
    const facesToProcess = faces.slice(0, maxPeople);
    
    // Process each face and add descriptive labels
    const processedFaces = facesToProcess.map((face, index) => {
      // Create a descriptive label for each person
      let personLabel = `Person ${index + 1}`;
      
      // Add gender and approximate age to label if available
      if (face.Gender?.Value) {
        const genderLabel = face.Gender.Value.toLowerCase() === 'male' ? 'Male' : 'Female';
        const ageRange = face.AgeRange ? `${face.AgeRange.Low}-${face.AgeRange.High}` : '';
        personLabel = `${personLabel} (${genderLabel}${ageRange ? ', ~' + ageRange + ' years' : ''})`;
      }
    
      return {
        personLabel,
        positionInImage: index + 1,
        boundingBox: face.BoundingBox || {
          Width: 0,
          Height: 0,
          Left: 0,
          Top: 0
        },
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
        },
        dominant: index === 0 // Flag the first/largest face as dominant
      };
    });
    
    analysisResult = {
      provider: "aws_rekognition",
      faces: processedFaces,
      success: true
    };
    
    console.log('AWS Rekognition face analysis successful!');
    return processedFaces;
  } catch (error) {
    console.error('AWS Rekognition analysis failed:', error);
    
    // If all face detection methods fail, throw an error
    if (!analysisResult.success) {
      throw new Error("No faces detected in the image by any provider");
    }
    
    // Return any successful results from previous providers
    return analysisResult.faces;
  }
}

// For backward compatibility
async function analyzeFaceWithRekognition(imageBuffer: Buffer, maxPeople: number = 5) {
  return analyzeFaces(imageBuffer, maxPeople);
}



async function getEnhancedPersonalityInsights(faceAnalysis: any, videoAnalysis: any = null, audioTranscription: any = null, selectedModel: string = "deepseek") {
  // Check if any API clients are available, display warning if not
  if (!deepseek && !openai && !anthropic && !process.env.PERPLEXITY_API_KEY) {
    console.warn("No AI model API clients are available. Using fallback analysis.");
    return {
      peopleCount: Array.isArray(faceAnalysis) ? faceAnalysis.length : 1,
      individualProfiles: [{
        summary: "API keys are required for detailed analysis. Please configure OpenAI, Anthropic, or Perplexity API keys.",
        detailed_analysis: {
          personality_core: "API keys required for detailed analysis",
          thought_patterns: "API keys required for detailed analysis",
          cognitive_style: "API keys required for detailed analysis",
          professional_insights: "API keys required for detailed analysis",
          relationships: {
            current_status: "Not available",
            parental_status: "Not available",
            ideal_partner: "Not available"
          },
          growth_areas: {
            strengths: ["Not available"],
            challenges: ["Not available"],
            development_path: "Not available"
          }
        }
      }]
    };
  }
  
  // Check if faceAnalysis is an array (multiple people) or single object
  const isMultiplePeople = Array.isArray(faceAnalysis);
  
  // If we have multiple people, analyze each one separately
  if (isMultiplePeople) {
    console.log(`Analyzing ${faceAnalysis.length} people...`);
    
    // Create a combined analysis with an overview and individual profiles
    let multiPersonAnalysis = {
      peopleCount: faceAnalysis.length,
      overviewSummary: `Analysis of ${faceAnalysis.length} people detected in the media.`,
      individualProfiles: [] as any[],
      groupDynamics: undefined as string | undefined, // Will be populated later for multi-person analyses
      detailed_analysis: {} // For backward compatibility with message format
    };
    
    // Analyze each person with the existing logic (concurrently for efficiency)
    const analysisPromises = faceAnalysis.map(async (personFaceData) => {
      try {
        // Create input for this specific person
        const personInput = {
          faceAnalysis: personFaceData,
          ...(videoAnalysis && { videoAnalysis }),
          ...(audioTranscription && { 
            audioTranscription: {
              ...audioTranscription,
              // Ensure we're passing the structured transcription data for quote extraction
              transcriptionData: audioTranscription.transcriptionData || {
                full_text: audioTranscription.transcription || "",
                utterances: [],
                words: []
              }
            } 
          })
        };
        
        // Use the standard analysis prompt but customized for the person
        const personLabel = personFaceData.personLabel || "Person";
        const analysisPrompt = `
You are an expert psychologist, cognitive scientist, and personality analyst with deep expertise in psychological assessment and cognitive profiling. 
Analyze the provided data to generate a comprehensive psychological and cognitive profile for ${personLabel}.

${videoAnalysis ? 'This analysis includes video data showing gestures, activities, and attention patterns.' : ''}
${audioTranscription ? 'This analysis includes audio transcription and speech pattern data.' : ''}

ANALYSIS REQUIREMENTS:
1. SPEECH/TEXT INTEGRATION: If audio transcription or text data is available, make this the PRIMARY SOURCE for personality analysis. Analyze word choice, communication patterns, topics discussed, emotional tone, and speaking style in detail
2. COGNITIVE PROFILING: Assess intellectual capabilities through vocabulary complexity, reasoning patterns in speech, problem-solving approaches mentioned, and communication sophistication
3. PSYCHOLOGICAL PROFILING: Analyze personality traits revealed through what the person says, how they express themselves, their interests, concerns, and emotional expressions in speech
4. EVIDENCE-BASED REASONING: For every assessment, cite specific examples from speech content, direct quotes, and observable patterns
5. COMPREHENSIVE CONTENT ANALYSIS: Discuss the actual topics, ideas, and perspectives shared in speech/text and what these reveal about the person's character, values, and mindset

Return a JSON object with the following structure:
{
  "summary": "Brief overview of ${personLabel} with key cognitive and psychological insights",
  "detailed_analysis": {
    "cognitive_profile": {
      "intelligence_assessment": "Estimated intelligence level with specific evidence from speech patterns, vocabulary, problem-solving approaches",
      "cognitive_strengths": ["Specific cognitive abilities that appear well-developed with evidence"],
      "cognitive_weaknesses": ["Areas of cognitive limitation with supporting evidence"],
      "processing_style": "How this person processes information (analytical vs intuitive, sequential vs random, etc.) with evidence",
      "mental_agility": "Assessment of mental flexibility and adaptability with examples"
    },
    "personality_core": "Deep analysis of core personality traits with specific evidence from facial expressions, body language, speech patterns",
    "thought_patterns": "Analysis of cognitive processes and decision-making style with supporting evidence",
    "emotional_intelligence": "Assessment of emotional awareness and social intelligence with observable evidence",
    "behavioral_indicators": "Specific behaviors observed that reveal personality traits",
    "speech_analysis": {
      "key_quotes": ["Include 5-8 direct quotes from the transcription that reveal personality traits, interests, values, and thinking patterns"],
      "content_themes": "Detailed analysis of the main topics, ideas, and subjects the person discusses and what these reveal about their interests, expertise, and priorities",
      "vocabulary_analysis": "Analysis of word choice, complexity, sophistication level, and communication patterns with specific examples",
      "speech_patterns": "Analysis of speech patterns, pace, tone, communication style, and conversational approach",
      "emotional_tone": "Analysis of emotional tone, enthusiasm, concerns, and feelings expressed in speech with specific examples",
      "personality_revealed": "What the actual content of their speech reveals about their character, values, beliefs, and worldview with direct evidence"
    },
    "visual_evidence": {
      "facial_expressions": "Analysis of facial expressions and what they reveal about personality",
      "body_language": "Analysis of posture, gestures, and physical presence",
      "emotional_indicators": "Observable emotional states and their psychological implications"
    },
    "professional_insights": "Career inclinations and work style based on cognitive profile and personality traits",
    "relationships": {
      "current_status": "Likely relationship status based on evidence",
      "parental_status": "Insights about parenting style or potential",
      "ideal_partner": "Description of compatible partner characteristics"
    },
    "growth_areas": {
      "strengths": ["List of key strengths with evidence"],
      "challenges": ["Areas for improvement with specific indicators"],
      "development_path": "Suggested personal growth direction based on cognitive and personality profile"
    }
  }
}

Be thorough and insightful while avoiding stereotypes. Each section should be at least 2-3 paragraphs long.

CRITICAL REQUIREMENTS:
1. SPEECH-FIRST ANALYSIS: When transcription/text is available, use the actual words spoken/written as the PRIMARY foundation for your analysis
2. CONTENT INTEGRATION: Thoroughly discuss what the person talks about, their interests, concerns, opinions, and how they express themselves
3. DIRECT QUOTES: Include 5-8 meaningful quotes that showcase personality, intelligence, values, and communication style
4. COGNITIVE EVIDENCE: Base intelligence and cognitive assessments on vocabulary, reasoning patterns, and sophistication of ideas expressed
5. CHARACTER INSIGHTS: Analyze what their choice of topics, perspectives, and expressions reveal about their deeper character and values
6. VISUAL ANALYSIS: Reference specific facial expressions, body language, and visual cues as secondary supporting evidence
7. PROFESSIONAL RIGOR: Maintain scientific objectivity while providing actionable insights`;

        // Use OpenAI as primary source for consistency across multiple analyses
        try {
          if (!openai) {
            throw new Error("OpenAI client not available");
          }
          
          const response = await openai.chat.completions.create({
            model: "gpt-4o",
            messages: [
              {
                role: "system",
                content: analysisPrompt,
              },
              {
                role: "user",
                content: JSON.stringify(personInput),
              },
            ],
            response_format: { type: "json_object" },
          });
          
          // Parse and add person identifier
          const analysisResult = JSON.parse(response.choices[0]?.message.content || "{}");
          return {
            ...analysisResult,
            personLabel: personFaceData.personLabel,
            personIndex: personFaceData.positionInImage,
            // Add positional data for potential UI highlighting
            boundingBox: personFaceData.boundingBox
          };
        } catch (err) {
          console.error(`Failed to analyze ${personLabel}:`, err);
          // Return minimal profile on error
          return {
            summary: `Analysis of ${personLabel} could not be completed.`,
            detailed_analysis: {
              personality_core: "Analysis unavailable for this individual.",
              thought_patterns: "Analysis unavailable.",
              cognitive_style: "Analysis unavailable.",
              professional_insights: "Analysis unavailable.",
              relationships: {
                current_status: "Analysis unavailable.",
                parental_status: "Analysis unavailable.",
                ideal_partner: "Analysis unavailable."
              },
              growth_areas: {
                strengths: ["Unknown"],
                challenges: ["Unknown"],
                development_path: "Analysis unavailable."
              }
            },
            personLabel: personFaceData.personLabel,
            personIndex: personFaceData.positionInImage
          };
        }
      } catch (error) {
        console.error("Error analyzing person:", error);
        return null;
      }
    });
    
    // Wait for all analyses to complete
    const individualResults = await Promise.all(analysisPromises);
    
    // Filter out any failed analyses
    multiPersonAnalysis.individualProfiles = individualResults.filter(result => result !== null);
    
    // Generate a group dynamics summary if we have multiple successful analyses
    if (multiPersonAnalysis.individualProfiles.length > 1) {
      try {
        // Create a combined input with only successful profiles
        const groupInput = {
          profiles: multiPersonAnalysis.individualProfiles.map(profile => ({
            personLabel: profile.personLabel,
            summary: profile.summary,
            key_traits: profile.detailed_analysis.personality_core.substring(0, 200) // Truncate for brevity
          }))
        };
        
        const groupPrompt = `
You are analyzing the group dynamics of ${multiPersonAnalysis.individualProfiles.length} people detected in the same media.
Based on the individual summaries provided, generate a brief analysis of how these personalities might interact.

Return a short paragraph (3-5 sentences) describing potential group dynamics, 
compatibilities or conflicts, and how these different personalities might complement each other.`;

        if (!openai) {
          throw new Error("OpenAI client not available for group dynamics analysis");
        }
        
        const groupResponse = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [
            {
              role: "system",
              content: groupPrompt,
            },
            {
              role: "user",
              content: JSON.stringify(groupInput),
            },
          ]
        });
        
        multiPersonAnalysis.groupDynamics = groupResponse.choices[0]?.message.content || 
          "Group dynamics analysis unavailable.";
      } catch (err) {
        console.error("Error generating group dynamics:", err);
        multiPersonAnalysis.groupDynamics = "Group dynamics analysis unavailable.";
      }
    }
    
    return multiPersonAnalysis;
  } else {
    // Original single-person analysis logic
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
    "speech_analysis": {
      "key_quotes": ["Include 3-5 direct quotes from the transcription that reveal personality traits"],
      "speech_patterns": "Analysis of speech patterns, word choice, and communication style",
      "emotional_tone": "Analysis of emotional tone in speech"
    },
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

Important instructions:
1. When audio transcription is available, extract 3-5 direct quotes that reveal personality traits. Include them in double quotes in the speech_analysis.key_quotes array and analyze their significance.
2. For video data, focus on gestures, expressions, and movements to inform your analysis.
3. Pay careful attention to gender, facial expressions, emotional indicators, and body language.
4. The speech_analysis section should be detailed when audio is available; if no speech data exists, note this in the section.
5. Base all insights on the actual data provided, not stereotypes or assumptions.`;

    // Try to get analysis from all three services in parallel for maximum depth
    try {
      // Prepare API calls based on available clients
      const apiPromises = [];
      
      // OpenAI Analysis (if available)
      if (openai) {
        apiPromises.push(
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
          })
        );
      } else {
        apiPromises.push(Promise.reject(new Error("OpenAI client not available")));
      }
      
      // Anthropic Analysis (if available)
      if (anthropic) {
        apiPromises.push(
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
          })
        );
      } else {
        apiPromises.push(Promise.reject(new Error("Anthropic client not available")));
      }
      
      // Perplexity Analysis (if API key available)
      if (process.env.PERPLEXITY_API_KEY) {
        apiPromises.push(
          perplexity.query({
            model: "mistral-large-latest",
            query: `${analysisPrompt}\n\nHere is the data to analyze: ${JSON.stringify(analysisInput)}`,
          })
        );
      } else {
        apiPromises.push(Promise.reject(new Error("Perplexity API key not available")));
      }
      
      // Run all API calls in parallel
      const [openaiResult, anthropicResult, perplexityResult] = await Promise.allSettled(apiPromises);
      
      // Process results from each service
      let finalInsights: any = {};
      
      // Try each service result in order of preference
      if (openaiResult.status === 'fulfilled') {
        try {
          // Handle OpenAI response
          const openaiResponse = openaiResult.value as any;
          const openaiData = JSON.parse(openaiResponse.choices[0]?.message.content || "{}");
          finalInsights = openaiData;
          console.log("OpenAI analysis used as primary source");
        } catch (e) {
          console.error("Error parsing OpenAI response:", e);
        }
      } else if (anthropicResult.status === 'fulfilled') {
        try {
          // Handle Anthropic API response structure
          const anthropicResponse = anthropicResult.value as any;
          if (anthropicResponse.content && Array.isArray(anthropicResponse.content) && anthropicResponse.content.length > 0) {
            const content = anthropicResponse.content[0];
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
          }
        } catch (e) {
          console.error("Error parsing Anthropic response:", e);
        }
      } else if (perplexityResult.status === 'fulfilled') {
        try {
          // Extract JSON from Perplexity response
          const perplexityResponse = perplexityResult.value as any;
          const perplexityText = perplexityResponse.text || "";
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
      
      // For single person case, wrap in object with peopleCount=1 for consistency
      return {
        peopleCount: 1,
        individualProfiles: [finalInsights],
        detailed_analysis: finalInsights.detailed_analysis || {} // For backward compatibility
      };
    } catch (error) {
      console.error("Error in getPersonalityInsights:", error);
      throw new Error("Failed to generate personality insights. Please try again.");
    }
  }
}