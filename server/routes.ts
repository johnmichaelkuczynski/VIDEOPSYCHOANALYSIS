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
import { SpeechClient } from '@google-cloud/speech';

// the newest OpenAI model is "gpt-4o" which was released May 13, 2024
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

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
      const { imageData, sessionId } = uploadImageSchema.parse(req.body);

      // Extract base64 data
      const base64Data = imageData.replace(/^data:image\/\w+;base64,/, "");
      const imageBuffer = Buffer.from(base64Data, 'base64');

      // Get face analysis from AWS Rekognition
      const faceAnalysis = await analyzeFaceWithRekognition(imageBuffer);

      // Get comprehensive personality insights from OpenAI
      const personalityInsights = await getPersonalityInsights(faceAnalysis);

      const analysis = await storage.createAnalysis({
        sessionId,
        imageUrl: imageData,
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

async function getPersonalityInsights(faceAnalysis: any) {
  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      {
        role: "system",
        content: `You are an expert personality analyst capable of providing deep psychological insights. Analyze the facial features and expressions to generate a comprehensive personality profile. Return a JSON object with the following structure:
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
Important: Pay careful attention to gender, facial expressions, and emotional indicators from the analysis data. Base your insights on the actual facial analysis data provided.`,
      },
      {
        role: "user",
        content: JSON.stringify(faceAnalysis),
      },
    ],
    response_format: { type: "json_object" },
  });

  return JSON.parse(response.choices[0]?.message.content || "{}");
}