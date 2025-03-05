import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import OpenAI from "openai";
import { insertAnalysisSchema, insertMessageSchema } from "@shared/schema";
import { z } from "zod";

// the newest OpenAI model is "gpt-4o" which was released May 13, 2024
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const uploadImageSchema = z.object({
  imageData: z.string(),
  sessionId: z.string(),
});

const sendMessageSchema = z.object({
  content: z.string(),
  sessionId: z.string(),
});

export async function registerRoutes(app: Express): Promise<Server> {
  app.post("/api/analyze", async (req, res) => {
    try {
      const { imageData, sessionId } = uploadImageSchema.parse(req.body);

      // Get face analysis from Azure
      const faceAnalysis = await analyzeFaceWithAzure(imageData);

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

      res.json(analysis);
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

      const aiResponse = JSON.parse(response.choices[0].message.content);

      const assistantMessage = await storage.createMessage({
        sessionId,
        analysisId: analysis?.id,
        content: aiResponse.response,
        role: "assistant",
      });

      res.json({ messages: [userMessage, assistantMessage] });
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}

async function analyzeFaceWithAzure(imageData: string) {
  // Mock Azure Face API call for now
  return {
    age: 30,
    gender: "female",
    emotion: {
      happiness: 0.8,
      neutral: 0.2,
    },
    faceAttributes: {
      smile: 0.9,
      glasses: "NoGlasses",
      headPose: {
        pitch: 0,
        roll: 0,
        yaw: 0,
      },
    },
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

Be thorough and insightful while avoiding stereotypes. Each section should be at least 2-3 paragraphs long.`,
      },
      {
        role: "user",
        content: JSON.stringify(faceAnalysis),
      },
    ],
    response_format: { type: "json_object" },
  });

  return JSON.parse(response.choices[0].message.content);
}