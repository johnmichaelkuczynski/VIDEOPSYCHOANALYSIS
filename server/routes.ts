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

      // Get initial personality insights from OpenAI
      const personalityInsights = await getPersonalityInsights(faceAnalysis);

      const analysis = await storage.createAnalysis({
        sessionId,
        imageUrl: imageData,
        faceAnalysis,
        personalityInsights,
      });

      await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: personalityInsights.summary,
        role: "assistant",
      });

      res.json(analysis);
    } catch (error) {
      res.status(400).json({ error: error.message });
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
        content: "Generate a professional personality analysis based on facial features. Avoid stereotypes and bias. Return a JSON object with summary and detailed insights.",
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
