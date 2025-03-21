import { pgTable, text, serial, integer, boolean, json, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const analyses = pgTable("analyses", {
  id: serial("id").primaryKey(),
  sessionId: text("session_id").notNull(),
  // Store the URL of the media (image or video)
  mediaUrl: text("media_url").notNull(),
  // Indicate whether the media is an image or video
  mediaType: text("media_type", { enum: ["image", "video"] }).notNull(),
  // Store the face analysis data from AWS Rekognition
  faceAnalysis: json("face_analysis").notNull(),
  // For videos, store additional analysis data
  videoAnalysis: json("video_analysis"),
  // For videos, store audio transcription
  audioTranscription: json("audio_transcription"),
  // Store the comprehensive personality insights
  personalityInsights: json("personality_insights").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const messages = pgTable("messages", {
  id: serial("id").primaryKey(),
  sessionId: text("session_id").notNull(),
  analysisId: integer("analysis_id").references(() => analyses.id),
  content: text("content").notNull(),
  role: text("role", { enum: ["user", "assistant"] }).notNull(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const shares = pgTable("shares", {
  id: serial("id").primaryKey(),
  analysisId: integer("analysis_id").references(() => analyses.id).notNull(),
  senderEmail: text("sender_email").notNull(),
  recipientEmail: text("recipient_email").notNull(),
  status: text("status", { enum: ["pending", "sent", "error"] }).notNull().default("pending"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertAnalysisSchema = createInsertSchema(analyses).omit({
  id: true,
  createdAt: true,
  videoAnalysis: true,
  audioTranscription: true,
});

export const insertMessageSchema = createInsertSchema(messages).omit({
  id: true,
  createdAt: true,
});

export const insertShareSchema = createInsertSchema(shares).omit({
  id: true,
  createdAt: true,
  status: true,
});

export type Analysis = typeof analyses.$inferSelect;
export type InsertAnalysis = z.infer<typeof insertAnalysisSchema>;
export type Message = typeof messages.$inferSelect;
export type InsertMessage = z.infer<typeof insertMessageSchema>;
export type Share = typeof shares.$inferSelect;
export type InsertShare = z.infer<typeof insertShareSchema>;

// Schema for validating media uploads
export const uploadMediaSchema = z.object({
  mediaData: z.string(),
  mediaType: z.enum(["image", "video"]),
  sessionId: z.string(),
});