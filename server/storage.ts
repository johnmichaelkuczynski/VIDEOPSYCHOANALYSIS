import { Analysis, InsertAnalysis, Message, InsertMessage, Share, InsertShare } from "@shared/schema";

export interface IStorage {
  createAnalysis(analysis: InsertAnalysis): Promise<Analysis>;
  updateAnalysisPeopleCount(analysisId: number, count: number): Promise<void>;
  getAnalysisBySessionId(sessionId: string): Promise<Analysis | undefined>;
  getAnalysisById(id: number): Promise<Analysis | undefined>;
  createMessage(message: InsertMessage): Promise<Message>;
  getMessagesBySessionId(sessionId: string): Promise<Message[]>;
  createShare(share: InsertShare): Promise<Share>;
  updateShareStatus(shareId: number, status: "pending" | "sent" | "error"): Promise<void>;
}

export class MemStorage implements IStorage {
  private analyses: Map<number, Analysis>;
  private messages: Map<number, Message>;
  private shares: Map<number, Share>;
  private currentAnalysisId: number;
  private currentMessageId: number;
  private currentShareId: number;

  constructor() {
    this.analyses = new Map();
    this.messages = new Map();
    this.shares = new Map();
    this.currentAnalysisId = 1;
    this.currentMessageId = 1;
    this.currentShareId = 1;
  }

  async createAnalysis(insertAnalysis: InsertAnalysis): Promise<Analysis> {
    const id = this.currentAnalysisId++;
    
    // Extract peopleCount from personalityInsights if available
    let peopleCount = 1; // Default to 1 
    try {
      const insights = insertAnalysis.personalityInsights as any;
      if (insights && typeof insights === 'object' && 'peopleCount' in insights) {
        peopleCount = insights.peopleCount || 1;
      }
    } catch (err) {
      console.error("Error extracting peopleCount:", err);
    }
    
    const analysis: Analysis = {
      ...insertAnalysis,
      id,
      createdAt: new Date(),
      // Add these fields with defaults to match the schema
      videoAnalysis: insertAnalysis.videoAnalysis || null,
      audioTranscription: insertAnalysis.audioTranscription || null,
      peopleCount: peopleCount
    };
    
    this.analyses.set(id, analysis);
    return analysis;
  }

  async getAnalysisBySessionId(sessionId: string): Promise<Analysis | undefined> {
    return Array.from(this.analyses.values()).find(
      (analysis) => analysis.sessionId === sessionId
    );
  }

  async getAnalysisById(id: number): Promise<Analysis | undefined> {
    return this.analyses.get(id);
  }
  
  async updateAnalysisPeopleCount(analysisId: number, count: number): Promise<void> {
    const analysis = this.analyses.get(analysisId);
    if (analysis) {
      this.analyses.set(analysisId, { 
        ...analysis, 
        peopleCount: count 
      });
    }
  }

  async createMessage(insertMessage: InsertMessage): Promise<Message> {
    const id = this.currentMessageId++;
    
    // Ensure analysisId is properly handled (null instead of undefined)
    const analysisId = insertMessage.analysisId !== undefined 
      ? insertMessage.analysisId 
      : null;
      
    const message: Message = {
      ...insertMessage,
      id,
      createdAt: new Date(),
      analysisId: analysisId
    };
    
    this.messages.set(id, message);
    return message;
  }

  async getMessagesBySessionId(sessionId: string): Promise<Message[]> {
    return Array.from(this.messages.values())
      .filter((message) => message.sessionId === sessionId)
      .sort((a, b) => (a.createdAt?.getTime() || 0) - (b.createdAt?.getTime() || 0));
  }

  async createShare(insertShare: InsertShare): Promise<Share> {
    const id = this.currentShareId++;
    const share: Share = {
      ...insertShare,
      id,
      status: "pending",
      createdAt: new Date(),
    };
    this.shares.set(id, share);
    return share;
  }

  async updateShareStatus(shareId: number, status: "pending" | "sent" | "error"): Promise<void> {
    const share = this.shares.get(shareId);
    if (share) {
      this.shares.set(shareId, { ...share, status });
    }
  }
}

export const storage = new MemStorage();