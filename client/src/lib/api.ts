import { apiRequest } from "./queryClient";

// Type definitions for enhanced API functionality
export type ModelType = "openai" | "anthropic" | "perplexity";
export type MediaType = "image" | "video" | "document" | "text";

export async function uploadMedia(
  mediaData: string, 
  mediaType: MediaType, 
  sessionId: string,
  options: {
    maxPeople?: number;
    selectedModel?: ModelType;
    title?: string;
    documentType?: "pdf" | "docx" | "other";
  } = {}
) {
  const { 
    maxPeople = 5, 
    selectedModel = "openai", 
    title,
    documentType
  } = options;
  
  const res = await apiRequest("POST", "/api/analyze", { 
    mediaData, 
    mediaType, 
    sessionId,
    maxPeople,
    selectedModel,
    title,
    documentType
  });
  return res.json();
}

export async function sendMessage(content: string, sessionId: string, selectedModel: ModelType = "openai") {
  const res = await apiRequest("POST", "/api/chat", { 
    content, 
    sessionId,
    selectedModel 
  });
  return res.json();
}

export async function analyzeText(
  content: string, 
  sessionId: string, 
  selectedModel: ModelType = "openai",
  title?: string
) {
  const res = await apiRequest("POST", "/api/analyze/text", { 
    content, 
    sessionId,
    selectedModel,
    title
  });
  return res.json();
}

export async function analyzeDocument(
  fileData: string,
  fileName: string,
  fileType: "pdf" | "docx",
  sessionId: string,
  selectedModel: ModelType = "openai",
  title?: string
) {
  const res = await apiRequest("POST", "/api/analyze/document", { 
    fileData, 
    fileName,
    fileType,
    sessionId,
    selectedModel,
    title
  });
  return res.json();
}

export async function shareAnalysis(analysisId: number, senderEmail: string, recipientEmail: string) {
  const res = await apiRequest("POST", "/api/share", { 
    analysisId,
    senderEmail,
    recipientEmail
  });
  return res.json();
}

export async function getSharedAnalysis(shareId: string) {
  const res = await apiRequest("GET", `/api/shared-analysis/${shareId}`, null);
  return res.json();
}

// Session management functions
export async function getAllSessions() {
  const res = await apiRequest("GET", "/api/sessions", null);
  return res.json();
}

export async function clearSession(sessionId: string) {
  const res = await apiRequest("POST", "/api/session/clear", { sessionId });
  return res.json();
}

export async function updateSessionName(sessionId: string, name: string) {
  const res = await apiRequest("PATCH", "/api/session/name", { sessionId, name });
  return res.json();
}

// Analysis functions
export async function getAllAnalysesBySession(sessionId: string) {
  const res = await apiRequest("GET", `/api/analyses?sessionId=${sessionId}`, null);
  return res.json();
}

export async function downloadAnalysis(analysisId: number, format: "json" | "pdf" | "text" = "pdf", includeCharts: boolean = true) {
  const res = await apiRequest("GET", `/api/analysis/${analysisId}/download?format=${format}&includeCharts=${includeCharts}`, null);
  return res;
}

export async function updateAnalysisTitle(analysisId: number, title: string) {
  const res = await apiRequest("PATCH", `/api/analysis/${analysisId}/title`, { title });
  return res.json();
}

// API status check
export async function checkAPIStatus() {
  const res = await apiRequest("GET", "/api/status", null);
  return res.json();
}