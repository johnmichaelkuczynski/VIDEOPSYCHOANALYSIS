import { apiRequest } from "./queryClient";

export async function uploadMedia(
  mediaData: string, 
  mediaType: "image" | "video", 
  sessionId: string,
  maxPeople: number = 5 // Default to 5 people
) {
  const res = await apiRequest("POST", "/api/analyze", { 
    mediaData, 
    mediaType, 
    sessionId,
    maxPeople // Add the maxPeople parameter to support multi-person analysis
  });
  return res.json();
}

export async function sendMessage(content: string, sessionId: string) {
  const res = await apiRequest("POST", "/api/chat", { content, sessionId });
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