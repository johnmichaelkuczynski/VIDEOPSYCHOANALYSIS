import { apiRequest } from "./queryClient";

export async function uploadMedia(mediaData: string, mediaType: "image" | "video", sessionId: string) {
  const res = await apiRequest("POST", "/api/analyze", { 
    mediaData, 
    mediaType, 
    sessionId 
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