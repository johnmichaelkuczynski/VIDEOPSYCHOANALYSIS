import { apiRequest } from "./queryClient";

export async function uploadImage(imageData: string, sessionId: string) {
  const res = await apiRequest("POST", "/api/analyze", { imageData, sessionId });
  return res.json();
}

export async function sendMessage(content: string, sessionId: string) {
  const res = await apiRequest("POST", "/api/chat", { content, sessionId });
  return res.json();
}
