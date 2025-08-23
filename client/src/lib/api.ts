import { apiRequest } from "./queryClient";

// Type definitions for enhanced API functionality
export type ModelType = "deepseek" | "openai" | "anthropic" | "perplexity";
export type MediaType = "image" | "video" | "document" | "text";

export async function uploadMedia(options: {
  sessionId: string;
  fileData: string;
  fileName: string;
  fileType: string;
  selectedModel?: ModelType;
  title?: string;
  maxPeople?: number;
  documentType?: "pdf" | "docx" | "other";
  videoSegmentStart?: number;
  videoSegmentDuration?: number;
}) {
  const { 
    sessionId,
    fileData,
    fileName,
    fileType,
    maxPeople = 5, 
    selectedModel = "deepseek", 
    title,
    documentType,
    videoSegmentStart = 0,
    videoSegmentDuration = 5
  } = options;
  
  console.log(`Uploading ${fileType} for analysis with model: ${selectedModel}, sessionId: ${sessionId}`);
  
  const res = await apiRequest("POST", "/api/upload/media", { 
    fileData, 
    fileName,
    fileType,
    sessionId,
    maxPeople,
    selectedModel,
    title,
    documentType,
    videoSegmentStart,
    videoSegmentDuration
  });
  
  const data = await res.json();
  console.log("Media analysis response:", data);
  
  // Extract the analysis text into a proper message format if missing
  if (data.analysisId && (!data.messages || data.messages.length === 0)) {
    if (data.personalityInsights) {
      console.log("Creating message from personality insights");
      let analysisContent = '';
      
      // Try to extract analysis text from different possible formats
      if (typeof data.personalityInsights === 'string') {
        analysisContent = data.personalityInsights;
      } else if (data.personalityInsights.analysis) {
        analysisContent = data.personalityInsights.analysis;
      } else if (data.personalityInsights.individualProfiles && 
                data.personalityInsights.individualProfiles.length > 0) {
        // Format multiple profiles into a single analysis
        const profiles = data.personalityInsights.individualProfiles;
        analysisContent = `# Personality Analysis\n\n`;
        
        profiles.forEach((profile: any, index: number) => {
          const summary = profile.summary || '';
          analysisContent += `## Person ${index + 1}\n\n${summary}\n\n`;
          
          if (profile.detailed_analysis) {
            Object.entries(profile.detailed_analysis).forEach(([key, value]) => {
              analysisContent += `### ${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}\n${value}\n\n`;
            });
          }
        });
      }
      
      if (analysisContent) {
        data.messages = [{
          id: Date.now(),
          analysisId: data.analysisId,
          sessionId,
          role: "assistant",
          content: analysisContent,
          createdAt: new Date().toISOString()
        }];
      }
    }
  }
  
  return data;
}

export async function sendMessage(content: string, sessionId: string, selectedModel: ModelType = "deepseek") {
  const res = await apiRequest("POST", "/api/chat", { 
    content, 
    sessionId,
    selectedModel 
  });
  return res.json();
}

export async function analyzeText(
  text: string, 
  sessionId: string, 
  selectedModel: ModelType = "deepseek",
  title?: string,
  additionalInfo?: string
) {
  console.log(`Analyzing text with model: ${selectedModel}, sessionId: ${sessionId}`);
  
  const res = await apiRequest("POST", "/api/analyze/text", { 
    text, 
    sessionId,
    selectedModel,
    title,
    additionalInfo
  });
  
  const data = await res.json();
  console.log("Text analysis response:", data);
  
  // Extract the analysis text into a proper message format if missing
  if (data.analysisId && (!data.messages || data.messages.length === 0)) {
    if (data.personalityInsights) {
      console.log("Creating message from text analysis insights");
      let analysisContent = '';
      
      // Try to extract analysis text from different possible formats
      if (typeof data.personalityInsights === 'string') {
        analysisContent = data.personalityInsights;
      } else if (data.personalityInsights.analysis) {
        analysisContent = data.personalityInsights.analysis;
      }
      
      if (analysisContent) {
        data.messages = [{
          id: Date.now(),
          analysisId: data.analysisId,
          sessionId,
          role: "assistant",
          content: analysisContent,
          createdAt: new Date().toISOString()
        }];
      }
    }
  }
  
  return data;
}

export async function analyzeDocument(
  fileData: string,
  fileName: string,
  fileType: string,
  sessionId: string,
  selectedModel: ModelType = "deepseek",
  title?: string
) {
  console.log(`Analyzing document with model: ${selectedModel}, sessionId: ${sessionId}`);
  
  const res = await apiRequest("POST", "/api/analyze/document", { 
    fileData,
    fileName,
    fileType,
    sessionId,
    selectedModel,
    title
  });
  
  const data = await res.json();
  console.log("Document analysis response:", data);
  
  return data;
}

export async function analyzeDocumentChunks(
  analysisId: number,
  selectedChunks: number[],
  selectedModel: ModelType = "deepseek"
) {
  console.log(`Analyzing document chunks with model: ${selectedModel}, analysisId: ${analysisId}`);
  
  const res = await apiRequest("POST", "/api/analyze/document-chunks", { 
    analysisId,
    selectedChunks,
    selectedModel
  });
  
  const data = await res.json();
  console.log("Document chunks analysis response:", data);
  
  return data;
}

export async function analyzeVideoSegment(
  analysisId: number,
  segmentId: number,
  selectedModel: ModelType = "deepseek",
  sessionId: string
) {
  console.log(`Analyzing video segment ${segmentId} with model: ${selectedModel}, analysisId: ${analysisId}`);
  
  const res = await apiRequest("POST", "/api/analyze/video-segment", { 
    analysisId,
    segmentId,
    selectedModel,
    sessionId
  });
  
  const data = await res.json();
  console.log("Video segment analysis response:", data);
  
  return data;
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

export async function downloadAnalysis(analysisId: number, format: "pdf" | "docx" | "txt" = "pdf") {
  // Direct download approach using window.open
  window.open(`/api/download/${analysisId}?format=${format}`, '_blank');
  return true;
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