import { useState, useRef, useEffect, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { nanoid } from "nanoid";
import { useLocation, useSearch } from "wouter";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import {
  AlertCircle,
  Upload,
  FileText,
  Image,
  Video,
  Send,
  Download,
  Share2,
  MessageCircle,
  X,
  ChevronDown,
  ChevronRight,
  PlayCircle,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { toast } from "@/hooks/use-toast";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { apiRequest } from "@/lib/queryClient";
import type { 
  Message, 
  MediaData, 
  AnalysisResponse, 
  ChatResponse, 
  ServiceStatus,
  ComprehensiveAnalysis,
  CognitiveParameter,
  PsychologicalParameter,
  ProtocolType
} from "@/lib/api";
import { analyzeTextWithProtocols } from "@/lib/api";

// Define parameters data
const cognitiveParameters: CognitiveParameter[] = [
  { id: "verbal_intelligence", name: "Verbal Intelligence", description: "Language processing, vocabulary, comprehension" },
  { id: "logical_reasoning", name: "Logical Reasoning", description: "Problem-solving, analytical thinking" },
  { id: "working_memory", name: "Working Memory", description: "Short-term memory and processing capacity" },
  { id: "attention_focus", name: "Attention & Focus", description: "Concentration abilities and sustained attention" },
  { id: "processing_speed", name: "Processing Speed", description: "Mental agility and information processing rate" },
  { id: "pattern_recognition", name: "Pattern Recognition", description: "Ability to identify patterns and relationships" },
  { id: "abstract_thinking", name: "Abstract Thinking", description: "Conceptual reasoning and theoretical understanding" },
  { id: "cognitive_flexibility", name: "Cognitive Flexibility", description: "Adaptability and mental switching abilities" },
  { id: "executive_function", name: "Executive Function", description: "Planning, organization, and self-control" },
  { id: "creativity", name: "Creativity", description: "Innovative thinking and original idea generation" },
  { id: "meta_cognition", name: "Meta-Cognition", description: "Self-awareness of thinking processes" },
  { id: "decision_making", name: "Decision Making", description: "Judgment and choice evaluation skills" },
  { id: "spatial_intelligence", name: "Spatial Intelligence", description: "Visual-spatial reasoning and mental imagery" },
  { id: "learning_style", name: "Learning Style", description: "Preferred methods of acquiring new information" },
  { id: "information_organization", name: "Information Organization", description: "Mental structuring and categorization abilities" },
  { id: "critical_thinking", name: "Critical Thinking", description: "Analysis, evaluation, and skeptical reasoning" },
  { id: "conceptual_understanding", name: "Conceptual Understanding", description: "Depth of comprehension and meaning-making" },
  { id: "intellectual_curiosity", name: "Intellectual Curiosity", description: "Drive to learn and explore new ideas" },
  { id: "cognitive_style", name: "Cognitive Style", description: "Preferred thinking approaches and mental habits" },
  { id: "mental_agility", name: "Mental Agility", description: "Speed and efficiency of mental operations" }
];

const psychologicalParameters: PsychologicalParameter[] = [
  { id: "emotional_regulation", name: "Emotional Regulation", description: "Management and control of emotions" },
  { id: "interpersonal_skills", name: "Interpersonal Skills", description: "Social interaction and relationship abilities" },
  { id: "self_awareness", name: "Self-Awareness", description: "Understanding of personal traits and behaviors" },
  { id: "motivation_drive", name: "Motivation & Drive", description: "Goal orientation and achievement motivation" },
  { id: "stress_resilience", name: "Stress Resilience", description: "Ability to cope with pressure and adversity" },
  { id: "personality_type", name: "Personality Type", description: "Core personality traits and characteristics" },
  { id: "communication_style", name: "Communication Style", description: "Verbal and non-verbal expression patterns" },
  { id: "leadership_potential", name: "Leadership Potential", description: "Influence and guidance capabilities" },
  { id: "empathy_compassion", name: "Empathy & Compassion", description: "Understanding and caring for others" },
  { id: "conflict_resolution", name: "Conflict Resolution", description: "Ability to manage and resolve disputes" },
  { id: "adaptability", name: "Adaptability", description: "Flexibility in changing situations" },
  { id: "risk_tolerance", name: "Risk Tolerance", description: "Comfort level with uncertainty and risk" },
  { id: "time_management", name: "Time Management", description: "Organization and prioritization of tasks" },
  { id: "team_collaboration", name: "Team Collaboration", description: "Working effectively with others" },
  { id: "ethical_reasoning", name: "Ethical Reasoning", description: "Moral judgment and value-based decision making" },
  { id: "cultural_awareness", name: "Cultural Awareness", description: "Sensitivity to diverse perspectives and backgrounds" },
  { id: "change_management", name: "Change Management", description: "Leading and adapting to organizational changes" },
  { id: "innovation_mindset", name: "Innovation Mindset", description: "Openness to new ideas and creative solutions" },
  { id: "work_life_balance", name: "Work-Life Balance", description: "Integration of personal and professional priorities" },
  { id: "personal_growth", name: "Personal Growth", description: "Commitment to continuous self-improvement" }
];

const shareFormSchema = z.object({
  senderEmail: z.string().email("Invalid email address"),
  recipientEmail: z.string().email("Invalid email address"),
});

export default function HomePage() {
  const [, navigate] = useLocation();
  const search = useSearch();
  const searchParams = new URLSearchParams(search);
  const sessionId = searchParams.get("session") || nanoid();
  const queryClient = useQueryClient();

  // State management
  const [messages, setMessages] = useState<Message[]>([]);
  const [uploadedMedia, setUploadedMedia] = useState<File | null>(null);
  const [mediaData, setMediaData] = useState<MediaData | null>(null);
  const [textInput, setTextInput] = useState("");
  const [additionalInfo, setAdditionalInfo] = useState("");
  const [selectedModel, setSelectedModel] = useState("anthropic");
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [input, setInput] = useState("");
  const [isShareDialogOpen, setIsShareDialogOpen] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [comprehensiveAnalysis, setComprehensiveAnalysis] = useState<ComprehensiveAnalysis | null>(null);
  const [showComprehensiveAnalysis, setShowComprehensiveAnalysis] = useState(false);
  const [expandedCognitiveParams, setExpandedCognitiveParams] = useState<Set<string>>(new Set());
  const [expandedPsychParams, setExpandedPsychParams] = useState<Set<string>>(new Set());
  
  // New 6-Protocol Evaluation System state
  const [selectedProtocols, setSelectedProtocols] = useState<ProtocolType[]>([]);
  const [protocolResults, setProtocolResults] = useState<any>(null);
  const [showProtocolResults, setShowProtocolResults] = useState(false);
  const [analysisMode, setAnalysisMode] = useState<'standard' | 'protocol'>('standard');
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Clear all analysis-related state
  const clearAllAnalysisState = useCallback(() => {
    setMessages([]);
    setUploadedMedia(null);
    setMediaData(null);
    setTextInput("");
    setAdditionalInfo("");
    setAnalysisId(null);
    setAnalysisProgress(0);
    setInput("");
    setComprehensiveAnalysis(null);
    setShowComprehensiveAnalysis(false);
    setExpandedCognitiveParams(new Set());
    setExpandedPsychParams(new Set());
    setIsAnalyzing(false);
    setProtocolResults(null);
    setShowProtocolResults(false);
    setSelectedProtocols([]);
  }, []);

  // API Status Query
  const { data: serviceStatus } = useQuery({
    queryKey: ["/api/status"],
    refetchInterval: 15000,
  }) as { data: ServiceStatus };

  // Share form
  const shareForm = useForm<z.infer<typeof shareFormSchema>>({
    resolver: zodResolver(shareFormSchema),
    defaultValues: {
      senderEmail: "",
      recipientEmail: "",
    },
  });

  // File upload handling
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (file) {
        clearAllAnalysisState();
        setUploadedMedia(file);
        const fileType = file.type.split("/")[0];
        if (fileType === "image" || fileType === "video") {
          handleUploadMedia.mutate(file);
        } else if (file.type === "application/pdf" || 
                   file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
                   file.type === "text/plain") {
          handleUploadDocument.mutate(file);
        } else {
          toast({
            variant: "destructive",
            title: "Unsupported File Type",
            description: "Please upload an image, video, PDF, DOCX, or TXT file.",
          });
        }
      }
    },
    [clearAllAnalysisState]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"],
      "video/*": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
      "text/plain": [".txt"],
    },
    multiple: false,
  });

  // Mutations
  const handleUploadMedia = useMutation({
    mutationFn: async (file: File) => {
      clearAllAnalysisState();
      setIsAnalyzing(true);
      setAnalysisProgress(10);
      
      const formData = new FormData();
      formData.append("media", file);
      formData.append("selectedModel", selectedModel);
      formData.append("sessionId", sessionId);
      
      const response = await fetch("/api/upload/media-multipart", {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Upload failed");
      }
      
      return response.json() as Promise<AnalysisResponse>;
    },
    onSuccess: (data) => {
      setAnalysisProgress(100);
      setMessages(data.messages);
      setAnalysisId(data.analysisId);
      setMediaData(data.mediaData || null);
      setIsAnalyzing(false);
      toast({
        title: "Analysis Complete!",
        description: "Your media has been analyzed successfully.",
      });
    },
    onError: (error) => {
      setIsAnalyzing(false);
      setAnalysisProgress(0);
      toast({
        variant: "destructive",
        title: "Upload Failed",
        description: error.message,
      });
    },
  });

  const handleTextAnalysis = useMutation({
    mutationFn: async () => {
      clearAllAnalysisState();
      setIsAnalyzing(true);
      setAnalysisProgress(10);
      
      const response = await fetch("/api/analyze/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          text: textInput, 
          model: selectedModel, 
          sessionId,
          additionalInfo: additionalInfo || undefined 
        }),
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Analysis failed");
      }
      
      return response.json() as Promise<AnalysisResponse>;
    },
    onSuccess: (data) => {
      setAnalysisProgress(100);
      setMessages(data.messages);
      setAnalysisId(data.analysisId);
      setIsAnalyzing(false);
      toast({
        title: "Analysis Complete!",
        description: "Your text has been analyzed successfully.",
      });
    },
    onError: (error) => {
      setIsAnalyzing(false);
      setAnalysisProgress(0);
      toast({
        variant: "destructive",
        title: "Analysis Failed",
        description: error.message || "Failed to analyze text",
      });
    },
  });

  const chatMutation = useMutation({
    mutationFn: async ({ message }: { message: string }) => {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, sessionId, model: selectedModel }),
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Chat failed");
      }
      
      return response.json() as Promise<ChatResponse>;
    },
    onSuccess: (data) => {
      setMessages(data.messages);
      setInput("");
    },
    onError: (error) => {
      toast({
        variant: "destructive",
        title: "Chat Failed",
        description: error.message || "Failed to send message",
      });
    },
  });

  const shareMutation = useMutation({
    mutationFn: async (data: z.infer<typeof shareFormSchema>) => {
      if (!analysisId) throw new Error("No analysis to share");
      const response = await fetch("/api/share", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...data, analysisId }),
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Share failed");
      }
      
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Analysis Shared!",
        description: "Your analysis has been sent successfully.",
      });
      setIsShareDialogOpen(false);
      shareForm.reset();
    },
    onError: (error) => {
      toast({
        variant: "destructive",
        title: "Share Failed",
        description: error.message || "Failed to share analysis",
      });
    },
  });

  // New 6-Protocol Analysis mutation
  const handleProtocolAnalysis = useMutation({
    mutationFn: async () => {
      if (!selectedProtocols.length) {
        throw new Error("Please select at least one protocol");
      }
      
      clearAllAnalysisState();
      setIsAnalyzing(true);
      setAnalysisProgress(10);
      
      const response = await analyzeTextWithProtocols(
        textInput,
        selectedProtocols,
        sessionId,
        selectedModel,
        undefined,
        additionalInfo
      );
      
      return response;
    },
    onSuccess: (data) => {
      setAnalysisProgress(100);
      setMessages(data.messages);
      setAnalysisId(data.analysisId);
      setProtocolResults(data.protocolResults);
      setShowProtocolResults(true);
      setIsAnalyzing(false);
      toast({
        title: "Protocol Analysis Complete!",
        description: "Your text has been analyzed using the 6-protocol system.",
      });
    },
    onError: (error: Error) => {
      setIsAnalyzing(false);
      setAnalysisProgress(0);
      toast({
        variant: "destructive",
        title: "Protocol Analysis Failed",
        description: error.message,
      });
    },
  });

  // Document Upload mutation
  const handleUploadDocument = useMutation({
    mutationFn: async (file: File) => {
      clearAllAnalysisState();
      setIsAnalyzing(true);
      setAnalysisProgress(10);
      
      const formData = new FormData();
      formData.append("document", file);
      formData.append("sessionId", sessionId);
      
      const response = await fetch("/api/upload/document", {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Document upload failed");
      }
      
      return response.json();
    },
    onSuccess: (data) => {
      setAnalysisProgress(100);
      setMessages(data.messages || []);
      setAnalysisId(data.analysisId);
      setIsAnalyzing(false);
      toast({
        title: "Document Analysis Complete!",
        description: "Your document has been analyzed successfully.",
      });
    },
    onError: (error) => {
      setIsAnalyzing(false);
      setAnalysisProgress(0);
      toast({
        variant: "destructive",
        title: "Document Upload Failed",
        description: error.message,
      });
    },
  });

  // Helper functions
  const clearSession = async (sessionId: string) => {
    try {
      const response = await fetch(`/api/clear-session/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      
      if (!response.ok) {
        console.error("Failed to clear session");
      }
    } catch (error) {
      console.error("Error clearing session:", error);
    }
  };

  const downloadAnalysis = async (id: string, format: "txt") => {
    try {
      const response = await fetch(`/api/download/${id}?format=${format}`);
      if (!response.ok) throw new Error("Download failed");
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.style.display = "none";
      a.href = url;
      a.download = `analysis-${id}.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Download Failed",
        description: "Failed to download analysis",
      });
    }
  };

  const handleChatSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    const newMessages = [...messages, { role: "user" as const, content: input }];
    setMessages(newMessages);
    chatMutation.mutate({ message: input });
  };

  const handleKeyPress = (e: React.KeyboardEvent, submitFn: () => void) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitFn();
    }
  };

  const onShareSubmit = (data: z.infer<typeof shareFormSchema>) => {
    shareMutation.mutate(data);
  };

  const emailServiceAvailable = serviceStatus?.sendgrid;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <div className="container mx-auto px-6 py-8 max-w-7xl">
        <header className="text-center mb-12">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg">
              <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              AI Personality Analysis
            </h1>
          </div>
          <p className="text-slate-600 text-xl font-medium mb-2">Advanced Psychological Insights Platform</p>
          <p className="text-slate-500 text-sm">Clinical-grade analysis with 40+ psychological markers</p>
        </header>
      
      {/* App List */}
      <div className="flex justify-center mb-8">
        <div className="flex space-x-6 text-sm text-muted-foreground">
          <a 
            href="https://mindread.xyz" 
            target="_blank" 
            rel="noopener noreferrer"
            className="hover:text-primary transition-colors"
          >
            MIND READ
          </a>
        </div>
      </div>
      
      <div className="space-y-8 mb-8">
        {/* Input Controls */}
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Model Selector */}
          <Card className="border-0 shadow-xl bg-white/80 backdrop-blur-sm">
            <div className="p-8">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-6 h-6 bg-gradient-to-r from-green-500 to-teal-500 rounded-lg flex items-center justify-center">
                  <span className="text-white text-sm font-bold">1</span>
                </div>
                <h2 className="text-2xl font-semibold text-slate-800">Select AI Model</h2>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { id: "anthropic", name: "🧠 ZHI-1 (Anthropic)", status: serviceStatus?.anthropic },
                  { id: "openai", name: "🚀 GPT-4o", status: serviceStatus?.openai },
                  { id: "deepseek", name: "🔮 DeepSeek", status: serviceStatus?.deepseek },
                  { id: "perplexity", name: "🌟 Perplexity", status: serviceStatus?.perplexity },
                ].map((model) => (
                  <Button
                    key={model.id}
                    variant={selectedModel === model.id ? "default" : "outline"}
                    className={`p-4 h-auto flex flex-col gap-2 transition-all ${
                      selectedModel === model.id 
                        ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg scale-105" 
                        : "hover:scale-102 hover:shadow-md"
                    }`}
                    onClick={() => setSelectedModel(model.id)}
                    disabled={!model.status}
                  >
                    <span className="font-semibold text-sm">{model.name}</span>
                    <Badge 
                      variant={model.status ? "secondary" : "destructive"} 
                      className="text-xs px-2 py-1"
                    >
                      {model.status ? "Online" : "Offline"}
                    </Badge>
                  </Button>
                ))}
              </div>

              {/* Service Status */}
              <div className="mt-6 p-4 bg-gradient-to-r from-gray-50 to-blue-50 rounded-xl">
                <h3 className="font-medium text-gray-800 mb-3">Service Status</h3>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 text-xs">
                  {serviceStatus && Object.entries({
                    "Face Analysis": serviceStatus.aws_rekognition || serviceStatus.facepp || serviceStatus.azure_face,
                    "Audio Processing": serviceStatus.gladia || serviceStatus.assemblyai || serviceStatus.deepgram,
                    "Video Analysis": serviceStatus.azure_video_indexer,
                    "Email Service": serviceStatus.sendgrid,
                  }).map(([service, status]) => (
                    <div key={service} className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${status ? "bg-green-500" : "bg-red-500"}`} />
                      <span>{service}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Card>

          {/* Upload Options */}
          <Card className="border-0 shadow-xl bg-white/80 backdrop-blur-sm">
            <div className="p-8">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-6 h-6 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                  <span className="text-white text-sm font-bold">2</span>
                </div>
                <h2 className="text-2xl font-semibold text-slate-800">Choose Input Type</h2>
              </div>
              
              <div className="grid md:grid-cols-3 gap-6">
                {/* Media Upload */}
                <div className="space-y-4">
                  <h3 className="text-lg font-medium text-slate-700 flex items-center gap-2">
                    <Image className="h-5 w-5" />
                    Upload Media
                  </h3>
                  
                  <div {...getRootProps()}>
                    {isDragActive ? (
                      <div className="border-2 border-dashed border-blue-400 rounded-xl p-8 text-center bg-blue-50 transition-colors">
                        <Upload className="mx-auto h-12 w-12 text-blue-500 mb-4" />
                        <p className="text-blue-700 font-medium">Drop your file here!</p>
                      </div>
                    ) : (
                      <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-blue-400 hover:bg-blue-50/50 transition-all cursor-pointer">
                        <input {...getInputProps()} />
                        <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                        <p className="text-gray-600 font-medium mb-2">
                          Drag & drop or click to upload
                        </p>
                        <p className="text-sm text-gray-500">
                          Images: PNG, JPG, GIF, WebP<br />
                          Videos: MP4, AVI, MOV, WebM<br />
                          Documents: PDF, DOCX, TXT
                        </p>
                      </div>
                    )}
                  </div>
                  
                  {uploadedMedia && (
                    <div className="bg-blue-50 p-4 rounded-lg">
                      <p className="text-sm font-medium text-blue-800">
                        📎 {uploadedMedia.name}
                      </p>
                      <p className="text-xs text-blue-600">
                        {(uploadedMedia.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  )}
                </div>
                
                {/* Text Analysis */}
                <div className="space-y-4">
                  <h3 className="text-lg font-medium text-slate-700 flex items-center gap-2">
                    <FileText className="h-5 w-5" />
                    Text Analysis
                  </h3>
                  
                  <Textarea
                    placeholder="Enter text for personality analysis (letters, essays, social media posts, etc.)"
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    className="min-h-[120px] resize-none border-2 focus:border-blue-400"
                  />
                  
                  <div className="space-y-3">
                    <Dialog>
                      <DialogTrigger asChild>
                        <Button variant="outline" size="sm" className="w-full">
                          Add Context (Optional)
                        </Button>
                      </DialogTrigger>
                      <DialogContent>
                        <DialogHeader>
                          <DialogTitle>Additional Context</DialogTitle>
                        </DialogHeader>
                        <Textarea
                          placeholder="Provide additional context about the person or situation..."
                          value={additionalInfo}
                          onChange={(e) => setAdditionalInfo(e.target.value)}
                          className="min-h-[120px]"
                        />
                        <DialogFooter>
                          <Button onClick={(e: any) => e.target.closest('[role="dialog"]')?.querySelector('button[data-state="closed"]')?.click()} className="w-full">
                            Save Context
                          </Button>
                        </DialogFooter>
                      </DialogContent>
                    </Dialog>
                    
                    <Button 
                      onClick={() => handleTextAnalysis.mutate()}
                      className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700" 
                      disabled={!textInput.trim() || isAnalyzing}
                    >
                      Comprehensive Analysis (40 Parameters)
                    </Button>
                  </div>
                  
                  {additionalInfo && (
                    <div className="bg-blue-50 p-3 rounded-md">
                      <p className="text-sm text-blue-800">
                        <strong>Additional Context:</strong> {additionalInfo.substring(0, 100)}
                        {additionalInfo.length > 100 && "..."}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </Card>

          {/* Progress Bar */}
          {isAnalyzing && (
            <Card className="border-0 shadow-xl bg-white/80 backdrop-blur-sm">
              <div className="p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-6 h-6 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg flex items-center justify-center animate-pulse">
                    <span className="text-white text-sm font-bold">⚡</span>
                  </div>
                  <h3 className="text-lg font-semibold">Analyzing...</h3>
                </div>
                <Progress value={analysisProgress} className="w-full h-3 mb-2" />
                <p className="text-sm text-gray-600 text-center">
                  Processing with {selectedModel === "anthropic" ? "ZHI-1 (Anthropic)" : selectedModel === "openai" ? "GPT-4o" : selectedModel === "deepseek" ? "DeepSeek" : "Perplexity"}...
                </p>
              </div>
            </Card>
          )}
        </div>
        
        {/* Analysis Results - Full Screen Popup */}
        {(messages.length > 0 || showComprehensiveAnalysis) && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl shadow-2xl w-full h-full max-w-7xl max-h-[95vh] flex flex-col">
              {/* Header */}
              <div className="flex justify-between items-center p-6 border-b bg-gradient-to-r from-blue-50 to-indigo-50 rounded-t-2xl">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center">
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <h2 className="text-2xl font-bold text-slate-800">ANALYSIS RESULTS</h2>
                </div>
                <div className="flex items-center gap-2">
                  {/* New Analysis Button */}
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="flex items-center gap-2"
                    onClick={async () => {
                      try {
                        await clearSession(sessionId);
                        const newSessionId = nanoid();
                        window.location.href = `/?session=${newSessionId}`;
                      } catch (error) {
                        console.error("Error clearing session:", error);
                        toast({
                          variant: "destructive",
                          title: "Error",
                          description: "Failed to clear previous analysis. Please refresh the page.",
                        });
                      }
                    }}
                    disabled={isAnalyzing}
                  >
                    <span>New Analysis</span>
                  </Button>
                  
                  {/* Download Button */}
                  {analysisId && (
                    <Button 
                      variant="default" 
                      size="sm" 
                      className="flex items-center gap-2 bg-gray-600 hover:bg-gray-700"
                      onClick={() => {
                        toast({
                          title: "Downloading Text File",
                          description: "Your analysis is being downloaded as TXT"
                        });
                        downloadAnalysis(analysisId, 'txt');
                      }}
                    >
                      <FileText className="h-4 w-4" />
                      <span>Download TXT</span>
                    </Button>
                  )}
                  
                  {/* Close Button */}
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    className="h-8 w-8 p-0"
                    onClick={() => {
                      setMessages([]);
                      setShowComprehensiveAnalysis(false);
                      clearAllAnalysisState();
                    }}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              
              {/* Content Area - Full Height with Scroll */}
              <div className="flex-1 overflow-hidden">
                <div className="h-full flex">
                  {/* Main Analysis Content */}
                  <div className="flex-1 overflow-y-auto p-6">
                    {/* Main Analysis Messages */}
                    {messages.length > 0 && (
                      <div className="space-y-6">
                        {/* Download info message */}
                        {analysisId && (
                          <div className="bg-blue-50 text-blue-800 p-4 rounded-lg mb-6 flex items-center gap-3 text-base">
                            <Download className="h-6 w-6" />
                            <span>
                              <strong>Save your analysis!</strong> Use the download button to save as TXT format.
                            </span>
                          </div>
                        )}
                        
                        {messages.filter(message => message.role === "assistant").map((message, index) => (
                          <div
                            key={index}
                            className="bg-white border border-gray-200 rounded-xl p-8 shadow-sm"
                          >
                            <div 
                              className="whitespace-pre-wrap text-base leading-relaxed text-gray-800"
                              dangerouslySetInnerHTML={{ 
                                __html: message.content
                                  .replace(/\n/g, '<br/>')
                                  .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-gray-900">$1</strong>')
                                  .replace(/^(#+)\s+(.*?)$/gm, (_: string, hashes: string, text: string) => 
                                    `<h${hashes.length} class="font-bold text-xl mt-6 mb-3 text-gray-900">${text}</h${hashes.length}>`)
                                  .replace(/- (.*?)$/gm, '<li class="ml-6 mb-2">• $1</li>')
                              }} 
                            />
                            
                            {/* Download buttons at bottom of each analysis */}
                            {analysisId && index === messages.filter(m => m.role === "assistant").length - 1 && (
                              <div className="flex gap-3 mt-8 pt-6 border-t justify-end">
                                <Button
                                  variant="outline"
                                  size="default"
                                  className="flex items-center gap-2"
                                  onClick={() => {
                                    toast({
                                      title: "Downloading Text File",
                                      description: "Your analysis is being downloaded as TXT"
                                    });
                                    downloadAnalysis(analysisId, 'txt');
                                  }}
                                >
                                  <FileText className="h-4 w-4" />
                                  <span>Save as TXT</span>
                                </Button>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  
                  {/* Chat Sidebar */}
                  {messages.length > 0 && (
                    <div className="w-96 border-l bg-gray-50 flex flex-col">
                      <div className="p-4 border-b bg-white">
                        <div className="flex justify-between items-center">
                          <h3 className="text-lg font-semibold">Discuss Analysis</h3>
                          <Button variant="ghost" size="sm" onClick={() => setMessages(messages.filter(m => m.role === "assistant"))}>
                            Clear Chat
                          </Button>
                        </div>
                      </div>
                      
                      <div className="flex-1 overflow-y-auto p-4">
                        <div className="space-y-4">
                          {messages.filter(m => m.role === "user").length === 0 ? (
                            <div className="text-center text-gray-500 mt-8">
                              <MessageCircle className="h-8 w-8 mx-auto mb-3 text-gray-400" />
                              <p className="text-sm">Ask questions about your analysis</p>
                            </div>
                          ) : (
                            messages.filter(m => m.role === "user" || (m.role === "assistant" && messages.some(msg => msg.role === "user"))).map((message, index) => (
                              <div
                                key={index}
                                className={`p-3 rounded-lg text-sm ${
                                  message.role === "user" ? "bg-blue-100 ml-4" : "bg-white mr-4 shadow-sm"
                                }`}
                              >
                                <div className="font-medium mb-1 text-xs text-gray-600">
                                  {message.role === "user" ? "You" : "AI"}
                                </div>
                                <div 
                                  className="text-gray-800 leading-relaxed"
                                  dangerouslySetInnerHTML={{ 
                                    __html: message.content
                                      .replace(/\n/g, '<br/>')
                                      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                  }} 
                                />
                              </div>
                            ))
                          )}
                        </div>
                      </div>
                      
                      <div className="p-4 border-t bg-white">
                        <form onSubmit={handleChatSubmit}>
                          <div className="flex gap-2">
                            <Textarea
                              value={input}
                              onChange={(e) => setInput(e.target.value)}
                              onKeyDown={(e) => handleKeyPress(e, handleChatSubmit)}
                              placeholder="Ask about your analysis..."
                              className="min-h-[60px] resize-none text-sm"
                            />
                            <Button 
                              type="submit" 
                              size="sm"
                              className="self-end"
                              disabled={!input.trim() || chatMutation.isPending}
                            >
                              <Send className="h-4 w-4" />
                            </Button>
                          </div>
                        </form>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      </div>
    </div>
  );
}