import { useState, useRef, useCallback } from "react";
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
  PlayCircle,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
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
  ProtocolType
} from "@/lib/api";
import { analyzeTextWithProtocols } from "@/lib/api";

export default function HomePage() {
  // Core state
  const [sessionId] = useState(() => nanoid());
  const [selectedModel, setSelectedModel] = useState("deepseek");
  const [selectedProtocol, setSelectedProtocol] = useState<ProtocolType>('cognitive');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [messages, setMessages] = useState<Message[]>([]);
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [mediaData, setMediaData] = useState<MediaData | null>(null);
  const [uploadedMedia, setUploadedMedia] = useState<File | null>(null);
  const [textInput, setTextInput] = useState("");
  const [additionalInfo, setAdditionalInfo] = useState("");
  const [input, setInput] = useState("");
  const [showChatModal, setShowChatModal] = useState(false);
  const [shareCode, setShareCode] = useState<string | null>(null);

  const queryClient = useQueryClient();
  const [location, setLocation] = useLocation();
  const searchQuery = useSearch();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Clear all analysis-related state
  const clearAllAnalysisState = useCallback(() => {
    setMessages([]);
    setAnalysisId(null);
    setMediaData(null);
    setUploadedMedia(null);
    setAnalysisProgress(0);
    setInput("");
    setIsAnalyzing(false);
  }, []);

  // API Status Query
  const { data: serviceStatus } = useQuery({
    queryKey: ['/api/status'],
    refetchInterval: 15000,
  });

  // Upload file handler
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (file) {
        clearAllAnalysisState();
        setUploadedMedia(file);
        const fileType = file.type.split("/")[0];
        if (fileType === "image" || fileType === "video") {
          handleUploadMediaWithProtocol.mutate(file);
        } else if (file.type === "application/pdf" || 
                   file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
                   file.type === "text/plain") {
          handleUploadDocumentWithProtocol.mutate(file);
        } else {
          toast({
            variant: "destructive",
            title: "Unsupported File Type",
            description: "Please upload an image, video, PDF, DOCX, or TXT file.",
          });
        }
      }
    },
    [clearAllAnalysisState, selectedProtocol]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".png", ".jpg", ".jpeg", ".gif", ".webp"],
      "video/*": [".mp4", ".mov", ".avi"],
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
      "text/plain": [".txt"]
    },
    multiple: false,
  });

  // Protocol-based upload mutations
  const handleUploadMediaWithProtocol = useMutation({
    mutationFn: async (file: File) => {
      clearAllAnalysisState();
      setIsAnalyzing(true);
      setAnalysisProgress(10);
      
      const formData = new FormData();
      formData.append("media", file);
      formData.append("selectedModel", selectedModel);
      formData.append("sessionId", sessionId);
      formData.append("protocol", selectedProtocol);
      
      const response = await fetch("/api/upload/media-protocol", {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Upload failed");
      }
      
      return response.json();
    },
    onSuccess: (data) => {
      setAnalysisProgress(100);
      setMessages(data.messages);
      setAnalysisId(data.analysisId);
      setMediaData(data.mediaData || null);
      setIsAnalyzing(false);
      toast({
        title: "Analysis Complete!",
        description: `${selectedProtocol.replace('-', ' ')} analysis completed successfully.`,
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

  const handleUploadDocumentWithProtocol = useMutation({
    mutationFn: async (file: File) => {
      clearAllAnalysisState();
      setIsAnalyzing(true);
      setAnalysisProgress(10);
      
      const formData = new FormData();
      formData.append("document", file);
      formData.append("selectedModel", selectedModel);
      formData.append("sessionId", sessionId);
      formData.append("protocol", selectedProtocol);
      
      const response = await fetch("/api/upload/document-protocol", {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Upload failed");
      }
      
      return response.json();
    },
    onSuccess: (data) => {
      setAnalysisProgress(100);
      setMessages(data.messages);
      setAnalysisId(data.analysisId);
      setIsAnalyzing(false);
      toast({
        title: "Analysis Complete!",
        description: `${selectedProtocol.replace('-', ' ')} document analysis completed successfully.`,
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

  const handleTextAnalysisWithProtocol = useMutation({
    mutationFn: async () => {
      clearAllAnalysisState();
      setIsAnalyzing(true);
      setAnalysisProgress(10);
      
      const response = await analyzeTextWithProtocols(
        textInput,
        [selectedProtocol],
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
      setIsAnalyzing(false);
      toast({
        title: "Analysis Complete!",
        description: `${selectedProtocol.replace('-', ' ')} text analysis completed successfully.`,
      });
    },
    onError: (error: Error) => {
      setIsAnalyzing(false);
      setAnalysisProgress(0);
      toast({
        variant: "destructive",
        title: "Analysis Failed",
        description: error.message,
      });
    },
  });

  // Download handler
  const handleDownload = (format: "pdf" | "docx" | "txt") => {
    if (!analysisId) return;
    
    const url = `/api/download/${analysisId}?format=${format}`;
    const link = document.createElement("a");
    link.href = url;
    link.download = `analysis-${analysisId}.${format}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <header className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-full flex items-center justify-center">
              <FileText className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              AI Personality Analysis
            </h1>
          </div>
          <p className="text-slate-600 text-xl font-medium mb-2">Advanced Psychological Insights Platform</p>
          <p className="text-slate-500 text-sm">6-Protocol Evaluation System</p>
          
          {/* Protocol Selection Toggle */}
          <div className="mt-6">
            <p className="text-slate-700 font-medium mb-3">Select Analysis Protocol:</p>
            <div className="flex flex-wrap justify-center gap-2">
              {[
                { value: 'cognitive' as ProtocolType, label: 'Cognitive' },
                { value: 'psychological' as ProtocolType, label: 'Psychological' },
                { value: 'psychopathological' as ProtocolType, label: 'Psychopathological' },
                { value: 'comprehensive-cognitive' as ProtocolType, label: 'Comprehensive Cognitive' },
                { value: 'comprehensive-psychological' as ProtocolType, label: 'Comprehensive Psychological' },
                { value: 'comprehensive-psychopathological' as ProtocolType, label: 'Comprehensive Psychopathological' },
              ].map((protocol) => (
                <button
                  key={protocol.value}
                  onClick={() => setSelectedProtocol(protocol.value)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    selectedProtocol === protocol.value
                      ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md'
                      : 'bg-white text-slate-700 border border-slate-300 hover:border-blue-400 hover:bg-blue-50'
                  }`}
                >
                  {protocol.label}
                </button>
              ))}
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="space-y-6">
            {/* File Upload */}
            <Card className="p-6 bg-white/80 backdrop-blur-sm border border-slate-200 shadow-lg">
              <h2 className="text-2xl font-bold mb-4 text-slate-800">Upload Media or Document</h2>
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive
                    ? "border-blue-400 bg-blue-50"
                    : "border-slate-300 hover:border-blue-400 hover:bg-slate-50"
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="mx-auto h-12 w-12 text-slate-400 mb-4" />
                <p className="text-lg font-medium text-slate-700 mb-2">
                  {isDragActive ? "Drop your file here..." : "Drag & drop a file here"}
                </p>
                <p className="text-sm text-slate-500 mb-4">
                  Supports images, videos, PDFs, Word docs, and text files
                </p>
                <Button variant="outline" className="mt-2">
                  Choose File
                </Button>
              </div>
            </Card>

            {/* Text Input */}
            <Card className="p-6 bg-white/80 backdrop-blur-sm border border-slate-200 shadow-lg">
              <h2 className="text-2xl font-bold mb-4 text-slate-800">Or Analyze Text</h2>
              <div className="space-y-4">
                <Textarea
                  placeholder="Paste your text here for analysis..."
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  rows={6}
                  className="resize-none"
                />
                <Textarea
                  placeholder="Additional context or information (optional)"
                  value={additionalInfo}
                  onChange={(e) => setAdditionalInfo(e.target.value)}
                  rows={2}
                  className="resize-none"
                />
                <Button
                  onClick={() => handleTextAnalysisWithProtocol.mutate()}
                  disabled={!textInput.trim() || handleTextAnalysisWithProtocol.isPending}
                  className="w-full"
                >
                  <Send className="w-4 h-4 mr-2" />
                  Analyze Text with {selectedProtocol.replace('-', ' ').toUpperCase()} Protocol
                </Button>
              </div>
            </Card>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {/* Analysis Progress */}
            {isAnalyzing && (
              <Card className="p-6 bg-white/80 backdrop-blur-sm border border-slate-200 shadow-lg">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
                  <h3 className="text-lg font-semibold">
                    Running {selectedProtocol.replace('-', ' ').toUpperCase()} Analysis...
                  </h3>
                </div>
                <Progress value={analysisProgress} className="mb-2" />
                <p className="text-sm text-slate-600">
                  This may take a few minutes. Please wait...
                </p>
              </Card>
            )}

            {/* Analysis Results */}
            {messages.length > 0 && !isAnalyzing && (
              <Card className="p-6 bg-white/80 backdrop-blur-sm border border-slate-200 shadow-lg">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-2xl font-bold text-slate-800">
                    {selectedProtocol.replace('-', ' ').toUpperCase()} Analysis Results
                  </h3>
                  {analysisId && (
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleDownload("pdf")}
                      >
                        <Download className="w-4 h-4 mr-1" />
                        PDF
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleDownload("docx")}
                      >
                        <Download className="w-4 h-4 mr-1" />
                        Word
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleDownload("txt")}
                      >
                        <Download className="w-4 h-4 mr-1" />
                        TXT
                      </Button>
                    </div>
                  )}
                </div>
                
                <ScrollArea className="h-96 w-full">
                  <div className="space-y-4">
                    {messages.map((message, index) => (
                      <div
                        key={index}
                        className={`p-4 rounded-lg ${
                          message.role === "assistant"
                            ? "bg-blue-50 border-l-4 border-blue-400"
                            : "bg-slate-50 border-l-4 border-slate-400"
                        }`}
                      >
                        <div className="whitespace-pre-wrap text-slate-700">
                          {message.content}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div ref={messagesEndRef} />
                </ScrollArea>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}