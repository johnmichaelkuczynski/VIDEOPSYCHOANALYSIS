import { useState, useCallback, useRef, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { nanoid } from "nanoid";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogTrigger } from "@/components/ui/dialog";
import { Form, FormField, FormItem, FormLabel, FormControl, FormMessage } from "@/components/ui/form";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { uploadMedia, sendMessage, shareAnalysis, getSharedAnalysis, analyzeText, analyzeDocument, analyzeDocumentChunks, analyzeVideoSegment, downloadAnalysis, clearSession, ModelType, MediaType } from "@/lib/api";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Upload, Send, FileImage, Film, Share2, AlertCircle, FileText, File, Download, Check, Eye, RefreshCw } from "lucide-react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";

// Define schemas for forms
const shareSchema = z.object({
  senderEmail: z.string().email("Please enter a valid email"),
  recipientEmail: z.string().email("Please enter a valid email"),
});

// Helper function to resize images
async function resizeImage(file: File, maxWidth: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (readerEvent) => {
      const img = new Image();
      img.onload = () => {
        // Calculate new dimensions while maintaining aspect ratio
        let width = img.width;
        let height = img.height;
        
        if (width > maxWidth) {
          height = Math.round((height * maxWidth) / width);
          width = maxWidth;
        }
        
        // Create canvas for resizing
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        
        // Draw and resize image on canvas
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error('Could not get canvas context'));
          return;
        }
        
        ctx.drawImage(img, 0, 0, width, height);
        
        // Convert canvas to data URL
        try {
          const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
          resolve(dataUrl);
        } catch (e) {
          reject(e);
        }
      };
      
      img.onerror = () => {
        reject(new Error('Failed to load image'));
      };
      
      if (typeof readerEvent.target?.result === 'string') {
        img.src = readerEvent.target.result;
      } else {
        reject(new Error('Failed to read file'));
      }
    };
    
    reader.onerror = () => {
      reject(new Error('Failed to read file'));
    };
    
    reader.readAsDataURL(file);
  });
}

// Helper function to get model display name
const getModelDisplayName = (model: string) => {
  switch (model) {
    case "deepseek": return "ZHI 3";
    case "openai": return "ZHI 2";
    case "anthropic": return "ZHI 1";
    case "perplexity": return "ZHI 4";
    default: return model;
  }
};

export default function Home({ isShareMode = false, shareId }: { isShareMode?: boolean, shareId?: string }) {
  const { toast } = useToast();
  const [sessionId] = useState(() => nanoid());
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState("");
  const [textInput, setTextInput] = useState("");
  const queryClient = useQueryClient();
  
  // Media states
  const [uploadedMedia, setUploadedMedia] = useState<string | null>(null);
  const [mediaType, setMediaType] = useState<MediaType>("image");
  const [mediaData, setMediaData] = useState<string | null>(null); // Store media data for re-analysis
  const [analysisId, setAnalysisId] = useState<number | null>(null);
  const [isShareDialogOpen, setIsShareDialogOpen] = useState(false);
  const [emailServiceAvailable, setEmailServiceAvailable] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ModelType>("anthropic");
  
  // Document analysis states
  const [documentChunks, setDocumentChunks] = useState<any[]>([]);
  const [selectedChunks, setSelectedChunks] = useState<number[]>([]);
  const [documentFileName, setDocumentFileName] = useState<string>("");
  const [documentFileType, setDocumentFileType] = useState<string>("");
  const [metricsAnalysis, setMetricsAnalysis] = useState<any>(null);
  const [expandedMetrics, setExpandedMetrics] = useState<Set<number>>(new Set());
  const [showChunkSelection, setShowChunkSelection] = useState(false);
  
  // Video segment states
  const [videoSegmentStart, setVideoSegmentStart] = useState<number>(0);
  const [videoSegmentDuration, setVideoSegmentDuration] = useState<number>(5);
  const [videoDuration, setVideoDuration] = useState<number>(0);
  const [videoSegments, setVideoSegments] = useState<any[]>([]);
  const [selectedVideoSegment, setSelectedVideoSegment] = useState<number | null>(null);
  const [requiresSegmentSelection, setRequiresSegmentSelection] = useState<boolean>(false);
  
  // Comprehensive text analysis states
  const [comprehensiveAnalysis, setComprehensiveAnalysis] = useState<any>(null);
  const [cognitiveParameters, setCognitiveParameters] = useState<any[]>([]);
  const [psychologicalParameters, setPsychologicalParameters] = useState<any[]>([]);
  const [expandedCognitiveParams, setExpandedCognitiveParams] = useState<Set<number>>(new Set());
  const [expandedPsychParams, setExpandedPsychParams] = useState<Set<number>>(new Set());
  const [additionalInfo, setAdditionalInfo] = useState<string>("");
  const [showAdditionalInfoDialog, setShowAdditionalInfoDialog] = useState<boolean>(false);
  const [showComprehensiveAnalysis, setShowComprehensiveAnalysis] = useState<boolean>(false);
  
  // UI states
  const [showAdvancedServices, setShowAdvancedServices] = useState<boolean>(false);
  
  // References
  const videoRef = useRef<HTMLVideoElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const documentInputRef = useRef<HTMLInputElement>(null);

  // Available services state
  const [availableServices, setAvailableServices] = useState<{
    openai: boolean;
    anthropic: boolean;
    perplexity: boolean;
    azure_face: boolean;
    facepp: boolean;
    google_vision: boolean;
    aws_rekognition: boolean;
    gladia: boolean;
    assemblyai: boolean;
    deepgram: boolean;
    azure_video_indexer: boolean;
    deepseek: boolean;
  }>({
    deepseek: true,
    openai: false,
    anthropic: false,
    perplexity: false,
    azure_face: false,
    facepp: false,
    google_vision: false,
    aws_rekognition: false,
    gladia: false,
    assemblyai: false,
    deepgram: false,
    azure_video_indexer: false
  });
  
  // Check API status on component mount
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch('/api/status');
        const status = await res.json();
        console.log("API Status:", status);
        
        // Set email service availability
        setEmailServiceAvailable(status.sendgrid || false);
        
        // Set available services
        setAvailableServices({
          deepseek: true, // DeepSeek is always available as default
          openai: status.openai || false,
          anthropic: status.anthropic || false,
          perplexity: status.perplexity || false,
          azure_face: status.azure_face || false,
          facepp: status.facepp || false,
          google_vision: status.google_vision || false,
          aws_rekognition: status.aws_rekognition || false,
          gladia: status.gladia || false,
          assemblyai: status.assemblyai || false,
          deepgram: status.deepgram || false,
          azure_video_indexer: status.azure_video_indexer || false
        });
        
        // Set default selected model based on availability
        // Anthropic (ZHI 1) is the default
        setSelectedModel("anthropic");
      } catch (error) {
        console.error("Error checking API status:", error);
      }
    };
    
    checkStatus();
  }, []);

  // Load shared analysis when shareId is provided
  useEffect(() => {
    if (shareId) {
      // Fetch and display the shared analysis
      getSharedAnalysis(shareId)
        .then(data => {
          if (data.analysis && data.messages) {
            // Set the analysis data
            setAnalysisId(data.analysis.id);
            
            // Set uploaded media preview if available
            if (data.analysis.mediaUrl) {
              setUploadedMedia(data.analysis.mediaUrl);
              setMediaType(data.analysis.mediaType as MediaType);
            }
            
            // Set messages
            setMessages(data.messages);
            
            // Set email service status
            setEmailServiceAvailable(data.emailServiceAvailable);
            
            toast({
              title: "Shared Analysis Loaded",
              description: "Viewing a shared personality analysis."
            });
          }
        })
        .catch(error => {
          console.error("Error loading shared analysis:", error);
          toast({
            variant: "destructive",
            title: "Error",
            description: "Failed to load shared analysis. It may have expired or been removed."
          });
        });
    }
  }, [shareId, toast]);
  
  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);
  
  // Text analysis
  const handleTextAnalysis = useMutation({
    mutationFn: async (text: string) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        setMessages([]);
        
        const response = await analyzeText(text, sessionId, selectedModel, undefined, additionalInfo);
        
        setAnalysisProgress(80);
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(response.messages);
        }
        
        // Store comprehensive analysis data if available
        if (response.comprehensiveAnalysis) {
          setComprehensiveAnalysis(response.comprehensiveAnalysis);
          setShowComprehensiveAnalysis(true);
        }
        
        if (response.cognitiveParameters) {
          setCognitiveParameters(response.cognitiveParameters);
        }
        
        if (response.psychologicalParameters) {
          setPsychologicalParameters(response.psychologicalParameters);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Text analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze text. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: (data) => {
      // Get all messages for the session to be sure we have the latest
      if (data?.analysisId) {
        // If we received an analysis ID, fetch any messages related to it
        fetch(`/api/messages?sessionId=${sessionId}`)
          .then(res => res.json())
          .then(data => {
            if (data && Array.isArray(data) && data.length > 0) {
              console.log("Fetched messages after text analysis:", data);
              setMessages(data);
            }
          })
          .catch(err => console.error("Error fetching messages after text analysis:", err));
      }
      
      toast({
        title: "Comprehensive Analysis Complete",
        description: "Your text has been analyzed across 40 psychological and cognitive parameters.",
      });
      setTextInput("");
    }
  });

  // Document analysis removed

  // Media upload and analysis
  const handleUploadMedia = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(0);
        setMessages([]);
        
        // Determine media type and set it
        const fileType = file.type.split('/')[0];
        const isVideo = fileType === 'video';
        const mediaFileType: MediaType = isVideo ? "video" : "image";
        setMediaType(mediaFileType);
        
        // Update progress
        setAnalysisProgress(20);
        
        // Show appropriate progress message for video vs image
        if (isVideo) {
          toast({
            title: "Processing Video",
            description: "Extracting segment and analyzing content...",
          });
        }
        
        // For images, resize if needed to meet AWS limits
        let mediaData: string;
        if (mediaFileType === "image" && file.size > 4 * 1024 * 1024) {
          mediaData = await resizeImage(file, 1600);
        } else {
          // For videos or smaller images, read as data URL
          const reader = new FileReader();
          mediaData = await new Promise<string>((resolve) => {
            reader.onload = (e) => resolve(e.target?.result as string);
            reader.readAsDataURL(file);
          });
          
          // For videos, get duration for segment selection and enforce size limits
          if (isVideo) {
            // Check file size - if over 15MB, we'll get segment selection
            if (file.size > 15 * 1024 * 1024) {
              console.log(`Large video file detected: ${(file.size / (1024 * 1024)).toFixed(2)} MB`);
            }
            
            const videoElement = document.createElement('video');
            await new Promise<void>((resolve) => {
              videoElement.onloadedmetadata = () => {
                setVideoDuration(videoElement.duration);
                resolve();
              };
              videoElement.src = mediaData;
            });
          }
        }
        
        // Set preview and store media data for re-analysis
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setAnalysisProgress(50);
        
        // Maximum 5 people to analyze
        const maxPeople = 5;
        
        // Upload for analysis
        const options = { 
          selectedModel, 
          maxPeople,
          ...(isVideo && { videoSegmentStart, videoSegmentDuration })
        };
        
        console.log(`Starting ${isVideo ? 'video segment' : 'image'} analysis:`, options);
        
        let response;
        
        // Use multipart upload for all videos (to avoid base64 encoding overhead)
        if (isVideo) {
          console.log(`Large video file detected: ${(file.size / 1024 / 1024).toFixed(2)} MB`);
          console.log("Starting video segment analysis:", { selectedModel, maxPeople: 5, videoSegmentStart: 0, videoSegmentDuration: 5 });
          console.log(`Uploading ${file.type} for analysis with model: ${selectedModel}, sessionId: ${sessionId}`);
          
          // Use FormData for large file uploads
          const formData = new FormData();
          formData.append('media', file);
          formData.append('sessionId', sessionId);
          formData.append('selectedModel', selectedModel);
          formData.append('title', `${mediaFileType} Analysis`);
          
          const uploadResponse = await fetch('/api/upload/media-multipart', {
            method: 'POST',
            body: formData
          });
          
          if (!uploadResponse.ok) {
            const errorData = await uploadResponse.json();
            throw new Error(errorData.error || 'Upload failed');
          }
          
          response = await uploadResponse.json();
        } else {
          // Use regular JSON upload for smaller files
          response = await uploadMedia({
            sessionId,
            fileData: mediaData,
            fileName: file.name,
            fileType: file.type,
            selectedModel,
            title: `${mediaFileType} Analysis`
          });
        }
        
        setAnalysisProgress(90);
        
        console.log("Response from uploadMedia:", response);
        
        // Handle different response types
        if (response.requiresSegmentSelection) {
          // Video is too large, show segment selection
          setRequiresSegmentSelection(true);
          setVideoSegments(response.segments || []);
          setVideoDuration(response.duration || 0);
          setAnalysisId(response.analysisId);
          
          toast({
            title: "Video Uploaded",
            description: "Please select a 5-second segment to analyze.",
          });
        } else if (response && response.analysisId) {
          setAnalysisId(response.analysisId);
          
          // Make sure we update the messages state with the response
          if (response && response.messages && Array.isArray(response.messages) && response.messages.length > 0) {
            console.log("Setting messages from response:", response.messages);
            setMessages(response.messages);
          } else {
            // If no messages were returned, let's add a default message
            console.warn("No messages returned from analysis");
            if (response?.analysisInsights) {
              setMessages([{
                role: "assistant",
                content: response.analysisInsights,
                id: Date.now(),
                createdAt: new Date().toISOString(),
                sessionId,
                analysisId: response.analysisId
              }]);
            }
          }
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Upload error:', error);
        toast({
          title: "Upload Failed",
          description: error.message || "Failed to upload media. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: (data) => {
      // Get all messages for the session to be sure we have the latest
      if (data?.analysisId) {
        // If we received an analysis ID, try to fetch any messages related to it
        fetch(`/api/messages?sessionId=${sessionId}`)
          .then(res => res.json())
          .then(data => {
            if (data && Array.isArray(data) && data.length > 0) {
              console.log("Fetched messages after analysis:", data);
              setMessages(data);
            }
          })
          .catch(err => console.error("Error fetching messages:", err));
      }
      
      toast({
        title: "Analysis Complete",
        description: "Your media has been successfully analyzed.",
      });
    },
    onError: () => {
      setUploadedMedia(null);
    }
  });

  // Chat with AI
  const chatMutation = useMutation({
    mutationFn: async (content: string) => {
      return sendMessage(content, sessionId, selectedModel);
    },
    onSuccess: (data) => {
      if (data && data.messages && Array.isArray(data.messages)) {
        // Add the new messages
        setMessages((prev) => [...prev, ...data.messages]);
        queryClient.invalidateQueries({ queryKey: ["/api/chat"] });
      }
    },
    onError: (error: any) => {
      console.error("Chat error:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: error.message || "Failed to send message.",
      });
    },
  });

  // Video segment analysis function
  const handleAnalyzeVideoSegment = async () => {
    if (!selectedVideoSegment || !analysisId || !mediaData) return;
    
    try {
      setIsAnalyzing(true);
      setAnalysisProgress(10);
      
      const response = await fetch('/api/analyze/video-segment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          analysisId,
          segmentId: selectedVideoSegment,
          selectedModel,
          sessionId
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to analyze video segment');
      }
      
      const data = await response.json();
      setAnalysisProgress(90);
      
      // Add the analysis message to the chat, replacing any existing video analysis
      if (data.message) {
        setMessages(prev => {
          const filteredMessages = prev.filter(msg => 
            !msg.content.includes("Video Segment Analysis") || 
            msg.role !== "assistant"
          );
          return [...filteredMessages, data.message];
        });
      }
      
      // Hide segment selection and show results
      setRequiresSegmentSelection(false);
      
      setAnalysisProgress(100);
      
      toast({
        title: "Segment Analysis Complete",
        description: "Your video segment has been successfully analyzed!",
      });
      
    } catch (error: any) {
      console.error('Video segment analysis error:', error);
      toast({
        title: "Analysis Failed",
        description: error.message || "Failed to analyze video segment. Please try again.",
        variant: "destructive",
      });
      setAnalysisProgress(0);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Email sharing
  const shareForm = useForm<z.infer<typeof shareSchema>>({
    resolver: zodResolver(shareSchema),
  });
  
  const shareMutation = useMutation({
    mutationFn: async (data: z.infer<typeof shareSchema>) => {
      if (!analysisId) throw new Error("No analysis to share");
      return shareAnalysis(analysisId, data.senderEmail, data.recipientEmail);
    },
    onSuccess: () => {
      toast({
        title: "Success",
        description: "Analysis shared successfully!",
      });
      setIsShareDialogOpen(false);
    },
    onError: () => {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to share analysis. Please try again.",
      });
    },
  });

  // Handle file upload
  const handleFileUpload = (file: File) => {
    const fileType = file.type.split('/')[0];
    
    // Check file size early for videos
    if (fileType === 'video' && file.size > 50 * 1024 * 1024) {
      toast({
        title: "File Too Large",
        description: "Video files must be under 50MB. Please compress your video or use a shorter clip.",
        variant: "destructive",
      });
      return;
    }
    
    if (fileType === 'image' || fileType === 'video') {
      handleUploadMedia.mutate(file);
    } else {
      toast({
        variant: "destructive",
        title: "Unsupported File Type",
        description: "Please upload an image or video file."
      });
    }
  };

  // Handle text analysis submission
  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (textInput.trim()) {
      handleTextAnalysis.mutate(textInput);
    }
  };
  
  // Handle chat message submission
  const handleChatSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      // Add user message immediately to UI
      setMessages(prev => [...prev, { role: 'user', content: input }]);
      chatMutation.mutate(input);
      setInput("");
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent, submitHandler: (e: React.FormEvent) => void) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitHandler(e as unknown as React.FormEvent);
    }
  };
  
  const onShareSubmit = (data: z.infer<typeof shareSchema>) => {
    shareMutation.mutate(data);
  };
  
  // Generic dropzone for all file types
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      handleFileUpload(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    noClick: true,
    noKeyboard: true,
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024, // 50MB limit
  });

  // Click handlers for different upload types
  const handleImageVideoClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };
  
  // Document upload handler
  const handleDocumentClick = () => {
    if (documentInputRef.current) {
      documentInputRef.current.click();
    }
  };
  
  const handleDocumentInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type === 'application/pdf' || 
          file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' || 
          file.type === 'text/plain') {
        handleDocumentUpload.mutate(file);
      } else {
        toast({
          variant: "destructive",
          title: "Unsupported File Type",
          description: "Please upload a PDF, DOCX, or TXT file."
        });
      }
    }
  };
  
  // Document upload mutation
  const handleDocumentUpload = useMutation({
    mutationFn: async (file: File) => {
      setIsAnalyzing(true);
      setAnalysisProgress(10);
      
      const reader = new FileReader();
      return new Promise((resolve, reject) => {
        reader.onload = async (e) => {
          try {
            const fileData = e.target?.result as string;
            setAnalysisProgress(30);
            
            const response = await analyzeDocument(
              fileData,
              file.name,
              file.type,
              sessionId,
              selectedModel
            );
            
            setAnalysisProgress(50);
            
            if (response && response.analysisId) {
              setAnalysisId(response.analysisId);
              setDocumentChunks(response.chunks || []);
              setDocumentFileName(file.name);
              setDocumentFileType(file.type);
              setShowChunkSelection(true);
              setAnalysisProgress(100);
            }
            
            resolve(response);
          } catch (error) {
            reject(error);
          }
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    },
    onSuccess: (data) => {
      toast({
        title: "Document Uploaded",
        description: "Select text chunks to analyze with 25 psychological metrics.",
      });
      setIsAnalyzing(false);
    },
    onError: (error: any) => {
      console.error("Document upload error:", error);
      toast({
        variant: "destructive",
        title: "Upload Failed",
        description: error.message || "Failed to upload document. Please try again.",
      });
      setIsAnalyzing(false);
      setAnalysisProgress(0);
    },
  });
  
  // Document chunks analysis mutation
  const handleChunkAnalysis = useMutation({
    mutationFn: async ({ analysisId, selectedChunks }: { analysisId: number, selectedChunks: number[] }) => {
      setIsAnalyzing(true);
      setAnalysisProgress(10);
      
      const response = await analyzeDocumentChunks(
        analysisId,
        selectedChunks,
        selectedModel
      );
      
      setAnalysisProgress(90);
      
      if (response && response.metricsAnalysis) {
        setMetricsAnalysis(response.metricsAnalysis);
        
        // Add message to chat
        if (response.message) {
          setMessages(prev => [...prev, response.message]);
        } else {
          // Fallback - create a display message if none provided
          const summaryMessage = {
            role: "assistant" as const,
            content: `Document Analysis Complete\n\n25 psychological metrics have been analyzed across ${selectedChunks.length} text chunks. The analysis includes detailed scoring, explanations, and direct quotations from your document.`
          };
          setMessages(prev => [...prev, summaryMessage]);
        }
        
        setAnalysisProgress(100);
      }
      
      return response;
    },
    onSuccess: (data) => {
      toast({
        title: "Analysis Complete",
        description: "25 psychological metrics have been analyzed.",
      });
      setIsAnalyzing(false);
    },
    onError: (error: any) => {
      console.error("Chunk analysis error:", error);
      toast({
        variant: "destructive",
        title: "Analysis Failed",
        description: error.message || "Failed to analyze chunks. Please try again.",
      });
      setIsAnalyzing(false);
      setAnalysisProgress(0);
    },
  });
  
  // Toggle chunk selection
  const toggleChunkSelection = (chunkId: number) => {
    setSelectedChunks(prev => 
      prev.includes(chunkId) 
        ? prev.filter(id => id !== chunkId)
        : [...prev, chunkId]
    );
  };
  
  // Toggle metric expansion
  const toggleMetricExpansion = (metricIndex: number) => {
    setExpandedMetrics(prev => {
      const newSet = new Set(prev);
      if (newSet.has(metricIndex)) {
        newSet.delete(metricIndex);
      } else {
        newSet.add(metricIndex);
      }
      return newSet;
    });
  };
  
  // Analyze selected chunks
  const analyzeSelectedChunks = () => {
    if (!analysisId || selectedChunks.length === 0) {
      toast({
        variant: "destructive",
        title: "No Chunks Selected",
        description: "Please select at least one chunk to analyze.",
      });
      return;
    }
    
    handleChunkAnalysis.mutate({ analysisId, selectedChunks });
  };
  
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      const fileType = file.type.split('/')[0];
      if (fileType === 'image' || fileType === 'video') {
        handleUploadMedia.mutate(file);
      } else {
        toast({
          variant: "destructive",
          title: "Unsupported File Type",
          description: "Please upload an image or video file."
        });
      }
    }
  };

  return (
    <div className="container mx-auto p-4 max-w-6xl" {...getRootProps()}>
      <h1 className="text-4xl font-bold text-center mb-4">AI Personality Analysis</h1>
      
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
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        {/* Left Column - Inputs and Upload */}
        <div className="space-y-6">
          {/* Model Selector */}
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">Step 1: Select AI Model</h2>
            <Select
              value={selectedModel}
              onValueChange={(value) => setSelectedModel(value as ModelType)}
              disabled={isAnalyzing}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select AI Model" />
              </SelectTrigger>
              <SelectContent>
                {availableServices.anthropic && <SelectItem value="anthropic">ZHI 1 (Recommended)</SelectItem>}
                {availableServices.openai && <SelectItem value="openai">ZHI 2</SelectItem>}
                <SelectItem value="deepseek">ZHI 3</SelectItem>
                {availableServices.perplexity && <SelectItem value="perplexity">ZHI 4</SelectItem>}
              </SelectContent>
            </Select>
            
            <div className="mt-4">
              <div className="flex justify-between items-center mb-2">
                <h3 className="text-sm font-medium">Available Services</h3>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  className="h-6 text-xs px-2"
                  onClick={() => setShowAdvancedServices(prev => !prev)}
                >
                  {showAdvancedServices ? "Hide Details" : "Show All"}
                </Button>
              </div>
              
              <div className="text-xs space-y-1 text-muted-foreground">
                <div className="flex items-center">
                  <div className={`w-2 h-2 rounded-full mr-2 ${availableServices.anthropic ? 'bg-green-500' : 'bg-red-500'}`}></div>
                  <span>ZHI 1</span>
                </div>
                <div className="flex items-center">
                  <div className={`w-2 h-2 rounded-full mr-2 ${availableServices.openai ? 'bg-green-500' : 'bg-red-500'}`}></div>
                  <span>ZHI 2</span>
                </div>
                <div className="flex items-center">
                  <div className={`w-2 h-2 rounded-full mr-2 ${availableServices.deepseek ? 'bg-green-500' : 'bg-red-500'}`}></div>
                  <span>ZHI 3</span>
                </div>
                <div className="flex items-center">
                  <div className={`w-2 h-2 rounded-full mr-2 ${availableServices.perplexity ? 'bg-green-500' : 'bg-red-500'}`}></div>
                  <span>ZHI 4</span>
                </div>
                
                {showAdvancedServices && (
                  <>
                    <div className="h-px bg-gray-200 my-2"></div>
                    <h4 className="font-medium mb-1">Face Analysis</h4>
                    <div className="flex items-center">
                      <div className={`w-2 h-2 rounded-full mr-2 ${availableServices.azure_face ? 'bg-green-500' : 'bg-red-500'}`}></div>
                      <span>Azure Face API</span>
                    </div>
                    <div className="flex items-center">
                      <div className={`w-2 h-2 rounded-full mr-2 ${availableServices.facepp ? 'bg-green-500' : 'bg-red-500'}`}></div>
                      <span>Face++</span>
                    </div>
                    <div className="flex items-center">
                      <div className={`w-2 h-2 rounded-full mr-2 ${availableServices.google_vision ? 'bg-green-500' : 'bg-red-500'}`}></div>
                      <span>Google Vision</span>
                    </div>
                    <div className="flex items-center">
                      <div className={`w-2 h-2 rounded-full mr-2 ${availableServices.aws_rekognition ? 'bg-green-500' : 'bg-red-500'}`}></div>
                      <span>AWS Rekognition</span>
                    </div>
                    
                    <div className="h-px bg-gray-200 my-2"></div>
                    <h4 className="font-medium mb-1">Transcription</h4>
                    <div className="flex items-center">
                      <div className={`w-2 h-2 rounded-full mr-2 ${availableServices.gladia ? 'bg-green-500' : 'bg-red-500'}`}></div>
                      <span>Gladia</span>
                    </div>
                    <div className="flex items-center">
                      <div className={`w-2 h-2 rounded-full mr-2 ${availableServices.assemblyai ? 'bg-green-500' : 'bg-red-500'}`}></div>
                      <span>AssemblyAI</span>
                    </div>
                    <div className="flex items-center">
                      <div className={`w-2 h-2 rounded-full mr-2 ${availableServices.deepgram ? 'bg-green-500' : 'bg-red-500'}`}></div>
                      <span>Deepgram</span>
                    </div>
                    
                    <div className="h-px bg-gray-200 my-2"></div>
                    <h4 className="font-medium mb-1">Video Analysis</h4>
                    <div className="flex items-center">
                      <div className={`w-2 h-2 rounded-full mr-2 ${availableServices.azure_video_indexer ? 'bg-green-500' : 'bg-red-500'}`}></div>
                      <span>Azure Video Indexer</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          </Card>
          
          {/* Upload Options */}
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">Step 2: Choose Input Type</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Button 
                variant="outline" 
                className="h-24 flex flex-col items-center justify-center" 
                onClick={handleImageVideoClick}
                disabled={isAnalyzing}
              >
                <FileImage className="h-8 w-8 mb-2" />
                <span>Image</span>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*,video/*"
                  style={{ display: 'none' }}
                  onChange={(e) => handleFileInputChange(e)}
                />
              </Button>
              
              <Button 
                variant="outline" 
                className="h-24 flex flex-col items-center justify-center" 
                onClick={handleImageVideoClick}
                disabled={isAnalyzing}
              >
                <Film className="h-8 w-8 mb-2" />
                <span>Video</span>
              </Button>
              
              <Button 
                variant="outline" 
                className="h-24 flex flex-col items-center justify-center" 
                onClick={handleDocumentClick}
                disabled={isAnalyzing}
              >
                <File className="h-8 w-8 mb-2" />
                <span>Document</span>
                <input
                  ref={documentInputRef}
                  type="file"
                  accept=".pdf,.docx,.txt"
                  style={{ display: 'none' }}
                  onChange={(e) => handleDocumentInputChange(e)}
                />
              </Button>
            </div>
            
            {isAnalyzing && (
              <div className="mt-4 space-y-2">
                <div className="flex justify-between">
                  <span>Analyzing...</span>
                  <span>{analysisProgress}%</span>
                </div>
                <Progress value={analysisProgress} className="w-full" />
              </div>
            )}
            
            {/* Drag area info */}
            <div className={`mt-4 p-4 border-2 border-dashed rounded-lg text-center cursor-pointer transition-colors ${isDragActive ? "border-primary bg-primary/5" : "border-muted"}`}>
              <input {...getInputProps()} />
              <p className="text-muted-foreground">
                Drag & drop files here to analyze
              </p>
              <p className="text-xs text-muted-foreground">
                Supports JPG, PNG, MP4, MOV (max 50MB)
              </p>
            </div>
          </Card>
          
          {/* Input Preview */}
          <Card className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">Input Preview</h2>
              {!uploadedMedia && (
                <Button 
                  variant="outline" 
                  onClick={() => {
                    // Clear other inputs and focus on text
                    setUploadedMedia(null);
                    setTextInput(textInput || "");
                  }}
                  className="flex items-center gap-2"
                  disabled={isAnalyzing}
                >
                  <span>Text Input</span>
                </Button>
              )}
            </div>
            
            {uploadedMedia && mediaType === "image" && (
              <div className="space-y-4">
                <img 
                  src={uploadedMedia} 
                  alt="Uploaded" 
                  className="max-w-full h-auto rounded-lg shadow-md mx-auto"
                />
                <div className="text-center text-sm text-muted-foreground mb-4">
                  Face detection will analyze personality traits and emotions
                </div>
                
                {/* Re-analyze with current model button */}
                <Button 
                  onClick={() => {
                    if (mediaData) {
                      // Clear messages for new analysis
                      setMessages([]);
                      setIsAnalyzing(true);
                      setAnalysisProgress(0);
                      
                      // Use the stored media data directly
                      uploadMedia({
                        sessionId,
                        fileData: mediaData,
                        fileName: "re-analysis.jpg",
                        fileType: "image/jpeg",
                        selectedModel,
                        title: "Image Re-analysis"
                      }).then(response => {
                        setAnalysisProgress(100);
                        
                        if (response && response.analysisId) {
                          setAnalysisId(response.analysisId);
                        }
                        
                        if (response && response.messages && Array.isArray(response.messages)) {
                          setMessages(response.messages);
                        }
                        
                        toast({
                          title: "Analysis Complete",
                          description: "Your image has been re-analyzed with " + getModelDisplayName(selectedModel),
                        });
                      }).catch(error => {
                        toast({
                          variant: "destructive",
                          title: "Error",
                          description: "Failed to re-analyze image. Please try again.",
                        });
                      }).finally(() => {
                        setIsAnalyzing(false);
                      });
                    }
                  }}
                  className="w-full"
                  disabled={isAnalyzing || !mediaData}
                >
                  Re-Analyze with {getModelDisplayName(selectedModel)}
                </Button>
              </div>
            )}
            
            {uploadedMedia && mediaType === "video" && (
              <div className="space-y-4">
                <video 
                  ref={videoRef}
                  src={uploadedMedia} 
                  controls
                  className="max-w-full h-auto rounded-lg shadow-md mx-auto"
                />
                <div className="text-center text-sm text-muted-foreground mb-4">
                  Video analysis will extract visual and audio insights
                </div>
                
                {/* Video Segment Selection for Large Videos */}
                {requiresSegmentSelection && videoSegments.length > 0 && (
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 space-y-3">
                    <h3 className="font-medium text-blue-900">Select Video Segment</h3>
                    <p className="text-sm text-blue-700">
                      Your video is large, so please select a 5-second segment to analyze for optimal performance:
                    </p>
                    <div className="text-xs text-blue-600 bg-blue-100 p-2 rounded">
                      💡 Tip: Analysis focuses on facial expressions, body language, and speech patterns in the selected segment.
                    </div>
                    
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-40 overflow-y-auto">
                      {videoSegments.map((segment) => (
                        <button
                          key={segment.id}
                          onClick={() => setSelectedVideoSegment(segment.id)}
                          className={`p-3 rounded-lg border text-left transition-all ${
                            selectedVideoSegment === segment.id
                              ? 'border-blue-500 bg-blue-100 text-blue-900'
                              : 'border-gray-200 hover:border-gray-300 bg-white'
                          }`}
                        >
                          <div className="font-medium">{segment.label}</div>
                          <div className="text-xs text-gray-600">{segment.duration}s duration</div>
                        </button>
                      ))}
                    </div>
                    
                    {videoDuration > 0 && (
                      <div className="text-xs text-blue-600">
                        Total video duration: {videoDuration.toFixed(1)}s | {videoSegments.length} segments available
                      </div>
                    )}
                    
                    {selectedVideoSegment && (
                      <Button
                        onClick={() => handleAnalyzeVideoSegment()}
                        disabled={isAnalyzing}
                        className="w-full"
                      >
                        {isAnalyzing ? "Analyzing..." : `Analyze Selected Segment`}
                      </Button>
                    )}
                  </div>
                )}
                
                {/* Original Video Segment Selection for Small Videos */}
                {!requiresSegmentSelection && mediaType === "video" && videoDuration > 0 && (
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 space-y-3">
                    <h3 className="font-medium text-blue-900">Video Segment Selection</h3>
                    <p className="text-sm text-blue-700">
                      For optimal performance, videos are processed in 5-second segments. Select which segment to analyze:
                    </p>
                    <div className="text-xs text-blue-600 bg-blue-100 p-2 rounded">
                      💡 Tip: Video processing may take 2-3 minutes depending on complexity. The system extracts facial analysis, 
                      audio transcription, and emotional insights from your selected segment.
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-blue-900 mb-1">
                          Start Time (seconds)
                        </label>
                        <Input
                          type="number"
                          min={0}
                          max={Math.max(0, videoDuration - 1)}
                          step={1}
                          value={videoSegmentStart}
                          onChange={(e) => setVideoSegmentStart(Math.max(0, parseInt(e.target.value) || 0))}
                          className="w-full"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-blue-900 mb-1">
                          Duration (max 5s)
                        </label>
                        <Input
                          type="number"
                          min={1}
                          max={5}
                          step={1}
                          value={videoSegmentDuration}
                          onChange={(e) => setVideoSegmentDuration(Math.min(5, Math.max(1, parseInt(e.target.value) || 5)))}
                          className="w-full"
                        />
                      </div>
                    </div>
                    
                    {videoDuration > 0 && (
                      <div className="text-xs text-blue-600">
                        Video duration: {videoDuration.toFixed(1)}s | 
                        Analyzing: {videoSegmentStart}s to {Math.min(videoSegmentStart + videoSegmentDuration, videoDuration).toFixed(1)}s
                      </div>
                    )}
                  </div>
                )}
                
                {/* View full transcript button - only shows after analysis */}
                {analysisId && (
                  <Button 
                    variant="outline"
                    className="w-full mb-2"
                    onClick={() => {
                      // Get the analysis to extract transcript
                      fetch(`/api/analysis/${analysisId}`)
                        .then(res => res.json())
                        .then(data => {
                          if (data.audioTranscription) {
                            // Format transcript for display
                            let formattedTranscript = "# Full Video Transcript\n\n";
                            
                            // Add provider info with emoji
                            const provider = data.audioTranscription.provider || 'AI transcription service';
                            formattedTranscript += `*Transcription provided by ${provider}*\n\n`;
                            
                            // Get the transcription data from the standardized format
                            const transcriptionData = data.audioTranscription.transcriptionData || data.audioTranscription.transcription;
                            
                            // Add timestamps and text if utterances are available
                            if (transcriptionData?.utterances && transcriptionData.utterances.length > 0) {
                              // Sort utterances by start time if available
                              const sortedUtterances = [...transcriptionData.utterances].sort((a, b) => a.start - b.start);
                              
                              // Add sentiment indicators if available
                              sortedUtterances.forEach((utterance: any) => {
                                const start = new Date(utterance.start * 1000).toISOString().substr(14, 5);
                                const end = new Date(utterance.end * 1000).toISOString().substr(14, 5);
                                
                                // Add sentiment emoji if available
                                let sentimentIndicator = '';
                                if (utterance.sentiment) {
                                  if (utterance.sentiment === 'positive') sentimentIndicator = ' 😊';
                                  else if (utterance.sentiment === 'negative') sentimentIndicator = ' 😔';
                                  else if (utterance.sentiment === 'neutral') sentimentIndicator = ' 😐';
                                }
                                
                                formattedTranscript += `**[${start} - ${end}]**${sentimentIndicator} ${utterance.text}\n\n`;
                              });
                            } else if (transcriptionData?.words && transcriptionData.words.length > 0) {
                              // Alternative format with words
                              let lastTimestamp = 0;
                              let currentParagraph = "";
                              
                              // Sort words by start time
                              const sortedWords = [...transcriptionData.words].sort((a, b) => a.start - b.start);
                              
                              sortedWords.forEach((word: any, index: number) => {
                                // Start a new paragraph every 15 seconds or on punctuation
                                const isPunctuation = word.text.match(/[.!?]$/);
                                if ((word.start > lastTimestamp + 15 || isPunctuation) && currentParagraph) {
                                  const paragraphTime = new Date(lastTimestamp * 1000).toISOString().substr(14, 5);
                                  formattedTranscript += `**[${paragraphTime}]** ${currentParagraph}\n\n`;
                                  currentParagraph = word.text + " ";
                                  lastTimestamp = word.start;
                                } else {
                                  currentParagraph += word.text + " ";
                                }
                                
                                // Add final paragraph
                                if (index === sortedWords.length - 1 && currentParagraph) {
                                  const paragraphTime = new Date(lastTimestamp * 1000).toISOString().substr(14, 5);
                                  formattedTranscript += `**[${paragraphTime}]** ${currentParagraph}\n\n`;
                                }
                              });
                            } else if (transcriptionData?.full_text) {
                              // Simple text without timestamps
                              formattedTranscript += transcriptionData.full_text;
                            } else if (data.audioTranscription.text) {
                              // Fallback to text field if available
                              formattedTranscript += data.audioTranscription.text;
                            } else if (data.audioTranscription.transcription && typeof data.audioTranscription.transcription === 'string') {
                              // Fallback to old format if needed
                              formattedTranscript += data.audioTranscription.transcription;
                            } else {
                              formattedTranscript += "No detailed transcript available.";
                            }
                            
                            // Show transcript in a dialog
                            setMessages(prevMessages => [
                              ...prevMessages,
                              {
                                role: "assistant",
                                content: formattedTranscript
                              }
                            ]);
                            
                            toast({
                              title: "Transcript Loaded",
                              description: "Full video transcript added to the conversation"
                            });
                          } else {
                            toast({
                              variant: "destructive",
                              title: "Transcript Unavailable",
                              description: "No transcript found for this video"
                            });
                          }
                        })
                        .catch(error => {
                          console.error("Error loading transcript:", error);
                          toast({
                            variant: "destructive",
                            title: "Error",
                            description: "Failed to load transcript"
                          });
                        });
                    }}
                  >
                    <FileText className="h-4 w-4 mr-2" />
                    View Full Transcript
                  </Button>
                )}
                
                {/* Re-analyze with current model button */}
                <Button 
                  onClick={async () => {
                    if (analysisId && videoSegments.length > 0) {
                      try {
                        setIsAnalyzing(true);
                        setAnalysisProgress(10);
                        
                        // Find segment that corresponds to current time range
                        let targetSegmentId = 1;
                        if (videoSegments.length > 1) {
                          const targetSegment = videoSegments.find(seg => 
                            seg.startTime <= videoSegmentStart && 
                            seg.endTime > videoSegmentStart
                          );
                          targetSegmentId = targetSegment?.id || 1;
                        }
                        
                        const response = await analyzeVideoSegment(analysisId, targetSegmentId, selectedModel, sessionId);
                        setAnalysisProgress(90);
                        
                        // Clear previous video analysis messages and add the new one
                        if (response.videoAnalysis) {
                          const newMessage = {
                            role: "assistant",
                            content: `# Video Segment Analysis (${getModelDisplayName(selectedModel)})\n\n${response.videoAnalysis.summary}\n\n${response.videoAnalysis.analysisText}`,
                            timestamp: new Date().toISOString()
                          };
                          
                          // Filter out previous video analysis messages and add the new one
                          setMessages(prev => {
                            const filteredMessages = prev.filter(msg => 
                              !msg.content.includes("Video Segment Analysis") || 
                              msg.role !== "assistant"
                            );
                            return [...filteredMessages, newMessage];
                          });
                        }
                        
                        setAnalysisProgress(100);
                        
                        toast({
                          title: "Re-Analysis Complete",
                          description: "Video segment re-analyzed with " + getModelDisplayName(selectedModel),
                        });
                        
                      } catch (error: any) {
                        console.error('Video re-analysis error:', error);
                        toast({
                          variant: "destructive",
                          title: "Re-Analysis Failed",
                          description: error.message || "Failed to re-analyze video segment. Please try again.",
                        });
                        setAnalysisProgress(0);
                      } finally {
                        setIsAnalyzing(false);
                      }
                    }
                  }}
                  className="w-full"
                  disabled={isAnalyzing || !analysisId}
                >
                  Re-Analyze Segment ({videoSegmentStart}s-{videoSegmentStart + videoSegmentDuration}s) with {getModelDisplayName(selectedModel)}
                </Button>
              </div>
            )}
            
            {/* Document Chunk Selection */}
            {showChunkSelection && documentChunks.length > 0 && (
              <div className="mb-6">
                <h3 className="text-lg font-semibold mb-4">Select Document Chunks</h3>
                <p className="text-sm text-gray-600 mb-4">
                  File: {documentFileName} | {documentChunks.length} chunks | Select chunks to analyze
                </p>
                
                <div className="space-y-3 max-h-64 overflow-y-auto">
                  {documentChunks.map((chunk) => (
                    <div
                      key={chunk.id}
                      className={`border rounded-lg p-3 cursor-pointer transition-all ${
                        selectedChunks.includes(chunk.id)
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => toggleChunkSelection(chunk.id)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <div className={`w-4 h-4 rounded border-2 flex items-center justify-center ${
                            selectedChunks.includes(chunk.id)
                              ? 'bg-blue-500 border-blue-500'
                              : 'border-gray-300'
                          }`}>
                            {selectedChunks.includes(chunk.id) && (
                              <Check className="w-3 h-3 text-white" />
                            )}
                          </div>
                          <span className="text-sm font-medium">Chunk {chunk.id}</span>
                        </div>
                        <span className="text-xs text-gray-500">{chunk.wordCount} words</span>
                      </div>
                      <p className="text-sm text-gray-700 mt-2">{chunk.preview}</p>
                    </div>
                  ))}
                </div>
                
                <div className="mt-4 flex justify-between items-center">
                  <p className="text-sm text-gray-600">
                    {selectedChunks.length} chunks selected
                  </p>
                  <Button 
                    onClick={analyzeSelectedChunks}
                    disabled={selectedChunks.length === 0 || isAnalyzing}
                    className="flex items-center space-x-2"
                  >
                    <Eye className="w-4 h-4" />
                    <span>Analyze Selected</span>
                  </Button>
                </div>
              </div>
            )}

            {/* 25 Metrics Display */}
            {metricsAnalysis && metricsAnalysis.metrics && (
              <div className="mb-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold">25 Psychological Metrics</h3>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      if (analysisId && selectedChunks.length > 0) {
                        handleChunkAnalysis.mutate({ analysisId, selectedChunks });
                      }
                    }}
                    disabled={isAnalyzing || !analysisId || selectedChunks.length === 0}
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Regenerate
                  </Button>
                </div>
                
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {metricsAnalysis.metrics.map((metric: any, index: number) => (
                    <div
                      key={index}
                      className="border rounded-lg p-4 cursor-pointer hover:bg-gray-50 transition-colors"
                      onClick={() => toggleMetricExpansion(index)}
                    >
                      <div className="flex justify-between items-center">
                        <div className="flex-1">
                          <h4 className="font-semibold text-sm">{metric.name}</h4>
                          <p className="text-xs text-gray-600 mt-1">{metric.explanation}</p>
                        </div>
                        <div className="flex items-center space-x-3">
                          <div className="text-right">
                            <div className="text-lg font-bold">{metric.score}</div>
                            <div className="text-xs text-gray-500">/100</div>
                          </div>
                          <div className="w-16 bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${metric.score}%` }}
                            />
                          </div>
                        </div>
                      </div>
                      
                      {expandedMetrics.has(index) && (
                        <div className="mt-3 pt-3 border-t">
                          <h5 className="font-medium text-sm mb-2">Detailed Analysis</h5>
                          <p className="text-sm text-gray-700 mb-3">{metric.detailedAnalysis}</p>
                          
                          {metric.quotes && metric.quotes.length > 0 && (
                            <div>
                              <h6 className="font-medium text-sm mb-2">Key Quotes</h6>
                              <div className="space-y-1">
                                {metric.quotes.map((quote: string, quoteIndex: number) => (
                                  <div key={quoteIndex} className="bg-gray-100 p-2 rounded text-sm italic">
                                    "{quote}"
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
                
                <div className="mt-4 pt-4 border-t">
                  <p className="text-xs text-gray-500 text-center">
                    Click on any metric to view detailed analysis and quotes
                  </p>
                </div>
              </div>
            )}
            
            {!uploadedMedia && (
              <div className="space-y-4">
                <form onSubmit={handleTextSubmit} className="space-y-4">
                  <Textarea
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    onKeyDown={(e) => handleKeyPress(e, handleTextSubmit)}
                    placeholder="Type or paste text to analyze..."
                    className="min-h-[250px] resize-y"
                    disabled={isAnalyzing}
                  />
                  
                  {/* Additional info dialog button */}
                  <div className="flex gap-2">
                    <Dialog open={showAdditionalInfoDialog} onOpenChange={setShowAdditionalInfoDialog}>
                      <DialogTrigger asChild>
                        <Button variant="outline" type="button">
                          Add Context
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="max-w-2xl">
                        <DialogHeader>
                          <DialogTitle>Additional Context Information</DialogTitle>
                          <p className="text-sm text-muted-foreground">
                            Provide any additional context, background, or specific information that might help the AI analyze your text more accurately.
                          </p>
                        </DialogHeader>
                        <Textarea
                          value={additionalInfo}
                          onChange={(e) => setAdditionalInfo(e.target.value)}
                          placeholder="Enter any relevant context, background information, or specific aspects you'd like the analysis to focus on..."
                          className="min-h-[200px] resize-y"
                        />
                        <DialogFooter>
                          <Button variant="outline" onClick={() => setShowAdditionalInfoDialog(false)}>
                            Cancel
                          </Button>
                          <Button onClick={() => setShowAdditionalInfoDialog(false)}>
                            Save Context
                          </Button>
                        </DialogFooter>
                      </DialogContent>
                    </Dialog>
                    
                    <Button 
                      type="submit" 
                      className="flex-1" 
                      disabled={!textInput.trim() || isAnalyzing}
                    >
                      Comprehensive Analysis (40 Parameters)
                    </Button>
                  </div>
                </form>
                
                {additionalInfo && (
                  <div className="bg-blue-50 p-3 rounded-md">
                    <p className="text-sm text-blue-800">
                      <strong>Additional Context:</strong> {additionalInfo.substring(0, 100)}
                      {additionalInfo.length > 100 && "..."}
                    </p>
                  </div>
                )}
              </div>
            )}
          </Card>
        </div>
        
        {/* Right Column - Results and Chat */}
        <div className="space-y-6">
          {/* ANALYSIS BOX */}
          <Card className="p-6 border-2 border-primary">
            <div className="flex justify-between items-center mb-4">
              <div className="flex items-center gap-2">
                <h2 className="text-xl font-bold">ANALYSIS</h2>
                {/* New Analysis button - always visible */}
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="flex items-center gap-2"
                  onClick={async () => {
                    try {
                      // First clear the session on the server
                      await clearSession(sessionId);
                      
                      // Generate a new session ID to ensure a completely clean state
                      const newSessionId = nanoid();
                      window.location.href = `/?session=${newSessionId}`;
                      
                      // Clear all current state to start a new analysis
                      setMessages([]);
                      setUploadedMedia(null);
                      setMediaData(null);
                      setTextInput("");
                      setAnalysisId(null);
                      setAnalysisProgress(0);
                      toast({
                        title: "New Analysis",
                        description: "Starting a completely new analysis session",
                      });
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
              </div>
              
              {messages.length > 0 && (
                <div className="flex gap-2">
                  {/* Download buttons */}
                  {analysisId && (
                    <>
                      <Button 
                        variant="default" 
                        size="sm" 
                        className="flex items-center gap-2 bg-green-600 hover:bg-green-700"
                        onClick={() => {
                          toast({
                            title: "Downloading PDF",
                            description: "Your analysis is being downloaded as PDF"
                          });
                          downloadAnalysis(analysisId, 'pdf');
                        }}
                      >
                        <Download className="h-4 w-4" />
                        <span>Download PDF</span>
                      </Button>
                      
                      <Button 
                        variant="default" 
                        size="sm" 
                        className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700"
                        onClick={() => {
                          toast({
                            title: "Downloading Word Document",
                            description: "Your analysis is being downloaded as DOCX"
                          });
                          downloadAnalysis(analysisId, 'docx');
                        }}
                      >
                        <File className="h-4 w-4" />
                        <span>Download Word</span>
                      </Button>
                      
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
                    </>
                  )}
                  
                  {/* Share button */}
                  {emailServiceAvailable && (
                    <Dialog open={isShareDialogOpen} onOpenChange={setIsShareDialogOpen}>
                      <DialogTrigger asChild>
                        <Button variant="outline" size="sm" className="flex items-center gap-2">
                          <Share2 className="h-4 w-4" />
                          <span>Share</span>
                        </Button>
                      </DialogTrigger>
                      <DialogContent>
                        <DialogHeader>
                          <DialogTitle>Share Analysis</DialogTitle>
                        </DialogHeader>
                        <Form {...shareForm}>
                          <form onSubmit={shareForm.handleSubmit(onShareSubmit)} className="space-y-4">
                            <FormField
                              control={shareForm.control}
                              name="senderEmail"
                              render={({ field }) => (
                                <FormItem>
                                  <FormLabel>Your Email</FormLabel>
                                  <FormControl>
                                    <Input {...field} type="email" placeholder="youremail@example.com" />
                                  </FormControl>
                                  <FormMessage />
                                </FormItem>
                              )}
                            />
                            <FormField
                              control={shareForm.control}
                              name="recipientEmail"
                              render={({ field }) => (
                                <FormItem>
                                  <FormLabel>Recipient's Email</FormLabel>
                                  <FormControl>
                                    <Input {...field} type="email" placeholder="recipient@example.com" />
                                  </FormControl>
                                  <FormMessage />
                                </FormItem>
                              )}
                            />
                            <DialogFooter>
                              <Button 
                                type="submit" 
                                disabled={shareMutation.isPending}
                                className="w-full"
                              >
                                {shareMutation.isPending ? "Sending..." : "Share Analysis"}
                              </Button>
                            </DialogFooter>
                          </form>
                        </Form>
                      </DialogContent>
                    </Dialog>
                  )}
                </div>
              )}
            </div>
            
            <div className="h-[400px] flex flex-col">
              {/* Comprehensive Analysis Display */}
              {showComprehensiveAnalysis && comprehensiveAnalysis && (
                <div className="mb-6 border rounded-lg bg-gradient-to-r from-blue-50 to-purple-50 p-1">
                  <div className="bg-white rounded-md p-4">
                    <h3 className="text-lg font-semibold mb-4 text-center">
                      Comprehensive 40-Parameter Analysis
                    </h3>
                    
                    <Tabs defaultValue="cognitive" className="w-full">
                      <TabsList className="grid w-full grid-cols-2">
                        <TabsTrigger value="cognitive">Cognitive Analysis (20)</TabsTrigger>
                        <TabsTrigger value="psychological">Psychological Analysis (20)</TabsTrigger>
                      </TabsList>
                      
                      <TabsContent value="cognitive" className="space-y-3 max-h-80 overflow-y-auto">
                        {cognitiveParameters.map((param) => {
                          const analysis = comprehensiveAnalysis?.cognitiveAnalysis?.[param.id];
                          if (!analysis) return null;
                          
                          return (
                            <Collapsible
                              key={param.id}
                              open={expandedCognitiveParams.has(param.id)}
                              onOpenChange={(open) => {
                                const newExpanded = new Set(expandedCognitiveParams);
                                if (open) {
                                  newExpanded.add(param.id);
                                } else {
                                  newExpanded.delete(param.id);
                                }
                                setExpandedCognitiveParams(newExpanded);
                              }}
                            >
                              <CollapsibleTrigger asChild>
                                <div className="border rounded-lg p-3 cursor-pointer hover:bg-gray-50 transition-colors">
                                  <div className="flex justify-between items-center">
                                    <div className="flex-1">
                                      <h4 className="font-semibold text-sm">{param.name}</h4>
                                      <p className="text-xs text-gray-600 mt-1">{param.description}</p>
                                    </div>
                                    <div className="flex items-center space-x-3">
                                      <div className="text-right">
                                        <div className="text-lg font-bold">{analysis.score || 'N/A'}</div>
                                        <div className="text-xs text-gray-500">/100</div>
                                      </div>
                                      {expandedCognitiveParams.has(param.id) ? 
                                        <ChevronDown className="h-4 w-4" /> : 
                                        <ChevronRight className="h-4 w-4" />
                                      }
                                    </div>
                                  </div>
                                </div>
                              </CollapsibleTrigger>
                              
                              <CollapsibleContent className="mt-2 p-3 bg-gray-50 rounded-md">
                                <div className="space-y-3">
                                  <div>
                                    <h5 className="font-medium text-sm mb-2">Analysis</h5>
                                    <p className="text-sm text-gray-700">{analysis.analysis}</p>
                                  </div>
                                  
                                  {analysis.quotations && analysis.quotations.length > 0 && (
                                    <div>
                                      <h5 className="font-medium text-sm mb-2">Key Quotations</h5>
                                      <div className="space-y-1">
                                        {analysis.quotations.map((quote: string, index: number) => (
                                          <div key={index} className="bg-white p-2 rounded text-sm italic border-l-4 border-blue-300">
                                            "{quote}"
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                  
                                  {analysis.evidence && (
                                    <div>
                                      <h5 className="font-medium text-sm mb-2">Evidence</h5>
                                      <p className="text-sm text-gray-600">{analysis.evidence}</p>
                                    </div>
                                  )}
                                </div>
                              </CollapsibleContent>
                            </Collapsible>
                          );
                        })}
                      </TabsContent>
                      
                      <TabsContent value="psychological" className="space-y-3 max-h-80 overflow-y-auto">
                        {psychologicalParameters.map((param) => {
                          const analysis = comprehensiveAnalysis?.psychologicalAnalysis?.[param.id];
                          if (!analysis) return null;
                          
                          return (
                            <Collapsible
                              key={param.id}
                              open={expandedPsychParams.has(param.id)}
                              onOpenChange={(open) => {
                                const newExpanded = new Set(expandedPsychParams);
                                if (open) {
                                  newExpanded.add(param.id);
                                } else {
                                  newExpanded.delete(param.id);
                                }
                                setExpandedPsychParams(newExpanded);
                              }}
                            >
                              <CollapsibleTrigger asChild>
                                <div className="border rounded-lg p-3 cursor-pointer hover:bg-gray-50 transition-colors">
                                  <div className="flex justify-between items-center">
                                    <div className="flex-1">
                                      <h4 className="font-semibold text-sm">{param.name}</h4>
                                      <p className="text-xs text-gray-600 mt-1">{param.description}</p>
                                    </div>
                                    <div className="flex items-center space-x-3">
                                      <div className="text-right">
                                        <div className="text-lg font-bold">{analysis.score || 'N/A'}</div>
                                        <div className="text-xs text-gray-500">/100</div>
                                      </div>
                                      {expandedPsychParams.has(param.id) ? 
                                        <ChevronDown className="h-4 w-4" /> : 
                                        <ChevronRight className="h-4 w-4" />
                                      }
                                    </div>
                                  </div>
                                </div>
                              </CollapsibleTrigger>
                              
                              <CollapsibleContent className="mt-2 p-3 bg-gray-50 rounded-md">
                                <div className="space-y-3">
                                  <div>
                                    <h5 className="font-medium text-sm mb-2">Analysis</h5>
                                    <p className="text-sm text-gray-700">{analysis.analysis}</p>
                                  </div>
                                  
                                  {analysis.quotations && analysis.quotations.length > 0 && (
                                    <div>
                                      <h5 className="font-medium text-sm mb-2">Key Quotations</h5>
                                      <div className="space-y-1">
                                        {analysis.quotations.map((quote: string, index: number) => (
                                          <div key={index} className="bg-white p-2 rounded text-sm italic border-l-4 border-purple-300">
                                            "{quote}"
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                  
                                  {analysis.evidence && (
                                    <div>
                                      <h5 className="font-medium text-sm mb-2">Evidence</h5>
                                      <p className="text-sm text-gray-600">{analysis.evidence}</p>
                                    </div>
                                  )}
                                </div>
                              </CollapsibleContent>
                            </Collapsible>
                          );
                        })}
                      </TabsContent>
                    </Tabs>
                    
                    {comprehensiveAnalysis.overallSummary && (
                      <div className="mt-4 p-3 bg-gradient-to-r from-blue-100 to-purple-100 rounded-md">
                        <h4 className="font-semibold text-sm mb-2">Overall Summary</h4>
                        <p className="text-sm">{comprehensiveAnalysis.overallSummary}</p>
                      </div>
                    )}
                    
                    {/* Post-Analysis Dialogue Interface */}
                    <div className="mt-4 p-4 border-t">
                      <h4 className="font-semibold text-sm mb-3">Discuss Your Analysis</h4>
                      <div className="space-y-3">
                        <Textarea
                          placeholder="Share your thoughts, provide additional context, or ask questions about the analysis..."
                          className="min-h-[80px] resize-none text-sm"
                        />
                        <div className="flex gap-2">
                          <Button size="sm">
                            Refine Analysis
                          </Button>
                          <Button variant="outline" size="sm">
                            Generate Report
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {messages.length === 0 && !showComprehensiveAnalysis ? (
                <div className="flex flex-col items-center justify-center space-y-4 h-full text-center text-muted-foreground">
                  <AlertCircle className="h-12 w-12" />
                  <div>
                    <p className="text-lg font-medium">No analysis yet</p>
                    <p>Upload media, enter text, or select a document to analyze.</p>
                    <p className="text-xs mt-4">Debug: {JSON.stringify({ messageCount: messages.length, sessionId })}</p>
                  </div>
                </div>
              ) : (
                <ScrollArea className="flex-1 pr-4 mb-4">
                  <div className="space-y-4">
                    <div className="text-xs text-muted-foreground mb-2">
                      Debug: Found {messages.length} messages, {messages.filter(m => m.role === "assistant").length} are from assistant
                    </div>
                    
                    {/* Download info message */}
                    {analysisId && (
                      <div className="bg-blue-50 text-blue-800 p-3 rounded-md mb-4 flex items-center gap-2 text-sm">
                        <Download className="h-5 w-5" />
                        <span>
                          <strong>Save your analysis!</strong> Use the download buttons to save as PDF, Word, or TXT format.
                        </span>
                      </div>
                    )}
                    {messages.filter(message => message.role === "assistant").map((message, index) => (
                      <div
                        key={index}
                        className="flex flex-col p-4 rounded-lg bg-white border border-gray-200 shadow-sm"
                      >
                        <div 
                          className="whitespace-pre-wrap text-md"
                          dangerouslySetInnerHTML={{ 
                            __html: message.content
                              .replace(/\n/g, '<br/>')
                              .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                              .replace(/^(#+)\s+(.*?)$/gm, (_: string, hashes: string, text: string) => 
                                `<h${hashes.length} class="font-bold text-lg mt-3 mb-1">${text}</h${hashes.length}>`)
                              .replace(/- (.*?)$/gm, '<li class="ml-4">• $1</li>')
                          }} 
                        />
                        
                        {/* Download buttons at bottom of each analysis */}
                        {analysisId && index === messages.filter(m => m.role === "assistant").length - 1 && (
                          <div className="flex gap-2 mt-4 justify-end">
                            <Button
                              variant="outline"
                              size="sm"
                              className="flex items-center gap-1 text-xs"
                              onClick={() => {
                                toast({
                                  title: "Downloading PDF",
                                  description: "Your analysis is being downloaded as PDF"
                                });
                                downloadAnalysis(analysisId, 'pdf');
                              }}
                            >
                              <Download className="h-3 w-3" />
                              <span>Save as PDF</span>
                            </Button>
                            
                            <Button
                              variant="outline"
                              size="sm"
                              className="flex items-center gap-1 text-xs"
                              onClick={() => {
                                toast({
                                  title: "Downloading Word Document",
                                  description: "Your analysis is being downloaded as DOCX"
                                });
                                downloadAnalysis(analysisId, 'docx');
                              }}
                            >
                              <File className="h-3 w-3" />
                              <span>Save as Word</span>
                            </Button>
                            
                            <Button
                              variant="outline"
                              size="sm"
                              className="flex items-center gap-1 text-xs"
                              onClick={() => {
                                toast({
                                  title: "Downloading Text File",
                                  description: "Your analysis is being downloaded as TXT"
                                });
                                downloadAnalysis(analysisId, 'txt');
                              }}
                            >
                              <FileText className="h-3 w-3" />
                              <span>Save as TXT</span>
                            </Button>
                          </div>
                        )}
                      </div>
                    ))}
                    <div ref={messagesEndRef} />
                  </div>
                </ScrollArea>
              )}
            </div>
          </Card>
          
          {/* CHAT BOX */}
          <Card className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">Chat</h2>
            </div>
            
            <div className="h-[300px] flex flex-col">
              {messages.length === 0 ? (
                <div className="flex flex-col items-center justify-center space-y-4 h-full text-center text-muted-foreground">
                  <div>
                    <p>No conversation yet. Ask questions about the analysis after it's generated.</p>
                  </div>
                </div>
              ) : (
                <ScrollArea className="flex-1 pr-4 mb-4">
                  <div className="space-y-4">
                    {messages.map((message, index) => (
                      <div
                        key={index}
                        className={`flex flex-col p-4 rounded-lg ${
                          message.role === "user" ? "bg-primary/10 ml-8" : "bg-primary/5 mr-4"
                        }`}
                      >
                        <span className="font-semibold text-sm mb-2">
                          {message.role === "user" ? "You" : "AI"}
                        </span>
                        <div 
                          className="whitespace-pre-wrap text-md"
                          dangerouslySetInnerHTML={{ 
                            __html: message.content
                              .replace(/\n/g, '<br/>')
                              .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                          }} 
                        />
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              )}
              
              <form onSubmit={handleChatSubmit} className="mt-auto">
                <div className="flex gap-2">
                  <Textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => handleKeyPress(e, handleChatSubmit)}
                    placeholder="Ask a question about the analysis..."
                    className="min-h-[80px] resize-none"
                  />
                  <Button 
                    type="submit" 
                    className="self-end"
                    disabled={!input.trim() || chatMutation.isPending}
                  >
                    <Send className="h-4 w-4" />
                  </Button>
                </div>
              </form>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}