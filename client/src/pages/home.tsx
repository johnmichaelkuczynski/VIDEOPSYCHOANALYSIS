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
import { uploadMedia, sendMessage, shareAnalysis, getSharedAnalysis, analyzeText, analyzeDocument, downloadAnalysis, clearSession, ModelType, MediaType } from "@/lib/api";
import { Upload, Send, FileImage, Film, Share2, AlertCircle, FileText, File, Download } from "lucide-react";
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
  const [selectedModel, setSelectedModel] = useState<ModelType>("openai");
  const [documentName, setDocumentName] = useState<string>("");
  
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
  }>({
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
        if (status.openai) {
          setSelectedModel("openai");
        } else if (status.anthropic) {
          setSelectedModel("anthropic");
        } else if (status.perplexity) {
          setSelectedModel("perplexity");
        }
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
        
        const response = await analyzeText(text, sessionId, selectedModel);
        
        setAnalysisProgress(80);
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(response.messages);
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
        title: "Analysis Complete",
        description: "Your text has been successfully analyzed.",
      });
      setTextInput("");
    }
  });

  // Document analysis with file upload
  const handleDocumentAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        setMessages([]);
        
        setDocumentName(file.name);
        setAnalysisProgress(30);
        
        // Read the file as data URL
        const reader = new FileReader();
        const fileData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setAnalysisProgress(50);
        
        // Determine file type
        const fileExt = file.name.split('.').pop()?.toLowerCase();
        const fileType = fileExt === 'pdf' ? 'pdf' : 'docx';
        
        const response = await analyzeDocument(
          fileData,
          file.name,
          fileType,
          sessionId,
          selectedModel
        );
        
        setAnalysisProgress(80);
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(response.messages);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Document analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze document. Please try again.",
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
              console.log("Fetched messages after document analysis:", data);
              setMessages(data);
            }
          })
          .catch(err => console.error("Error fetching messages after document analysis:", err));
      }
      
      toast({
        title: "Analysis Complete",
        description: "Your document has been successfully analyzed.",
      });
    }
  });

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
        }
        
        // Set preview and store media data for re-analysis
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setAnalysisProgress(50);
        
        // Maximum 5 people to analyze
        const maxPeople = 5;
        
        // Upload for analysis
        const response = await uploadMedia(
          mediaData, 
          mediaFileType, 
          sessionId, 
          { 
            selectedModel, 
            maxPeople 
          }
        );
        
        setAnalysisProgress(90);
        
        if (response && response.analysisId) {
          setAnalysisId(response.analysisId);
        }
        
        console.log("Response from uploadMedia:", response);
        
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
    
    if (fileType === 'image' || fileType === 'video') {
      handleUploadMedia.mutate(file);
    } else if (
      file.type === 'application/pdf' || 
      file.type === 'application/msword' || 
      file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
      file.type === 'text/plain'
    ) {
      handleDocumentAnalysis.mutate(file);
    } else {
      toast({
        variant: "destructive",
        title: "Unsupported File Type",
        description: "Please upload an image, video, PDF, DOC, DOCX, or TXT file."
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
  
  const handleDocumentClick = () => {
    if (documentInputRef.current) {
      documentInputRef.current.click();
    }
  };
  
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>, type: 'media' | 'document') => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (type === 'media') {
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
      } else {
        handleDocumentAnalysis.mutate(file);
      }
    }
  };

  return (
    <div className="container mx-auto p-4 max-w-6xl" {...getRootProps()}>
      <h1 className="text-4xl font-bold text-center mb-8">AI Personality Analysis</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        {/* Left Column - Inputs and Upload */}
        <div className="space-y-6">
          {/* Model Selector */}
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">Step 1: Select AI Model</h2>
            <Select
              value={selectedModel}
              onValueChange={(value) => setSelectedModel(value as ModelType)}
              disabled={isAnalyzing || (!availableServices.openai && !availableServices.anthropic && !availableServices.perplexity)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select AI Model" />
              </SelectTrigger>
              <SelectContent>
                {availableServices.openai && <SelectItem value="openai">OpenAI GPT-4o</SelectItem>}
                {availableServices.anthropic && <SelectItem value="anthropic">Anthropic Claude</SelectItem>}
                {availableServices.perplexity && <SelectItem value="perplexity">Perplexity</SelectItem>}
                {!availableServices.openai && !availableServices.anthropic && !availableServices.perplexity && 
                  <SelectItem value="none" disabled>No AI models available</SelectItem>}
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
                  <div className={`w-2 h-2 rounded-full mr-2 ${availableServices.openai ? 'bg-green-500' : 'bg-red-500'}`}></div>
                  <span>OpenAI</span>
                </div>
                <div className="flex items-center">
                  <div className={`w-2 h-2 rounded-full mr-2 ${availableServices.anthropic ? 'bg-green-500' : 'bg-red-500'}`}></div>
                  <span>Anthropic</span>
                </div>
                <div className="flex items-center">
                  <div className={`w-2 h-2 rounded-full mr-2 ${availableServices.perplexity ? 'bg-green-500' : 'bg-red-500'}`}></div>
                  <span>Perplexity</span>
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
            <div className="grid grid-cols-3 gap-4">
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
                  onChange={(e) => handleFileInputChange(e, 'media')}
                />
              </Button>
              
              <Button 
                variant="outline" 
                className="h-24 flex flex-col items-center justify-center" 
                onClick={handleDocumentClick}
                disabled={isAnalyzing}
              >
                <FileText className="h-8 w-8 mb-2" />
                <span>Document</span>
                <input
                  ref={documentInputRef}
                  type="file"
                  accept=".pdf,.doc,.docx,.txt"
                  style={{ display: 'none' }}
                  onChange={(e) => handleFileInputChange(e, 'document')}
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
                Supports JPG, PNG, MP4, MOV, PDF, DOC, DOCX (max 50MB)
              </p>
            </div>
          </Card>
          
          {/* Input Preview */}
          <Card className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">Input Preview</h2>
              {!uploadedMedia && !documentName && (
                <Button 
                  variant="outline" 
                  onClick={() => {
                    // Clear other inputs and focus on text
                    setUploadedMedia(null);
                    setDocumentName("");
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
                      uploadMedia(
                        mediaData, 
                        "image", 
                        sessionId, 
                        { 
                          selectedModel, 
                          maxPeople: 5 
                        }
                      ).then(response => {
                        setAnalysisProgress(100);
                        
                        if (response && response.analysisId) {
                          setAnalysisId(response.analysisId);
                        }
                        
                        if (response && response.messages && Array.isArray(response.messages)) {
                          setMessages(response.messages);
                        }
                        
                        toast({
                          title: "Analysis Complete",
                          description: "Your image has been re-analyzed with " + 
                            (selectedModel === "openai" ? "OpenAI" : selectedModel === "anthropic" ? "Anthropic" : "Perplexity"),
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
                  Re-Analyze with {selectedModel === "openai" ? "OpenAI" : selectedModel === "anthropic" ? "Anthropic" : "Perplexity"}
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
                            
                            // Add provider info
                            formattedTranscript += `*Transcription provided by ${data.audioTranscription.provider || 'AI transcription service'}*\n\n`;
                            
                            // Add timestamps and text if available
                            if (data.audioTranscription.transcription?.utterances) {
                              data.audioTranscription.transcription.utterances.forEach((utterance: any) => {
                                const start = new Date(utterance.start * 1000).toISOString().substr(14, 5);
                                const end = new Date(utterance.end * 1000).toISOString().substr(14, 5);
                                formattedTranscript += `**[${start} - ${end}]** ${utterance.text}\n\n`;
                              });
                            } else if (data.audioTranscription.transcription?.words) {
                              // Alternative format with words
                              let lastTimestamp = 0;
                              let currentParagraph = "";
                              
                              data.audioTranscription.transcription.words.forEach((word: any, index: number) => {
                                // Start a new paragraph every 15 seconds
                                if (word.start > lastTimestamp + 15 && currentParagraph) {
                                  const paragraphTime = new Date(lastTimestamp * 1000).toISOString().substr(14, 5);
                                  formattedTranscript += `**[${paragraphTime}]** ${currentParagraph}\n\n`;
                                  currentParagraph = word.text + " ";
                                  lastTimestamp = word.start;
                                } else {
                                  currentParagraph += word.text + " ";
                                }
                                
                                // Add final paragraph
                                if (index === data.audioTranscription.transcription.words.length - 1 && currentParagraph) {
                                  const paragraphTime = new Date(lastTimestamp * 1000).toISOString().substr(14, 5);
                                  formattedTranscript += `**[${paragraphTime}]** ${currentParagraph}\n\n`;
                                }
                              });
                            } else if (data.audioTranscription.text) {
                              // Simple text without timestamps
                              formattedTranscript += data.audioTranscription.text;
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
                  onClick={() => {
                    if (mediaData) {
                      // Clear messages for new analysis
                      setMessages([]);
                      setIsAnalyzing(true);
                      setAnalysisProgress(0);
                      
                      // Use the stored media data directly
                      uploadMedia(
                        mediaData, 
                        "video", 
                        sessionId, 
                        { 
                          selectedModel, 
                          maxPeople: 5 
                        }
                      ).then(response => {
                        setAnalysisProgress(100);
                        
                        if (response && response.analysisId) {
                          setAnalysisId(response.analysisId);
                        }
                        
                        if (response && response.messages && Array.isArray(response.messages)) {
                          setMessages(response.messages);
                        }
                        
                        toast({
                          title: "Analysis Complete",
                          description: "Your video has been re-analyzed with " + 
                            (selectedModel === "openai" ? "OpenAI" : selectedModel === "anthropic" ? "Anthropic" : "Perplexity"),
                        });
                      }).catch(error => {
                        toast({
                          variant: "destructive",
                          title: "Error",
                          description: "Failed to re-analyze video. Please try again.",
                        });
                      }).finally(() => {
                        setIsAnalyzing(false);
                      });
                    }
                  }}
                  className="w-full"
                  disabled={isAnalyzing || !mediaData}
                >
                  Re-Analyze with {selectedModel === "openai" ? "OpenAI" : selectedModel === "anthropic" ? "Anthropic" : "Perplexity"}
                </Button>
              </div>
            )}
            
            {documentName && (
              <div className="space-y-4">
                <div className="p-4 bg-muted rounded-lg flex items-center">
                  <FileText className="w-6 h-6 mr-2" />
                  <span>{documentName}</span>
                </div>
                <div className="text-center text-sm text-muted-foreground mb-4">
                  Document content will be analyzed for personality insights
                </div>
                
                {/* Re-analyze with current model button */}
                <Button 
                  onClick={() => {
                    // Here we should trigger re-analysis of the current document
                    // with the currently selected model, but we need actual document data
                    // Since we don't store the file data after upload, we'll need to prompt
                    // user to re-upload the file
                    toast({
                      title: "Re-upload Required",
                      description: "Please re-upload the document to analyze with the new model.",
                    });
                    // Clear document name to allow re-upload
                    setDocumentName("");
                    // Focus on document upload
                    handleDocumentClick();
                  }}
                  className="w-full"
                  disabled={isAnalyzing}
                >
                  Re-Analyze with {selectedModel === "openai" ? "OpenAI" : selectedModel === "anthropic" ? "Anthropic" : "Perplexity"}
                </Button>
              </div>
            )}
            
            {!uploadedMedia && !documentName && (
              <form onSubmit={handleTextSubmit} className="space-y-4">
                <Textarea
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  onKeyDown={(e) => handleKeyPress(e, handleTextSubmit)}
                  placeholder="Type or paste text to analyze..."
                  className="min-h-[250px] resize-y"
                  disabled={isAnalyzing}
                />
                <Button 
                  type="submit" 
                  className="w-full" 
                  disabled={!textInput.trim() || isAnalyzing}
                >
                  Analyze Text
                </Button>
              </form>
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
                      setDocumentName("");
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
                  {/* Download PDF button */}
                  {analysisId && (
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="flex items-center gap-2"
                      onClick={() => downloadAnalysis(analysisId, 'pdf')}
                    >
                      <Download className="h-4 w-4" />
                      <span>PDF</span>
                    </Button>
                  )}
                  
                  {/* Download DOCX button */}
                  {analysisId && (
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="flex items-center gap-2"
                      onClick={() => downloadAnalysis(analysisId, 'docx')}
                    >
                      <File className="h-4 w-4" />
                      <span>DOCX</span>
                    </Button>
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
              {messages.length === 0 ? (
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
                    {messages.filter(message => message.role === "user" || (message.role === "assistant" && messages.some(m => m.role === "user"))).map((message, index) => (
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