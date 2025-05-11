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
import { Form, FormField, FormItem, FormLabel, FormControl, FormMessage, FormDescription } from "@/components/ui/form";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { 
  uploadMedia, 
  sendMessage, 
  shareAnalysis, 
  getSharedAnalysis, 
  analyzeText, 
  analyzeDocument, 
  getAllSessions, 
  clearSession, 
  updateSessionName, 
  getAllAnalysesBySession, 
  downloadAnalysis, 
  updateAnalysisTitle,
  checkAPIStatus,
  ModelType,
  MediaType
} from "@/lib/api";
import { 
  Upload, 
  Send, 
  FileImage, 
  Film, 
  Share2, 
  AlertCircle, 
  FileText, 
  File, 
  Text, 
  Download, 
  Trash2, 
  Edit, 
  RefreshCw, 
  MessageSquare
} from "lucide-react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";

// Define schemas for forms
const shareSchema = z.object({
  senderEmail: z.string().email("Please enter a valid email"),
  recipientEmail: z.string().email("Please enter a valid email"),
});

const textSchema = z.object({
  content: z.string().min(1, "Content is required"),
  selectedModel: z.enum(["openai", "anthropic", "perplexity"]).default("openai")
});

const documentSchema = z.object({
  selectedModel: z.enum(["openai", "anthropic", "perplexity"]).default("openai")
});

const mediaSchema = z.object({
  selectedModel: z.enum(["openai", "anthropic", "perplexity"]).default("openai"),
  maxPeople: z.number().min(1).max(10).default(5)
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
  const queryClient = useQueryClient();
  
  // Media states
  const [uploadedMedia, setUploadedMedia] = useState<string | null>(null);
  const [mediaType, setMediaType] = useState<MediaType>("image");
  const [analysisId, setAnalysisId] = useState<number | null>(null);
  const [isShareDialogOpen, setIsShareDialogOpen] = useState(false);
  const [emailServiceAvailable, setEmailServiceAvailable] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeTab, setActiveTab] = useState<string>("upload");
  const [documentName, setDocumentName] = useState<string>("");
  const [isMobile, setIsMobile] = useState(false);
  
  // Check for mobile viewport
  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);
  // References
  const videoRef = useRef<HTMLVideoElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Forms 
  const shareForm = useForm<z.infer<typeof shareSchema>>({
    resolver: zodResolver(shareSchema),
  });
  
  const textForm = useForm<z.infer<typeof textSchema>>({
    resolver: zodResolver(textSchema),
    defaultValues: {
      content: "",
      selectedModel: "openai"
    }
  });
  
  const documentForm = useForm<z.infer<typeof documentSchema>>({
    resolver: zodResolver(documentSchema),
    defaultValues: {
      selectedModel: "openai"
    }
  });
  
  const mediaForm = useForm<z.infer<typeof mediaSchema>>({
    resolver: zodResolver(mediaSchema),
    defaultValues: {
      selectedModel: "openai",
      maxPeople: 5
    }
  });

  // Check API status on component mount
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const status = await checkAPIStatus();
        setEmailServiceAvailable(status.sendgrid || false);
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
  
  // Text analysis form submission
  const handleTextAnalysis = useMutation({
    mutationFn: async (data: z.infer<typeof textSchema>) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        setMessages([]);
        
        const response = await analyzeText(data.content, sessionId, data.selectedModel);
        
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
    onSuccess: () => {
      toast({
        title: "Analysis Complete",
        description: "Your text has been successfully analyzed.",
      });
      textForm.reset();
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
        
        // Extract file type
        const fileExt = file.name.split('.').pop()?.toLowerCase();
        const fileType = fileExt === 'pdf' ? 'pdf' : 'docx';
        
        setAnalysisProgress(50);
        
        const selectedModel = documentForm.getValues().selectedModel;
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
    onSuccess: () => {
      toast({
        title: "Analysis Complete",
        description: "Your document has been successfully analyzed.",
      });
      documentForm.reset();
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
        
        // Set preview
        setUploadedMedia(mediaData);
        setAnalysisProgress(50);
        
        // Get form values
        const selectedModel = mediaForm.getValues().selectedModel;
        const maxPeople = mediaForm.getValues().maxPeople;
        
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
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(response.messages);
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
    onSuccess: () => {
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
      const selectedModel = mediaForm.getValues().selectedModel;
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

  // Dropzone for media files
  const onDropMedia = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      handleUploadMedia.mutate(acceptedFiles[0]);
    }
  }, [handleUploadMedia]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: onDropMedia,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif'],
      'video/*': ['.mp4', '.mov', '.webm']
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024, // 50MB limit
  });

  // Dropzone for document files
  const onDropDocument = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      handleDocumentAnalysis.mutate(acceptedFiles[0]);
    }
  }, [handleDocumentAnalysis]);

  const { 
    getRootProps: getDocumentRootProps, 
    getInputProps: getDocumentInputProps, 
    isDragActive: isDocumentDragActive 
  } = useDropzone({
    onDrop: onDropDocument,
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt']
    },
    maxFiles: 1,
    maxSize: 25 * 1024 * 1024, // 25MB limit
  });

  // Form submission handlers
  const onTextSubmit = (data: z.infer<typeof textSchema>) => {
    handleTextAnalysis.mutate(data);
  };

  const onShareSubmit = (data: z.infer<typeof shareSchema>) => {
    shareMutation.mutate(data);
  };

  // Chat message submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      // Add user message immediately to UI
      setMessages(prev => [...prev, { role: 'user', content: input }]);
      chatMutation.mutate(input);
      setInput("");
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="container mx-auto p-4 max-w-6xl">
      <h1 className="text-4xl font-bold text-center mb-8">AI Personality Analysis</h1>
      
      <Tabs 
        defaultValue="upload" 
        value={activeTab} 
        onValueChange={setActiveTab}
        className="w-full mb-6"
      >
        <TabsList className="grid w-full grid-cols-4 mb-8">
          <TabsTrigger value="upload" className="flex items-center gap-2">
            <Upload className="h-4 w-4" />
            <span>Media</span>
          </TabsTrigger>
          <TabsTrigger value="text" className="flex items-center gap-2">
            <Text className="h-4 w-4" />
            <span>Text</span>
          </TabsTrigger>
          <TabsTrigger value="document" className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            <span>Document</span>
          </TabsTrigger>
          <TabsTrigger value="chat" className="flex items-center gap-2">
            <MessageSquare className="h-4 w-4" />
            <span>Chat</span>
          </TabsTrigger>
        </TabsList>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left side - Input section */}
          <Card className="p-6">
            <TabsContent value="upload" className="space-y-4">
              <h2 className="text-2xl font-semibold mb-4">Upload Images & Videos</h2>
              
              <Form {...mediaForm}>
                <form className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <FormField
                      control={mediaForm.control}
                      name="selectedModel"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>AI Model</FormLabel>
                          <Select
                            onValueChange={field.onChange}
                            defaultValue={field.value}
                            disabled={isAnalyzing}
                          >
                            <FormControl>
                              <SelectTrigger>
                                <SelectValue placeholder="Select AI Model" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              <SelectItem value="openai">OpenAI GPT-4o</SelectItem>
                              <SelectItem value="anthropic">Anthropic Claude</SelectItem>
                              <SelectItem value="perplexity">Perplexity</SelectItem>
                            </SelectContent>
                          </Select>
                        </FormItem>
                      )}
                    />
                    
                    <FormField
                      control={mediaForm.control}
                      name="maxPeople"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Max People</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              min={1}
                              max={10}
                              {...field}
                              onChange={(e) => field.onChange(parseInt(e.target.value))}
                              disabled={isAnalyzing}
                            />
                          </FormControl>
                        </FormItem>
                      )}
                    />
                  </div>
                </form>
              </Form>
              
              <div
                {...getRootProps()}
                className={`p-8 border-2 border-dashed rounded-lg text-center cursor-pointer transition-colors
                  ${isDragActive ? "border-primary bg-primary/5" : "border-muted"}`}
              >
                <input {...getInputProps()} />
                {isAnalyzing ? (
                  <div className="space-y-4">
                    <div className="animate-pulse">Analyzing media...</div>
                    <Progress value={analysisProgress} className="w-full" />
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="flex items-center justify-center space-x-2">
                      <FileImage className="w-8 h-8 text-muted-foreground" />
                      <Film className="w-8 h-8 text-muted-foreground" />
                    </div>
                    <p className="text-muted-foreground">
                      Drag & drop an image or video, or click to select
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Supports JPG, PNG, MP4, MOV, WEBM (max 50MB)
                    </p>
                  </div>
                )}
              </div>
              
              {uploadedMedia && mediaType === "image" && (
                <div className="mt-4">
                  <img 
                    src={uploadedMedia} 
                    alt="Uploaded" 
                    className="max-w-full h-auto rounded-lg shadow-md"
                  />
                </div>
              )}
              
              {uploadedMedia && mediaType === "video" && (
                <div className="mt-4">
                  <video 
                    ref={videoRef}
                    src={uploadedMedia} 
                    controls
                    className="max-w-full h-auto rounded-lg shadow-md"
                  />
                </div>
              )}
            </TabsContent>
            
            <TabsContent value="text" className="space-y-4">
              <h2 className="text-2xl font-semibold mb-4">Text Analysis</h2>
              
              <Form {...textForm}>
                <form onSubmit={textForm.handleSubmit(onTextSubmit)} className="space-y-4">
                  <FormField
                    control={textForm.control}
                    name="selectedModel"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>AI Model</FormLabel>
                        <Select
                          onValueChange={field.onChange}
                          defaultValue={field.value}
                          disabled={isAnalyzing}
                        >
                          <FormControl>
                            <SelectTrigger>
                              <SelectValue placeholder="Select AI Model" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            <SelectItem value="openai">OpenAI GPT-4o</SelectItem>
                            <SelectItem value="anthropic">Anthropic Claude</SelectItem>
                            <SelectItem value="perplexity">Perplexity</SelectItem>
                          </SelectContent>
                        </Select>
                      </FormItem>
                    )}
                  />
                  
                  <FormField
                    control={textForm.control}
                    name="content"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Text Content</FormLabel>
                        <FormControl>
                          <Textarea
                            {...field}
                            placeholder="Enter your text for personality analysis (journal entry, conversation, etc.)"
                            className="min-h-[200px]"
                            disabled={isAnalyzing}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  
                  <Button type="submit" className="w-full" disabled={isAnalyzing || textForm.formState.isSubmitting}>
                    {isAnalyzing ? "Analyzing..." : "Analyze Text"}
                  </Button>
                  {isAnalyzing && <Progress value={analysisProgress} className="w-full mt-2" />}
                </form>
              </Form>
            </TabsContent>
            
            <TabsContent value="document" className="space-y-4">
              <h2 className="text-2xl font-semibold mb-4">Document Analysis</h2>
              
              <Form {...documentForm}>
                <form className="space-y-4">
                  <FormField
                    control={documentForm.control}
                    name="selectedModel"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>AI Model</FormLabel>
                        <Select
                          onValueChange={field.onChange}
                          defaultValue={field.value}
                          disabled={isAnalyzing}
                        >
                          <FormControl>
                            <SelectTrigger>
                              <SelectValue placeholder="Select AI Model" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            <SelectItem value="openai">OpenAI GPT-4o</SelectItem>
                            <SelectItem value="anthropic">Anthropic Claude</SelectItem>
                            <SelectItem value="perplexity">Perplexity</SelectItem>
                          </SelectContent>
                        </Select>
                      </FormItem>
                    )}
                  />
                </form>
              </Form>
              
              <div
                {...getDocumentRootProps()}
                className={`p-8 border-2 border-dashed rounded-lg text-center cursor-pointer transition-colors
                  ${isDocumentDragActive ? "border-primary bg-primary/5" : "border-muted"}`}
              >
                <input {...getDocumentInputProps()} />
                {isAnalyzing ? (
                  <div className="space-y-4">
                    <div className="animate-pulse">Analyzing document...</div>
                    <Progress value={analysisProgress} className="w-full" />
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="flex items-center justify-center">
                      <FileText className="w-8 h-8 text-muted-foreground" />
                    </div>
                    <p className="text-muted-foreground">
                      Drag & drop a document, or click to select
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Supports PDF, DOC, DOCX, TXT (max 25MB)
                    </p>
                  </div>
                )}
              </div>
              
              {documentName && (
                <div className="mt-4 p-4 bg-muted rounded-lg flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <FileText className="w-6 h-6" />
                    <span>{documentName}</span>
                  </div>
                </div>
              )}
            </TabsContent>
            
            <TabsContent value="chat" className="space-y-4">
              <h2 className="text-2xl font-semibold mb-4">Chat with AI</h2>
              <p className="text-muted-foreground mb-4">
                Discuss your analysis with the AI or ask follow-up questions.
              </p>
              
              <form onSubmit={handleSubmit} className="space-y-4">
                <Textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyPress}
                  placeholder="Type your message here..."
                  className="min-h-[120px] resize-y"
                />
                <Button 
                  type="submit" 
                  className="w-full" 
                  disabled={!input.trim() || chatMutation.isPending}
                >
                  {chatMutation.isPending ? "Sending..." : "Send Message"}
                </Button>
              </form>
            </TabsContent>
          </Card>
          
          {/* Right side - Results section */}
          <Card className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-semibold">Analysis Results</h2>
              
              {messages.length > 0 && emailServiceAvailable && (
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
            
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center space-y-4 min-h-[400px] text-center text-muted-foreground">
                <AlertCircle className="h-12 w-12" />
                <div>
                  <p className="text-lg font-medium">No analysis yet</p>
                  <p>Upload media, enter text, or select a document to analyze.</p>
                </div>
              </div>
            ) : (
              <ScrollArea className="h-[500px] pr-4">
                <div className="space-y-4">
                  {messages.map((message, index) => (
                    <div
                      key={index}
                      className={`flex flex-col p-4 rounded-lg ${
                        message.role === "user" ? "bg-primary/10 ml-8" : "bg-muted mr-8"
                      }`}
                    >
                      <span className="font-medium text-sm mb-1">
                        {message.role === "user" ? "You" : "AI Analysis"}
                      </span>
                      <div 
                        className="whitespace-pre-wrap"
                        dangerouslySetInnerHTML={{ 
                          __html: message.content
                            .replace(/\n/g, '<br/>')
                            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                        }} 
                      />
                    </div>
                  ))}
                  <div ref={messagesEndRef} />
                </div>
              </ScrollArea>
            )}
          </Card>
        </div>
      </Tabs>
    </div>
  );
}