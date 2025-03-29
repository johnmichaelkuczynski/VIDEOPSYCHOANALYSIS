import { useState, useCallback, useRef, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { nanoid } from "nanoid";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Form, FormField, FormItem, FormLabel, FormControl, FormMessage } from "@/components/ui/form";
import { uploadMedia, sendMessage, shareAnalysis } from "@/lib/api";
import { Upload, Send, FileImage, Film, Share2, AlertCircle } from "lucide-react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { 
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";

// Helper function to resize images to keep them under the AWS 5MB limit
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
        
        // Convert to data URL with reduced quality for JPEG to ensure smaller file size
        const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
        resolve(dataUrl);
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

const shareSchema = z.object({
  senderEmail: z.string().email("Please enter a valid email"),
  recipientEmail: z.string().email("Please enter a valid email"),
});

export default function Home({ isShareMode = false }: { isShareMode?: boolean }) {
  const { toast } = useToast();
  const [sessionId] = useState(() => nanoid());
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [input, setInput] = useState("");
  
  // Media states
  const [uploadedMedia, setUploadedMedia] = useState<string | null>(null);
  const [mediaType, setMediaType] = useState<"image" | "video">("image");
  const [analysisId, setAnalysisId] = useState<number | null>(null);
  const [isShareDialogOpen, setIsShareDialogOpen] = useState(false);
  const [emailServiceAvailable, setEmailServiceAvailable] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  // Reference for video element
  const videoRef = useRef<HTMLVideoElement>(null);

  const queryClient = useQueryClient();

  // Open share dialog when in share mode
  useEffect(() => {
    if (isShareMode && messages.length > 0 && emailServiceAvailable) {
      setIsShareDialogOpen(true);
    }
  }, [isShareMode, messages.length, emailServiceAvailable]);


  // Simulate analysis progress
  useEffect(() => {
    if (isAnalyzing) {
      const interval = setInterval(() => {
        setAnalysisProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            return 100;
          }
          return prev + 5;
        });
      }, 500);
      
      return () => clearInterval(interval);
    }
  }, [isAnalyzing]);

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      // Determine media type based on file type
      const fileType = file.type.split('/')[0];
      const isMimeTypeVideo = fileType === 'video';
      const currentMediaType = isMimeTypeVideo ? "video" : "image";
      
      setMediaType(currentMediaType);
      setIsAnalyzing(true);
      setAnalysisProgress(0);
      
      // For images, we need to resize to ensure it's under AWS Rekognition's 5MB limit
      let mediaData: string;
      
      if (currentMediaType === "image") {
        // Resize the image
        mediaData = await resizeImage(file, 1200); // Resize to max width of 1200px
      } else {
        // Read video as data URL without resizing
        const reader = new FileReader();
        mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
      }
      
      // Set previews
      setUploadedMedia(mediaData);
      
      // Upload for analysis
      return uploadMedia(mediaData, currentMediaType, sessionId);
    },
    onSuccess: (data) => {
      setIsAnalyzing(false);
      setAnalysisProgress(100);
      setAnalysisId(data.id);
      setEmailServiceAvailable(data.emailServiceAvailable);
      
      // Clear previous messages when a new analysis is completed
      setMessages([]);
      
      // Get the latest messages that were created for this session
      // This contains the properly formatted content for both single and multi-person analyses
      const messagesQuery = `/api/messages?sessionId=${sessionId}`;
      
      fetch(messagesQuery)
        .then(response => response.json())
        .then(messagesData => {
          if (messagesData.length > 0) {
            // Only show messages from the latest analysis
            const latestAnalysisId = Math.max(...messagesData.filter(msg => msg.analysisId).map(msg => msg.analysisId));
            const latestMessages = messagesData.filter(msg => msg.analysisId === latestAnalysisId);
            
            if (latestMessages.length > 0) {
              setMessages(latestMessages);
              toast({
                title: "Analysis Complete",
                description: "New analysis completed successfully."
              });
            } else {
              setMessages(messagesData); // Fallback to all messages if filtering fails
            }
          } else {
            // Fallback in case messages aren't available
            toast({
              title: "Analysis Complete",
              description: "Analysis completed, but message content could not be loaded."
            });
          }
        })
        .catch(err => {
          console.error("Failed to fetch messages:", err);
          toast({
            title: "Analysis Complete",
            description: "Analysis completed, but message content could not be loaded."
          });
        });
        
      queryClient.invalidateQueries({ queryKey: ["/api/analyze"] });
    },
    onError: (error) => {
      setIsAnalyzing(false);
      setAnalysisProgress(0);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to analyze media. Please try again.",
      });
    },
  });

  const chatMutation = useMutation({
    mutationFn: async (content: string) => {
      console.log("Sending chat message:", { content, sessionId });
      return sendMessage(content, sessionId);
    },
    onSuccess: (data) => {
      console.log("Chat response data:", data);
      
      // If we received messages, add them to the state
      if (data && data.messages && Array.isArray(data.messages)) {
        console.log("Adding new messages to state:", data.messages);
        
        // Add only the new messages that aren't already in the state
        // Get the IDs of existing messages
        const existingMessageIds = messages.map(msg => (msg as any).id).filter(Boolean);
        
        // Filter out messages that are already in the state
        const newMessages = data.messages.filter(msg => !msg.id || !existingMessageIds.includes(msg.id));
        
        if (newMessages.length > 0) {
          // Make sure each message has a valid role and content property
          const validMessages = newMessages.map(msg => ({
            ...msg,
            role: msg.role || 'assistant',
            content: msg.content || '',
          }));
          
          console.log("Final messages to add:", validMessages);
          setMessages(prev => [...prev, ...validMessages]);
          queryClient.invalidateQueries({ queryKey: ["/api/chat"] });
        } else {
          console.log("No new messages to add");
        }
      } else {
        console.warn("Received invalid message data format:", data);
      }
      
      // Check for configError that indicates an API key issue
      if (data.configError === "OPENAI_API_KEY_MISSING") {
        toast({
          variant: "destructive",
          title: "API Key Missing",
          description: "OpenAI API key is required for chat functionality. Please configure the API key.",
        });
      }
    },
    onError: (error: any) => {
      console.error("Chat error:", error);
      
      // Check if the error has a response with details
      const errorMessage = error.response?.data?.error || "Failed to send message. Please try again.";
      const configError = error.response?.data?.configError;
      
      if (configError === "OPENAI_API_KEY_MISSING") {
        toast({
          variant: "destructive",
          title: "API Key Missing",
          description: "OpenAI API key is required for chat functionality. Please configure the API key.",
        });
      } else {
        toast({
          variant: "destructive",
          title: "Error",
          description: errorMessage,
        });
      }
      
      // If we received a user message in the error response, add it to the messages
      if (error.response?.data?.messages) {
        setMessages((prev) => [...prev, ...error.response.data.messages]);
      }
    },
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

  const shareForm = useForm<z.infer<typeof shareSchema>>({
    resolver: zodResolver(shareSchema),
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      uploadMutation.mutate(acceptedFiles[0]);
    }
  }, [uploadMutation]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".jpeg", ".jpg", ".png"],
      "video/*": [".mp4", ".mov", ".webm"]
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024, // 50MB limit
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
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

  const onShareSubmit = (data: z.infer<typeof shareSchema>) => {
    shareMutation.mutate(data);
  };

  return (
    <div className="container mx-auto p-4 max-w-6xl">
      <h1 className="text-4xl font-bold text-center mb-8">AI Personality Analysis</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        <Card className="p-6">
          <h2 className="text-2xl font-semibold mb-4">Upload Media</h2>
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
          {uploadedMedia && (
            <div className="mt-4">
              {mediaType === "image" ? (
                <img 
                  src={uploadedMedia} 
                  alt="Uploaded" 
                  className="max-w-full h-auto rounded-lg shadow-md"
                />
              ) : (
                <video 
                  ref={videoRef}
                  src={uploadedMedia} 
                  controls
                  className="max-w-full h-auto rounded-lg shadow-md"
                />
              )}
            </div>
          )}
        </Card>

        <Card className="p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-semibold">Analysis Results</h2>
            {messages.length > 0 && emailServiceAvailable && (
              <Dialog open={isShareDialogOpen} onOpenChange={setIsShareDialogOpen}>
                <DialogTrigger asChild>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => {
                      window.history.pushState({}, "", "/share");
                    }}
                  >
                    <Share2 className="w-4 h-4 mr-2" />
                    Share
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
                              <Input placeholder="your@email.com" {...field} />
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
                              <Input placeholder="recipient@email.com" {...field} />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                      <Button 
                        type="submit" 
                        className="w-full"
                        disabled={shareMutation.isPending}
                      >
                        {shareMutation.isPending ? "Sending..." : "Send"}
                      </Button>
                    </form>
                  </Form>
                </DialogContent>
              </Dialog>
            )}
          </div>
          <ScrollArea className="h-[600px] pr-4">
            {messages.map((msg: any, i) => {
              console.log("Rendering message:", msg);
              
              return (
                <div
                  key={i}
                  className={`mb-6 p-6 rounded-lg ${
                    msg.role === "assistant"
                      ? "bg-primary/10"
                      : "bg-muted"
                  }`}
                >
                  <p className="text-xs text-gray-500 mb-2">
                    {msg.role === "assistant" ? "AI Assistant" : "You"}
                  </p>
                  
                  <div className="prose prose-sm max-w-none">
                    {typeof msg.content === 'string' && msg.content.split('\n').map((rawLine: string, j: number) => {
                      // Clean up the line by removing markdown symbols
                      const line = rawLine
                        .replace(/^\s*#{1,3}\s+/, '') // Remove heading markers (### Title)
                        .replace(/^\s*\*\*/, '') // Remove leading asterisks
                        .replace(/\*\*\s*$/, '') // Remove trailing asterisks
                        .replace(/\*\*/g, '') // Remove all other asterisks pairs
                        .replace(/^>\s*/, '') // Remove block quotes

                      // Check if line is a divider (─────)
                      if (line.startsWith('─')) {
                        return <hr key={j} className="my-3 border-gray-300" />;
                      }
                      
                      // Check if line is a section header with emoji (e.g., "👤 Subject 1")
                      if (/^[👤🧠🖼️📷🧾🧬💼❤️📈🤝]/.test(line)) {
                        return <h3 key={j} className="text-lg font-bold mt-4 mb-2 text-primary">{line}</h3>;
                      }
                      
                      // Check for numbered lists (e.g., "1. Item")
                      if (/^\d+\./.test(line)) {
                        return <p key={j} className="ml-8 mb-1">{line}</p>;
                      }
                      
                      // Special formatting for Growth Areas with bullet points
                      if (line.startsWith('•')) {
                        return <li key={j} className="ml-8 mb-1">{line.substring(1).trim()}</li>;
                      }
                      
                      // Handle bullet lists
                      if (line.startsWith('-') || line.startsWith('*')) {
                        return <li key={j} className="ml-8 mb-1">{line.substring(1).trim()}</li>;
                      }
                      
                      // For subsections like "Strengths:", "Challenges:", etc.
                      if (line.endsWith(':')) {
                        return <h4 key={j} className="font-semibold mt-2 mb-1">{line}</h4>;
                      }
                      
                      // Regular paragraph - clean up and display
                      return <p key={j} className="mb-2">{line}</p>;
                    })}
                  </div>
                </div>
              );
            })}
          </ScrollArea>
        </Card>
      </div>

      <Card className="p-6">
        <h2 className="text-2xl font-semibold mb-4">Chat with AI</h2>
        
        {messages.length > 0 && messages.some(msg => msg.role === "user") && messages.length === 1 && (
          <Alert variant="destructive" className="mb-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>API Key Required</AlertTitle>
            <AlertDescription>
              OpenAI API key is required for chat functionality. Please configure the API key in the server's environment variables.
            </AlertDescription>
          </Alert>
        )}
        
        <ScrollArea className="h-[400px] pr-4 mb-4">
          {messages.map((msg: any, i) => {
            console.log("Rendering message:", msg);
            
            return (
              <div
                key={i}
                className={`mb-6 p-6 rounded-lg ${
                  msg.role === "assistant"
                    ? "bg-primary/10"
                    : "bg-muted"
                }`}
              >
                <p className="text-xs text-gray-500 mb-2">
                  {msg.role === "assistant" ? "AI Assistant" : "You"}
                </p>
                
                <div className="prose prose-sm max-w-none">
                  {typeof msg.content === 'string' && msg.content.split('\n').map((rawLine: string, j: number) => {
                    // Clean up the line by removing markdown symbols
                    const line = rawLine
                      .replace(/^\s*#{1,3}\s+/, '') // Remove heading markers (### Title)
                      .replace(/^\s*\*\*/, '') // Remove leading asterisks
                      .replace(/\*\*\s*$/, '') // Remove trailing asterisks
                      .replace(/\*\*/g, '') // Remove all other asterisks pairs
                      .replace(/^>\s*/, '') // Remove block quotes

                    // Check if line is a divider (─────)
                    if (line.startsWith('─')) {
                      return <hr key={j} className="my-3 border-gray-300" />;
                    }
                    
                    // Check if line is a section header with emoji (e.g., "👤 Subject 1")
                    if (/^[👤🧠🖼️📷🧾🧬💼❤️📈🤝]/.test(line)) {
                      return <h3 key={j} className="text-lg font-bold mt-4 mb-2 text-primary">{line}</h3>;
                    }
                    
                    // Check for numbered lists (e.g., "1. Item")
                    if (/^\d+\./.test(line)) {
                      return <p key={j} className="ml-8 mb-1">{line}</p>;
                    }
                    
                    // Special formatting for Growth Areas with bullet points
                    if (line.startsWith('•')) {
                      return <li key={j} className="ml-8 mb-1">{line.substring(1).trim()}</li>;
                    }
                    
                    // Handle bullet lists
                    if (line.startsWith('-') || line.startsWith('*')) {
                      return <li key={j} className="ml-8 mb-1">{line.substring(1).trim()}</li>;
                    }
                    
                    // For subsections like "Strengths:", "Challenges:", etc.
                    if (line.endsWith(':')) {
                      return <h4 key={j} className="font-semibold mt-2 mb-1">{line}</h4>;
                    }
                    
                    // Regular paragraph - clean up and display
                    return <p key={j} className="mb-2">{line}</p>;
                  })}
                </div>
              </div>
            );
          })}
        </ScrollArea>
        
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask questions about the analysis..."
            disabled={isAnalyzing || chatMutation.isPending}
            className="flex-1"
          />
          <Button
            type="submit"
            disabled={!input.trim() || isAnalyzing || chatMutation.isPending}
          >
            <Send className="w-4 h-4 mr-2" />
            Send
          </Button>
        </form>
      </Card>
    </div>
  );
}