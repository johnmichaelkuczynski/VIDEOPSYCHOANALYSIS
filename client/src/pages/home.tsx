import { useState, useCallback } from "react";
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
import { uploadImage, sendMessage, shareAnalysis } from "@/lib/api";
import { Upload, Send, FileImage, Share2 } from "lucide-react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";

const shareSchema = z.object({
  senderEmail: z.string().email("Please enter a valid email"),
  recipientEmail: z.string().email("Please enter a valid email"),
});

export default function Home() {
  const { toast } = useToast();
  const [sessionId] = useState(() => nanoid());
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [input, setInput] = useState("");
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [analysisId, setAnalysisId] = useState<number | null>(null);
  const [isShareDialogOpen, setIsShareDialogOpen] = useState(false);
  const [emailServiceAvailable, setEmailServiceAvailable] = useState(false);

  const queryClient = useQueryClient();

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const reader = new FileReader();
      const imageData = await new Promise<string>((resolve) => {
        reader.onload = (e) => resolve(e.target?.result as string);
        reader.readAsDataURL(file);
      });
      setUploadedImage(imageData);
      return uploadImage(imageData, sessionId);
    },
    onSuccess: (data) => {
      setAnalysisId(data.id);
      setEmailServiceAvailable(data.emailServiceAvailable);
      const analysis = data.personalityInsights;
      const detailedAnalysis = data.personalityInsights.detailed_analysis;

      // Format the detailed analysis sections
      const formattedContent = `
Personality Analysis Summary:
${analysis.summary}

Core Personality:
${detailedAnalysis.personality_core}

Thought Patterns:
${detailedAnalysis.thought_patterns}

Cognitive Style:
${detailedAnalysis.cognitive_style}

Professional Insights:
${detailedAnalysis.professional_insights}

Relationships:
Current Status: ${detailedAnalysis.relationships.current_status}
Parental Status: ${detailedAnalysis.relationships.parental_status}
Ideal Partner: ${detailedAnalysis.relationships.ideal_partner}

Growth Areas:
Strengths:
${detailedAnalysis.growth_areas.strengths.map((s: string) => `- ${s}`).join('\n')}

Challenges:
${detailedAnalysis.growth_areas.challenges.map((c: string) => `- ${c}`).join('\n')}

Development Path:
${detailedAnalysis.growth_areas.development_path}`;

      setMessages([{ role: "assistant", content: formattedContent }]);
      queryClient.invalidateQueries({ queryKey: ["/api/analyze"] });
    },
    onError: () => {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to analyze image. Please try again.",
      });
    },
  });

  const chatMutation = useMutation({
    mutationFn: async (content: string) => {
      return sendMessage(content, sessionId);
    },
    onSuccess: (data) => {
      setMessages((prev) => [...prev, ...data.messages]);
      queryClient.invalidateQueries({ queryKey: ["/api/chat"] });
    },
    onError: () => {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to send message. Please try again.",
      });
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
    },
    maxFiles: 1,
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
          <h2 className="text-2xl font-semibold mb-4">Upload Photo</h2>
          <div
            {...getRootProps()}
            className={`p-8 border-2 border-dashed rounded-lg text-center cursor-pointer transition-colors
              ${isDragActive ? "border-primary bg-primary/5" : "border-muted"}`}
          >
            <input {...getInputProps()} />
            {uploadMutation.isPending ? (
              <div className="animate-pulse">Analyzing image...</div>
            ) : (
              <div className="space-y-4">
                <FileImage className="w-12 h-12 mx-auto text-muted-foreground" />
                <p className="text-muted-foreground">
                  Drag & drop an image here, or click to select
                </p>
              </div>
            )}
          </div>
          {uploadedImage && (
            <div className="mt-4">
              <img 
                src={uploadedImage} 
                alt="Uploaded" 
                className="max-w-full h-auto rounded-lg shadow-md"
              />
            </div>
          )}
        </Card>

        <Card className="p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-semibold">Analysis Results</h2>
            {messages.length > 0 && emailServiceAvailable && (
              <Dialog open={isShareDialogOpen} onOpenChange={setIsShareDialogOpen}>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm">
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
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`mb-6 p-6 rounded-lg ${
                  msg.role === "assistant"
                    ? "bg-primary/10"
                    : "bg-muted"
                }`}
              >
                <div className="prose prose-sm max-w-none">
                  {msg.content.split('\n').map((line, j) => (
                    <p key={j} className={`mb-2 ${line.startsWith('-') ? 'ml-4' : ''}`}>
                      {line}
                    </p>
                  ))}
                </div>
              </div>
            ))}
          </ScrollArea>
        </Card>
      </div>

      <Card className="p-6">
        <h2 className="text-2xl font-semibold mb-4">Chat with AI</h2>
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask questions about the analysis..."
            disabled={uploadMutation.isPending || chatMutation.isPending}
            className="flex-1"
          />
          <Button
            type="submit"
            disabled={!input.trim() || uploadMutation.isPending || chatMutation.isPending}
          >
            <Send className="w-4 h-4 mr-2" />
            Send
          </Button>
        </form>
      </Card>
    </div>
  );
}