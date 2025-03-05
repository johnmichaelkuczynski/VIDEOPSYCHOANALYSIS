import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { nanoid } from "nanoid";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { uploadImage, sendMessage } from "@/lib/api";
import { Upload, Send, FileImage } from "lucide-react";

export default function Home() {
  const { toast } = useToast();
  const [sessionId] = useState(() => nanoid());
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [input, setInput] = useState("");

  const queryClient = useQueryClient();

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const reader = new FileReader();
      const imageData = await new Promise<string>((resolve) => {
        reader.onload = (e) => resolve(e.target?.result as string);
        reader.readAsDataURL(file);
      });
      return uploadImage(imageData, sessionId);
    },
    onSuccess: (data) => {
      setMessages([{ role: "assistant", content: data.personalityInsights.summary }]);
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

  return (
    <div className="container mx-auto p-4 max-w-4xl">
      <h1 className="text-4xl font-bold text-center mb-8">AI Personality Analysis</h1>
      
      <Card className="mb-8">
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
      </Card>

      <Card className="mb-4">
        <ScrollArea className="h-[400px] p-4">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`mb-4 p-4 rounded-lg ${
                msg.role === "assistant"
                  ? "bg-primary/10 ml-4"
                  : "bg-muted mr-4"
              }`}
            >
              {msg.content}
            </div>
          ))}
        </ScrollArea>
      </Card>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          disabled={uploadMutation.isPending || chatMutation.isPending}
        />
        <Button
          type="submit"
          disabled={!input.trim() || uploadMutation.isPending || chatMutation.isPending}
        >
          <Send className="w-4 h-4" />
        </Button>
      </form>
    </div>
  );
}
