'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { format } from 'date-fns';
import { 
  Trash2, 
  ArrowUp, 
  Camera, 
  ImageIcon, 
  X as XIcon,
  Paperclip,
  Upload
} from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import ReactMarkdown from 'react-markdown';
import rehypeSanitize from 'rehype-sanitize';
import { formatMarkdownResponse } from '@/lib/format-markdown';
import MicButton from '@/components/mic-button';
import AppointmentBooking from './appointment-booking';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu';
import CameraCapture from '@/components/camera-capture';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  createdAt: Date;
  metadata?: {
    imageUrl?: string;
    imageKey?: string;
    showBooking?: boolean;
    specialists?: any[];
    [key: string]: any;
  };
}

export default function UnifiedChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [showBooking, setShowBooking] = useState(false);
  const [specialists, setSpecialists] = useState([]);
  const [cameraActive, setCameraActive] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([
        {
          id: Date.now().toString(),
          content: 'Hello! I\'m your medical AI assistant. Ask me about clinical cases, medical literature, symptoms, drug interactions, or upload medical images for analysis.',
          role: 'assistant',
          createdAt: new Date()
        }
      ]);
    }
  }, [messages]);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, cameraActive]);

  const addMessage = (newMessage: Omit<Message, 'id' | 'createdAt'>) => {
    const messageWithId = {
      ...newMessage,
      id: Date.now().toString(),
      createdAt: new Date()
    };
    setMessages(prev => [...prev, messageWithId]);
  };

  const handleCameraCapture = (imageSrc: string) => {
    const byteString = atob(imageSrc.split(',')[1]);
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }
    const blob = new Blob([ab], { type: 'image/jpeg' });
    const file = new File([blob], "camera-capture.jpg", { type: 'image/jpeg' });
    
    setUploadedImage(imageSrc);
    setUploadedFile(file);
    setCameraActive(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if ((!message.trim() && !uploadedImage) || isLoading) return;
    
    if (uploadedImage && uploadedFile) {
      addMessage({
        content: message.trim() ? `${message} (with image)` : 'Image for analysis',
        role: 'user',
        metadata: { imageUrl: uploadedImage }
      });
      
      setMessage('');
      setIsLoading(true);
      
      try {
        const formData = new FormData();
        formData.append('image', uploadedFile);
        formData.append('prompt', message || "Analyze this medical image");
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);
        
        const response = await fetch('/api/image-analysis', {
          method: 'POST',
          body: formData,
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
          throw new Error(`Error: ${response.status}`);
        }
        
        const result = await response.json();
        
        addMessage({
          content: formatMarkdownResponse(result.response || "I've analyzed your image."),
          role: 'assistant'
        });
        
        setUploadedImage(null);
        setUploadedFile(null);
      } catch (error: any) {
        console.error('Error processing image:', error);
        let errorMessage = 'Sorry, I encountered an error analyzing your image.';
        if (error.name === 'AbortError') {
          errorMessage = 'The request took too long to complete. Our backend services might be experiencing high traffic or connectivity issues. Please try again in a moment.';
        } else if (error.message.includes('failed to connect')) {
          errorMessage = 'There seems to be a connectivity issue with our backend services. Please try again in a few minutes.';
        }
        
        addMessage({
          content: errorMessage,
          role: 'assistant'
        });
      } finally {
        setIsLoading(false);
      }
      return;
    }
    
    if (message.trim()) {
      addMessage({
        content: message,
        role: 'user'
      });
      
      setMessage('');
      setIsLoading(true);
      
      try {
        const isSymptomQuery = /symptom|pain|ache|feeling|suffering|hurts|sick/i.test(message);
        const chatType = isSymptomQuery ? 'symptom' : 'clinical';
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);
        
        const response = await fetch(`/api/chat/${chatType}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: message,
            sessionId: 'unified-session'
          }),
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
          throw new Error(`Error: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.show_booking && data.specialists) {
          setShowBooking(true);
          setSpecialists(data.specialists);
          
          addMessage({
            content: formatMarkdownResponse(data.response),
            role: 'assistant',
            metadata: {
              showBooking: true,
              specialists: data.specialists
            }
          });
        } else {
          addMessage({
            content: formatMarkdownResponse(data.response || "I'm sorry, I couldn't process your request properly. Please try again."),
            role: 'assistant'
          });
        }
      } catch (error: any) {
        console.error('Error getting response:', error);
        let errorMessage = 'Sorry, I encountered an error processing your request. Please try again.';
        if (error.name === 'AbortError') {
          errorMessage = 'The request took too long to complete. Our backend services might be experiencing connectivity issues with some external APIs. Please try again in a moment.';
        } else if (error.message.includes('failed to connect')) {
          errorMessage = 'There seems to be a connectivity issue with our backend services. Please try again in a few minutes.';
        }
        
        addMessage({
          content: errorMessage,
          role: 'assistant'
        });
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      const imageUrl = URL.createObjectURL(file);
      setUploadedImage(imageUrl);
      setUploadedFile(file);
    }
  };
  
  const handleSpeechRecognized = (text: string) => {
    setMessage(prev => prev ? `${prev} ${text}` : text);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-[calc(100vh-7rem)] w-full">
          <div className="flex flex-col space-y-4 p-4 pb-10">
            {messages.map((msg) => (
              <div 
                key={msg.id} 
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div 
                  className={`flex max-w-[80%] items-start space-x-3 rounded-lg p-4
                    ${msg.role === 'user' 
                      ? 'bg-primary text-primary-foreground' 
                      : 'bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700'}`}
                >
                  {msg.role === 'assistant' && (
                    <Avatar className="h-8 w-8 mt-1">
                      <AvatarFallback className="bg-primary text-primary-foreground">AI</AvatarFallback>
                    </Avatar>
                  )}
                  <div className="space-y-2">
                    {(msg.metadata?.imageUrl || msg.metadata?.imageKey) && (
                      <div className="mb-2 overflow-hidden rounded-md border">
                        <img 
                          src={msg.metadata?.imageUrl || (msg.metadata?.imageKey ? sessionStorage.getItem(msg.metadata.imageKey) : '')} 
                          alt="Uploaded image" 
                          className="h-auto max-h-60 w-full object-contain" 
                        />
                      </div>
                    )}
                    
                    <div className={`prose prose-sm max-w-none break-words ${msg.role === 'user' ? 'text-primary-foreground' : 'text-foreground'}`}>
                      <ReactMarkdown rehypePlugins={[rehypeSanitize]}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                    
                    {msg.metadata?.showBooking && msg.metadata?.specialists && (
                      <div className="mt-4 border-t pt-4">
                        <AppointmentBooking specialists={msg.metadata.specialists} />
                      </div>
                    )}
                    
                    <div className="text-xs opacity-50">
                      {format(new Date(msg.createdAt), 'h:mm a')}
                    </div>
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>
      </div>

      {/* Camera UI when active */}
      {cameraActive && (
        <div className="absolute inset-x-0 bottom-16 z-50 bg-background p-4 shadow-lg border-t">
          <div className="relative">
            <Button
              className="absolute right-0 top-0 z-10"
              size="icon"
              variant="ghost"
              onClick={() => setCameraActive(false)}
            >
              <XIcon className="h-4 w-4" />
            </Button>
            <CameraCapture onCapture={handleCameraCapture} />
          </div>
        </div>
      )}
 
      <div className="border-t bg-white dark:bg-gray-800 p-4 shadow-sm sticky bottom-0 left-0 right-0">
        <form onSubmit={handleSubmit} className="flex w-full space-x-2">
          {uploadedImage && (
            <div className="relative mr-2 flex items-center">
              <div className="h-10 w-10 overflow-hidden rounded-md border">
                <img
                  src={uploadedImage}
                  alt="Preview"
                  className="h-full w-full object-cover"
                />
              </div>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="absolute -right-2 -top-2 h-5 w-5 rounded-full bg-red-500 p-0 text-white hover:bg-red-600"
                onClick={() => {
                  setUploadedImage(null);
                  setUploadedFile(null);
                }}
              >
                <XIcon className="h-3 w-3" />
              </Button>
            </div>
          )}
          
          <div className="flex items-center space-x-2">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                >
                  <Paperclip className="h-5 w-5" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start">
                <DropdownMenuItem onClick={() => setCameraActive(true)}>
                  <Camera className="mr-2 h-4 w-4" />
                  Take a photo
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => fileInputRef.current?.click()}>
                  <ImageIcon className="mr-2 h-4 w-4" />
                  Upload an image
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
            
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="image/*"
              onChange={handleFileUpload}
            />
            
            <MicButton onSpeechRecognized={handleSpeechRecognized} />
          </div>
          
          <Textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
            placeholder={uploadedImage
              ? "Add a message or click Send to analyze this image..."
              : "Type your medical question here..."}
            className="flex-1 min-h-10 resize-none"
            disabled={isLoading}
          />
          
          <Button 
            type="submit" 
            disabled={isLoading || (!message.trim() && !uploadedImage)}
            className="shrink-0"
          >
            {isLoading ? (
              <div className="animate-spin">⟳</div>
            ) : (
              <ArrowUp className="h-4 w-4" />
            )}
            <span className="sr-only">Send</span>
          </Button>
        </form>
      </div>
    </div>
  );
}