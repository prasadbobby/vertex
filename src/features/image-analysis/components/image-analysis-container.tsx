// src/features/image-analysis/components/image-analysis-container.tsx
'use client';
import { useEffect, useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Textarea } from '@/components/ui/textarea';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { useChatStore, Message } from '../../../features/chat/utils/store';
import { useRouter } from 'next/navigation';
import { Camera, SendIcon, ImageIcon, XIcon, Trash2 } from 'lucide-react';
import { format } from 'date-fns';
import CameraCapture from './camera-capture';
import ImageAnalysisSidebar from './image-analysis-sidebar';
import ReactMarkdown from 'react-markdown';
import rehypeSanitize from 'rehype-sanitize';
import { formatMarkdownResponse } from '@/lib/format-markdown';
import { extractTextFromHtml } from '@/lib/html-utils';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

interface ImageAnalysisContainerProps {
    type: 'image';
    sessionId?: string;
}

export default function ImageAnalysisContainer({ type, sessionId }: ImageAnalysisContainerProps) {
    const {
        sessions,
        activeSession,
        activeSessionId,
        createSession,
        setActiveSessionId,
        addMessage,
        clearSession
    } = useChatStore();

    const router = useRouter();
    const [message, setMessage] = useState('');
    const [showCamera, setShowCamera] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [uploadedImage, setUploadedImage] = useState<string | null>(null);
    const [uploadedFile, setUploadedFile] = useState<File | null>(null);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const initRef = useRef(false);

    // Initialize session
    useEffect(() => {
        if (initRef.current) return;

        const initializeSession = async () => {
            initRef.current = true;

            if (sessionId) {
                const sessionExists = sessions.some(s => s.id === sessionId);
                if (sessionExists) {
                    setActiveSessionId(sessionId);
                } else {
                    const newId = createSession(type);
                    router.replace(`/dashboard/image-analysis/${newId}`);
                }
            } else {
                const imageSessions = sessions.filter(s => s.type === type);
                if (imageSessions.length > 0) {
                    setActiveSessionId(imageSessions[0].id);
                    router.replace(`/dashboard/image-analysis/${imageSessions[0].id}`);
                } else {
                    const newId = createSession(type);
                    router.replace(`/dashboard/image-analysis/${newId}`);
                }
            }
        };

        initializeSession();
    }, [type, sessionId, sessions, setActiveSessionId, createSession, router]);

    // Improved scroll to bottom when messages change
    useEffect(() => {
        if (messagesEndRef.current) {
            // Use a small timeout to ensure the DOM has updated
            const timeoutId = setTimeout(() => {
                messagesEndRef.current?.scrollIntoView({ 
                    behavior: 'smooth',
                    block: 'end'
                });
            }, 100);
            
            return () => clearTimeout(timeoutId);
        }
    }, [activeSession?.messages]);

    // Scroll to bottom when camera state changes
    useEffect(() => {
        const timeoutId = setTimeout(() => {
            messagesEndRef.current?.scrollIntoView({ 
                behavior: 'smooth',
                block: 'end'
            });
        }, 150);
        
        return () => clearTimeout(timeoutId);
    }, [showCamera]);

    // Initialize with welcome message if this is a new session
    useEffect(() => {
        if (activeSession && activeSession.messages.length === 0 && activeSessionId) {
          addMessage(activeSessionId, {
            content: 'Welcome to Image Analysis! Upload a medical image or take a photo to begin analysis.',
            role: 'assistant'
          });
        }
      }, [activeSession, activeSessionId, addMessage]);

    // After uploading an image, focus the textarea
    useEffect(() => {
        if (uploadedImage) {
            const timeoutId = setTimeout(() => {
                const textarea = document.querySelector('textarea');
                if (textarea) {
                    textarea.focus();
                }
            }, 100);
            return () => clearTimeout(timeoutId);
        }
    }, [uploadedImage]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setErrorMessage(null);

        if ((!message.trim() && !uploadedImage) || isAnalyzing || !activeSessionId) return;

        // Handle image upload with or without text
        if (uploadedImage && uploadedFile) {
            // Create FormData to send image
            const formData = new FormData();
            formData.append('image', uploadedFile);
            formData.append('prompt', message || "Analyze this medical image");
            
            // User message content - include the text if provided
            const userMessageContent = message.trim() 
                ? `Uploaded image with query: ${message}`
                : `Uploaded image: ${uploadedFile.name}`;
            
            // Add user message with image
            addMessage(activeSessionId, {
                content: userMessageContent,
                role: 'user',
                metadata: { imageUrl: uploadedImage }
            });

            // Clear the uploaded image and text
            setUploadedImage(null);
            setUploadedFile(null);
            setMessage('');

            // Set analyzing state
            setIsAnalyzing(true);
            
            try {
                // Send to your API endpoint that connects to Python backend
                const response = await fetch('/api/image-analysis', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error(`Failed to analyze image: ${response.status}`);
                }
                
                const result = await response.json();
                console.log("Image analysis result:", result);
                
                // Extract the text content - check all possible response fields
                let analysisContent = '';
                
                // First try raw_response (plain text)
                if (result.raw_response && typeof result.raw_response === 'string') {
                    analysisContent = result.raw_response;
                }
                // Then try response (might be HTML)
                else if (result.response && typeof result.response === 'string') {
                    // If it's HTML, extract just the text content
                    analysisContent = extractTextFromHtml(result.response);
                }
                
                // If we still don't have content, use a fallback message
                if (!analysisContent.trim()) {
                    analysisContent = "The analysis was completed, but no detailed information was returned.";
                }
                
                // Add AI response
                addMessage(activeSessionId, {
                    content: analysisContent,
                    role: 'assistant'
                });
            } catch (error: any) {
                console.error('Error analyzing image:', error);
                setErrorMessage(`Error: ${error.message || 'Failed to analyze image'}`);
                addMessage(activeSessionId, {
                    content: 'Sorry, I encountered an error analyzing your image. Please try again.',
                    role: 'assistant'
                });
            } finally {
                setIsAnalyzing(false);
            }
        } else if (message.trim()) {
            // Text-only message about a previously uploaded image
            addMessage(activeSessionId, {
                content: message,
                role: 'user'
            });

            // Clear input
            setMessage('');

            // Set analyzing state
            setIsAnalyzing(true);

            try {
                // Send text question about the image
                const response = await fetch('/api/image-analysis/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        question: message,
                        sessionId: activeSessionId
                    }),
                });
                
                if (!response.ok) {
                    throw new Error('Failed to get response');
                }
                
                const data = await response.json();
                
                // Add AI response
                addMessage(activeSessionId, {
                    content: data.response || 'I couldn\'t process that question properly. Please try again.',
                    role: 'assistant'
                });
            } catch (error: any) {
                console.error('Error getting image chat response:', error);
                setErrorMessage(`Error: ${error.message || 'Failed to process question'}`);
                addMessage(activeSessionId, {
                    content: 'Sorry, I encountered an error processing your question. Please try again.',
                    role: 'assistant'
                });
            } finally {
                setIsAnalyzing(false);
            }
        }
    };

    const handleCameraCapture = (imageSrc: string) => {
        setUploadedImage(imageSrc);
        
        // Convert base64 to file
        const byteString = atob(imageSrc.split(',')[1]);
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }
        const blob = new Blob([ab], { type: 'image/jpeg' });
        const file = new File([blob], "camera-capture.jpg", { type: 'image/jpeg' });
        
        setUploadedFile(file);
        setShowCamera(false);
    };

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            const file = e.target.files[0];
            const imageUrl = URL.createObjectURL(file);
            setUploadedImage(imageUrl);
            setUploadedFile(file);
        }
    };

    return (
        <div className="flex h-[calc(100vh-4rem)]">
            {/* Left Sidebar - Sessions */}
            <ImageAnalysisSidebar type={type} />

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col bg-gray-50 dark:bg-gray-900">
                <div className="border-b bg-white dark:bg-gray-800 px-4 py-3 shadow-sm">
                    <div className="flex items-center">
                        <div className="flex items-center">
                            <ImageIcon className="h-5 w-5 text-primary mr-2" />
                            <h3 className="font-medium">Image Analysis</h3>
                        </div>
                        <div className="ml-auto flex space-x-2">
                            <TooltipProvider>
                                <Tooltip>
                                    <TooltipTrigger asChild>
                                        <Button 
                                            variant="outline" 
                                            size="icon"
                                            onClick={() => activeSessionId && clearSession(activeSessionId)}
                                            disabled={!activeSessionId || !activeSession?.messages.length}
                                        >
                                            <Trash2 className="h-4 w-4" />
                                        </Button>
                                    </TooltipTrigger>
                                    <TooltipContent>Clear Chat</TooltipContent>
                                </Tooltip>
                            </TooltipProvider>
                        </div>
                    </div>
                </div>

                <div className="flex-1 overflow-hidden">
                    <ScrollArea className="h-[calc(100vh-10rem)] w-full">
                        {errorMessage && (
                            <div className="p-2 m-4 text-sm text-red-800 bg-red-100 rounded-md">
                                {errorMessage}
                                <button 
                                    className="ml-2 text-red-900 underline"
                                    onClick={() => setErrorMessage(null)}
                                >
                                    Dismiss
                                </button>
                            </div>
                        )}

                        <div className="flex flex-col space-y-4 p-4 pb-10">
                            {activeSession?.messages.map((msg: Message) => (
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
                                                <AvatarFallback className="bg-primary text-primary-foreground">IA</AvatarFallback>
                                            </Avatar>
                                        )}
                                        <div className="space-y-2">
                                            {/* Handle both direct imageUrl and stored imageKey */}
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
                                                {msg.content.trim() ? (
                                                    <ReactMarkdown rehypePlugins={[rehypeSanitize]}>
                                                        {msg.content}
                                                    </ReactMarkdown>
                                                ) : (
                                                    <p className="text-muted-foreground italic">
                                                        Empty response received
                                                    </p>
                                                )}
                                            </div>
                                            <div className="text-xs opacity-50">
                                                {format(new Date(msg.createdAt), 'h:mm a')}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))}
                            <div ref={messagesEndRef} />

                            {/* Camera interface when active */}
                            {showCamera && (
                                <div className="fixed inset-x-0 bottom-16 z-50 bg-background p-4 shadow-lg border-t">
                                    <div className="relative">
                                        <Button
                                            className="absolute right-0 top-0 z-10"
                                            size="icon"
                                            variant="ghost"
                                            onClick={() => setShowCamera(false)}
                                        >
                                            <XIcon className="h-4 w-4" />
                                        </Button>
                                        <CameraCapture onCapture={handleCameraCapture} />
                                    </div>
                                </div>
                            )}
                        </div>
                    </ScrollArea>
                </div>

                <div className="border-t bg-white dark:bg-gray-800 p-4 shadow-sm">
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
                                <div className="ml-2 text-xs text-green-600 font-medium">
                                    Image ready for analysis
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
                            <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                onClick={() => setShowCamera(!showCamera)}
                            >
                                <Camera className="h-5 w-5" />
                            </Button>

                            <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                onClick={() => fileInputRef.current?.click()}
                            >
                                <ImageIcon className="h-5 w-5" />
                                <input
                                    type="file"
                                    ref={fileInputRef}
                                    className="hidden"
                                    accept="image/*"
                                    onChange={handleFileUpload}
                                />
                            </Button>
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
                                : "Type a message or upload an image..."}
                            className="flex-1 min-h-10 resize-none"
                        />

                        <Button
                            type="submit"
                            disabled={isAnalyzing || (!message.trim() && !uploadedImage) || !activeSessionId}
                            size="icon"
                        >
                            {isAnalyzing ? (
                                <div className="animate-spin">⟳</div>
                            ) : (
                                <SendIcon className="h-4 w-4" />
                            )}
                            <span className="sr-only">Send</span>
                        </Button>
                    </form>
                </div>
            </div>
        </div>
    );
}