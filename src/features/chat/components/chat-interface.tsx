// src/features/chat/components/chat-interface.tsx
'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useChatStore, Message } from '../utils/store';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { format } from 'date-fns';
import { Trash2, ArrowUp, Microscope, FileText, Activity, Pill } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { useRouter } from 'next/navigation';
import ReactMarkdown from 'react-markdown';
import rehypeSanitize from 'rehype-sanitize';
import { formatMarkdownResponse } from '@/lib/format-markdown';

interface ChatInterfaceProps {
  type: 'clinical' | 'literature' | 'symptom' | 'drug';
  sessionId?: string;
}

export default function ChatInterface({ type, sessionId }: ChatInterfaceProps) {
  const {
    sessions,
    activeSession,
    activeSessionId,
    createSession,
    setActiveSessionId,
    addMessage,
    clearSession,
    deleteSession
  } = useChatStore();
  
  const router = useRouter();
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const initRef = useRef(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  
  // Improved initialization logic with ref to prevent loops
  useEffect(() => {
    // Skip if already initialized
    if (initRef.current) return;
    
    const initializeSession = async () => {
      // Set flag to prevent repeated initialization
      initRef.current = true;
      
      // If sessionId is provided
      if (sessionId) {
        // Check if this session exists
        const sessionExists = sessions.some(s => s.id === sessionId);
        if (sessionExists) {
          // Set as active if it exists
          setActiveSessionId(sessionId);
        } else if (type === 'clinical') {
          // For clinical, find existing clinical sessions first
          const clinicalSessions = sessions.filter(s => s.type === 'clinical');
          if (clinicalSessions.length > 0) {
            // Use first existing clinical session
            setActiveSessionId(clinicalSessions[0].id);
            router.replace(`/dashboard/chat/clinical/${clinicalSessions[0].id}`);
          } else {
            // Only create new session if none exist
            const newId = createSession('clinical');
            router.replace(`/dashboard/chat/clinical/${newId}`);
          }
        } else {
          // For other types, create new session
          const newId = createSession(type);
          router.replace(`/dashboard/chat/${type}/${newId}`);
        }
      } else {
        // No sessionId provided, handle by type
        if (type === 'clinical') {
          // For clinical, find existing clinical sessions first
          const clinicalSessions = sessions.filter(s => s.type === 'clinical');
          if (clinicalSessions.length > 0) {
            // Use first existing clinical session
            setActiveSessionId(clinicalSessions[0].id);
            router.replace(`/dashboard/chat/clinical/${clinicalSessions[0].id}`);
          } else {
            // Only create new if none exist
            const newId = createSession('clinical');
            router.replace(`/dashboard/chat/clinical/${newId}`);
          }
        } else {
          // For other types, check for existing sessions
          const typeSessions = sessions.filter(s => s.type === type);
          if (typeSessions.length > 0) {
            setActiveSessionId(typeSessions[0].id);
            router.replace(`/dashboard/chat/${type}/${typeSessions[0].id}`);
          } else {
            const newId = createSession(type);
            router.replace(`/dashboard/chat/${type}/${newId}`);
          }
        }
      }
    };
    
    initializeSession();
  }, [type, sessionId, sessions, setActiveSessionId, createSession, router]);
  

  useEffect(() => {
    if (activeSession && activeSession.messages.length === 0 && activeSessionId) {
      // Add appropriate welcome message based on type
      let welcomeMessage = '';
      switch (type) {
        case 'clinical':
          welcomeMessage = 'Welcome to Clinical Case Analysis! Describe a clinical case or patient scenario for analysis.';
          break;
        case 'literature':
          welcomeMessage = 'Welcome to Medical Literature Review! Ask about recent medical research or specific conditions.';
          break;
        case 'symptom':
          welcomeMessage = 'Welcome to Symptom Analysis! Describe symptoms for potential causes and recommendations.';
          break;
        case 'drug':
          welcomeMessage = 'Welcome to Drug Interaction Analysis! Enter medications to check for potential interactions.';
          break;
        default:
          welcomeMessage = 'Welcome! How can I assist you today?';
      }
      
      addMessage(activeSessionId, {
        content: welcomeMessage,
        role: 'assistant'
      });
    }
  }, [activeSession, activeSessionId, addMessage, type]);

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
  
// in the handleSubmit function, improve the error handling and response processing:

const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();
  
  
  if (!message.trim() || isLoading || !activeSessionId) return;
  
  // Add user message
  addMessage(activeSessionId, {
    content: message,
    role: 'user'
  });
  
  // Clear input
  setMessage('');
  
  // Focus back on textarea
  textareaRef.current?.focus();
  
  // Set loading state
  setIsLoading(true);
  
  try {
    console.log(`Sending request to /api/chat/${type}`, {
      message,
      sessionId: activeSessionId
    });
    
    // Connect to your API endpoint that talks to Python backend
    const response = await fetch(`/api/chat/${type}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: message,
        sessionId: activeSessionId
      }),
    });
    
    if (!response.ok) {
      console.error(`API returned status: ${response.status}`);
      throw new Error(`Failed to get response: ${response.status}`);
    }
    
    const data = await response.json();
    console.log("Response from API:", data);
    
    // Add assistant response from Python backend
    const responseContent = data.response || data.message || "I couldn't process that request properly. Please try again.";
    console.log("Adding response to chat:", responseContent);
    
    addMessage(activeSessionId, {
      content: formatMarkdownResponse(responseContent),
      role: 'assistant',
      metadata: data.show_booking ? { showBooking: true, specialists: data.specialists } : undefined
    });

    const isDietQuery = /diet|nutrition|meal plan|eating plan|food plan|healthy eating|weight loss plan/i.test(message);

if (isDietQuery) {
  // Show diet form instead of sending to backend immediately
  setShowDietForm(true);
  
  // Add a prompt message from the assistant
  addMessage(activeSessionId, {
    content: "I'd be happy to help you create a personalized diet plan! Please complete the form below to get started.",
    role: 'assistant',
    metadata: { showDietForm: true }
  });
  
  setIsLoading(false);
  return;
}
    
    // If there are specialists data and booking UI should be shown
    if (data.show_booking && data.specialists) {
      // You can handle showing the booking UI here if needed
      console.log("Specialists data received:", data.specialists);
    }
  } catch (error) {
    console.error('Error getting AI response:', error);
    // Add error message
    addMessage(activeSessionId, {
      content: 'Sorry, I encountered an error processing your request. Please try again.',
      role: 'assistant'
    });
  } finally {
    setIsLoading(false);
  }
};
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };
  
  const getChatTitle = () => {
    switch (type) {
      case 'clinical':
        return 'Clinical Case Analysis';
      case 'literature':
        return 'Medical Literature Review';
      case 'symptom':
        return 'Symptom Analysis';
      case 'drug':
        return 'Drug Interaction';
      default:
        return 'AI Chat';
    }
  };
  
  const getChatIcon = () => {
    switch (type) {
      case 'clinical':
        return 'CA';
      case 'literature':
        return 'LR';
      case 'symptom':
        return 'SA';
      case 'drug':
        return 'DI';
      default:
        return 'AI';
    }
  };

  return (
    <div className="flex h-full flex-col bg-gray-50 dark:bg-gray-900">
      <div className="border-b bg-white dark:bg-gray-800 px-4 py-3 shadow-sm">
        <div className="flex items-center">
          <div className="flex items-center">
            {type === 'clinical' && <Microscope className="h-5 w-5 text-primary mr-2" />}
            {type === 'literature' && <FileText className="h-5 w-5 text-primary mr-2" />}
            {type === 'symptom' && <Activity className="h-5 w-5 text-primary mr-2" />}
            {type === 'drug' && <Pill className="h-5 w-5 text-primary mr-2" />}
            <h3 className="font-medium">{getChatTitle()}</h3>
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
      
      <div className="flex-1 overflow-hidden" ref={scrollAreaRef}>
        <ScrollArea className="h-[calc(100vh-10rem)] w-full">
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
                      <AvatarFallback className="bg-primary text-primary-foreground">{getChatIcon()}</AvatarFallback>
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
  {msg.content ? (
    <ReactMarkdown 
      rehypePlugins={[rehypeSanitize]}
      components={{
        h2: ({node, ...props}) => <h2 className="text-primary font-medium mt-4 mb-2" {...props} />,
        h3: ({node, ...props}) => <h3 className="font-medium mt-3 mb-1" {...props} />,
        p: ({node, ...props}) => <p className="mb-2" {...props} />,
        ul: ({node, ...props}) => <ul className="pl-5 mb-2" {...props} />,
        li: ({node, ...props}) => <li className="mb-1" {...props} />
      }}
    >
      {msg.content}
    </ReactMarkdown>
  ) : (
    <p className="text-muted-foreground italic">No content available</p>
  )}
  
  {/* Show diet plan form if metadata indicates */}
  {msg.metadata?.showDietForm && !showDietForm && (
    <Button 
      onClick={() => setShowDietForm(true)}
      className="mt-2"
    >
      <Salad className="mr-2 h-4 w-4" />
      Create Personalized Diet Plan
    </Button>
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
          </div>
        </ScrollArea>
      </div>
      
      <div className="border-t bg-white dark:bg-gray-800 p-4 shadow-sm">
        <form onSubmit={handleSubmit} className="flex w-full space-x-2">
          <Textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={`Enter your ${type} query...`}
            className="flex-1 min-h-10 resize-none"
            disabled={isLoading || !activeSessionId}
          />
          
          <Button 
            type="submit" 
            disabled={!message.trim() || isLoading || !activeSessionId}
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