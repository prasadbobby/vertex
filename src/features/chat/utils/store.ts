// src/features/chat/utils/store.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type Message = {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  createdAt: Date;
  metadata?: {
    imageUrl?: string;
    imageKey?: string;
    [key: string]: any;
  };
};

export type ChatSession = {
  id: string;
  title: string;
  messages: Message[];
  type: 'clinical' | 'literature' | 'symptom' | 'drug' | 'image';
  createdAt: Date;
  updatedAt: Date;
};

interface ChatState {
  sessions: ChatSession[];
  activeSessionId: string | null;
  activeSession: ChatSession | null;
  createSession: (type: 'clinical' | 'literature' | 'symptom' | 'drug' | 'image') => string;
  setActiveSessionId: (id: string) => void;
  addMessage: (sessionId: string, message: Omit<Message, 'id' | 'createdAt'>) => void;
  clearSession: (sessionId: string) => void;
  deleteSession: (sessionId: string) => void;
  getSessionsByType: (type: 'clinical' | 'literature' | 'symptom' | 'drug' | 'image') => ChatSession[];
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,
      activeSession: null,
      
      createSession: (type) => {
        // Create a new session without restrictions for any type
        const id = Date.now().toString();
        const newSession: ChatSession = {
          id,
          title: `New ${type} session`,
          messages: [],
          type,
          createdAt: new Date(),
          updatedAt: new Date()
        };
        
        set((state) => ({
          sessions: [...state.sessions, newSession],
          activeSessionId: id,
          activeSession: newSession
        }));
        
        return id;
      },
      
      setActiveSessionId: (id) => {
        // Don't update if already the active session
        if (get().activeSessionId === id) return;
        
        const session = get().sessions.find(s => s.id === id) || null;
        set({ activeSessionId: id, activeSession: session });
      },
      
      addMessage: (sessionId: string, messageData: Omit<Message, 'id' | 'createdAt'>) => {
        const newMessage: Message = {
          id: Date.now().toString(),
          content: messageData.content,
          role: messageData.role,
          createdAt: new Date(),
          metadata: messageData.metadata
        };
        
        // If there's an image URL in metadata, store it in session storage to persist it
        if (messageData.metadata?.imageUrl) {
          // Create a key for this image
          const imageKey = `chat-image-${newMessage.id}`;
          try {
            sessionStorage.setItem(imageKey, messageData.metadata.imageUrl);
            
            // Update metadata to use the key instead of the direct URL
            newMessage.metadata = {
              ...messageData.metadata,
              imageKey
            };
          } catch (error) {
            console.error("Failed to store image in session storage:", error);
            // Keep the original imageUrl if session storage fails
          }
        }
        
        set((state) => {
          const updatedSessions = state.sessions.map(session => {
            if (session.id === sessionId) {
              // Check if this is the first user message and update title if it is
              const isFirstUserMessage = 
                session.messages.filter(m => m.role === 'user').length === 0 && 
                messageData.role === 'user';
        
              let updatedTitle = session.title;
              if (isFirstUserMessage) {
                // Extract first 30 chars for title
                const rawTitle = messageData.content.trim();
                updatedTitle = rawTitle.length > 30 
                  ? rawTitle.slice(0, 30) + '...' 
                  : rawTitle;
              }
              
              return {
                ...session,
                title: updatedTitle,
                messages: [...session.messages, newMessage],
                updatedAt: new Date()
              };
            }
            return session;
          });
          
          const activeSession = updatedSessions.find(s => s.id === sessionId) || null;
          
          return {
            sessions: updatedSessions,
            activeSession
          };
        });
      },
      
      clearSession: (sessionId) => {
        set((state) => ({
          sessions: state.sessions.map(session => 
            session.id === sessionId
              ? { ...session, messages: [], updatedAt: new Date() }
              : session
          )
        }));
      },
      
      deleteSession: (sessionId) => {
        set((state) => {
          const updatedSessions = state.sessions.filter(session => session.id !== sessionId);
          
          // If the active session is deleted, set a new active session
          let newActiveSessionId = state.activeSessionId;
          let newActiveSession = state.activeSession;
          
          if (state.activeSessionId === sessionId) {
            newActiveSessionId = updatedSessions.length > 0 ? updatedSessions[0].id : null;
            newActiveSession = newActiveSessionId 
              ? updatedSessions.find(s => s.id === newActiveSessionId) || null
              : null;
          }
          
          return {
            sessions: updatedSessions,
            activeSessionId: newActiveSessionId,
            activeSession: newActiveSession
          };
        });
      },
      
      getSessionsByType: (type) => {
        return get().sessions.filter(session => session.type === type);
      }
    }),
    {
      name: 'healthcare-chat-store',
      skipHydration: true
    }
  )
);