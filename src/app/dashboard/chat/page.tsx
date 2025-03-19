import PageContainer from '@/components/layout/page-container';
import UnifiedChatInterface from '@/features/chat/components/unified-chat-interface';

export const metadata = {
  title: 'Healthcare Dashboard - AI Assistant'
};

export default function ChatPage() {
  return (
    <PageContainer>
      <div className="space-y-4">
        <h2 className="text-3xl font-bold tracking-tight">AI Medical Assistant</h2>
        <p className="text-muted-foreground">
          Chat with our AI assistant for clinical cases, medical literature, symptoms, drug interactions, and image analysis
        </p>
        <UnifiedChatInterface />
      </div>
    </PageContainer>
  );
}