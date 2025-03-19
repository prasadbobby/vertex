// src/app/api/chat/[type]/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { getFallbackResponse } from '@/lib/fallback-responses';

export async function POST(
  req: NextRequest,
  { params }: { params: { type: string } }
) {
  try {
    // Properly await the params object
    const resolvedParams = await Promise.resolve(params);
    const chatType = resolvedParams.type;
    
    const { message, sessionId } = await req.json();
    
    // Connect to Python backend
    const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5002';
    
    console.log(`Sending request to Python backend: ${pythonBackendUrl}/query`);
    
    const pythonResponse = await fetch(`${pythonBackendUrl}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        type: chatType, 
        query: message 
      }),
    });
    
    if (!pythonResponse.ok) {
      console.error(`Python backend returned status: ${pythonResponse.status}`);
      throw new Error(`Python backend returned status: ${pythonResponse.status}`);
    }
    
    const data = await pythonResponse.json();
    console.log("Data received from Python backend:", data);
    
    return NextResponse.json(data);
  } 
  catch (error: any) {
    console.error('Error in chat API:', error);
    
    // Use fallback response if it's a connectivity issue
    if (error.name === 'AbortError' || 
        (error.message && error.message.includes('connect')) || 
        (error.message && error.message.includes('timeout'))) {
      
      return NextResponse.json(
        { 
          status: "partial_success", 
          response: getFallbackResponse(message),
          message: "Using fallback response due to backend connectivity issues."
        },
        { status: 200 }
      );
    }
    
    return NextResponse.json(
      { 
        status: "error", 
        response: 'Failed to process request. Our servers might be experiencing high traffic.' 
      },
      { status: 500 }
    );
  }
}