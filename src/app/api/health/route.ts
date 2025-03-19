// src/app/api/health/route.ts
import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5002';
    
    console.log(`Testing connection to Python backend at: ${pythonBackendUrl}`);
    
    const response = await fetch(`${pythonBackendUrl}/status`, {
      method: 'GET',
      cache: 'no-store',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      console.error(`Python backend health check failed: ${response.status}`);
      return NextResponse.json(
        { status: 'error', message: 'Python backend health check failed' },
        { status: 503 }
      );
    }
    
    const data = await response.json();
    console.log("Python backend status:", data);
    
    return NextResponse.json({
      status: 'ok',
      nextjs: 'healthy',
      python: data.status || 'connected'
    });
    
  } catch (error) {
    console.error('Error connecting to Python backend:', error);
    return NextResponse.json(
      { status: 'error', message: 'Failed to connect to Python backend' },
      { status: 503 }
    );
  }
}