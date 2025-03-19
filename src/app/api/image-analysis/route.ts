// src/app/api/image-analysis/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const image = formData.get('image') as File;
    const prompt = formData.get('prompt') as string;
    
    if (!image) {
      return NextResponse.json(
        { error: 'No image provided' },
        { status: 400 }
      );
    }
    
    console.log("Image analysis request:", {
      imageName: image.name,
      imageSize: image.size,
      imageType: image.type,
      prompt: prompt || "No prompt provided"
    });
    
    // Forward to Python backend
    const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5002';
    
    // Create a new FormData object to send to the Python backend
    const pythonFormData = new FormData();
    pythonFormData.append('image', image);
    pythonFormData.append('prompt', prompt || 'Analyze this medical image');
    
    const response = await fetch(`${pythonBackendUrl}/analyze-image`, {
      method: 'POST',
      body: pythonFormData
    });
    
    if (!response.ok) {
      console.error(`Python backend returned status: ${response.status}`);
      throw new Error(`Failed to analyze image: ${response.status}`);
    }
    
    const result = await response.json();
    console.log("Response from Python backend:", {
      status: result.status,
      hasResponse: Boolean(result.response),
      responseLength: result.response ? result.response.length : 0,
      hasRawResponse: Boolean(result.raw_response),
      rawResponseLength: result.raw_response ? result.raw_response.length : 0,
      processingTime: result.processing_time
    });
    
    return NextResponse.json(result);
    
  } catch (error) {
    console.error('Error in image analysis API:', error);
    return NextResponse.json(
      { status: "error", response: 'Failed to analyze image' },
      { status: 500 }
    );
  }
}