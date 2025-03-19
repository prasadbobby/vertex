import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const bookingData = await req.json();
    
    // Validate required fields
    if (!bookingData.doctor_id || !bookingData.doctor_name || !bookingData.slot_id || !bookingData.email) {
      return NextResponse.json(
        { status: 'error', message: 'Missing required booking information' },
        { status: 400 }
      );
    }
    
    // Connect to Python backend
    const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5002';
    
    // Forward the booking request to the Python backend
    const response = await fetch(`${pythonBackendUrl}/book-appointment`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(bookingData),
    });
    
    if (!response.ok) {
      throw new Error(`Python backend returned status: ${response.status}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
    
  } catch (error: any) {
    console.error('Error booking appointment:', error);
    return NextResponse.json(
      { status: 'error', message: error.message || 'Failed to book appointment' },
      { status: 500 }
    );
  }
}