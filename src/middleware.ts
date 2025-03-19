// src/middleware.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// This function can be marked `async` if using `await` inside
export function middleware(request: NextRequest) {
  // Simply pass through all requests without authentication checks
  return NextResponse.next();
}

// Optional: configure middleware to run only on specific paths if needed
export const config = {
  matcher: []  // Empty array means middleware won't run on any paths
};