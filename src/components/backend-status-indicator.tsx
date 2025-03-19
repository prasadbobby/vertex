'use client';

import { useEffect, useState } from 'react';
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

export default function BackendStatusIndicator() {
  const [backendStatus, setBackendStatus] = useState<'loading' | 'connected' | 'error'>('loading');
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    async function checkBackendStatus() {
      try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.status === 'ok') {
          setBackendStatus('connected');
        } else {
          setBackendStatus('error');
          setErrorMessage(data.message || 'Could not connect to AI backend');
        }
      } catch (error) {
        setBackendStatus('error');
        setErrorMessage('Failed to connect to AI backend services');
      }
    }

    checkBackendStatus();
  }, []);

  if (backendStatus === 'loading' || backendStatus === 'connected') {
    return null;
  }

  return (
    <Alert variant="destructive" className="mb-4">
      <AlertCircle className="h-4 w-4" />
      <AlertTitle>Backend Connection Error</AlertTitle>
      <AlertDescription>
        {errorMessage}. Some features may not work properly. Please try again later or contact support.
      </AlertDescription>
    </Alert>
  );
}