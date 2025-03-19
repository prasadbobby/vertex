// src/lib/logging.ts
export function logApiCall(endpoint: string, data: any, response?: any) {
    console.log(`API Call to ${endpoint}:`, {
      request: data,
      response: response ? (typeof response === 'string' ? response.substring(0, 100) + '...' : response) : 'No response yet'
    });
  }