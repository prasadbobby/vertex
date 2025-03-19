// src/lib/format-markdown.ts
export function formatMarkdownResponse(text: string): string {
  if (!text) return '';
  
  // Clean up any HTML tags that might be in the response
  text = text.replace(/<[^>]*>/g, '');
  
  // Ensure proper spacing between sections with emoji headers
  const improvedText = text
    // Make sure emoji headers are properly spaced
    .replace(/##\s*([🏥|💊|⚠️|📊|📋|👨‍⚕️|🔬|📚|🔍|🚨|👁️|🔄|🔮])/g, '## $1')
    // Ensure proper line breaks before headers
    .replace(/([^\n])(\n##)/g, '$1\n\n$2')
    // Fix any irregular bullet points
    .replace(/\n[-*]\s/g, '\n* ');
    
  return improvedText;
}