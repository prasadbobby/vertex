// src/lib/html-utils.ts
export function extractTextFromHtml(html: string): string {
  if (!html) return '';
  
  // If it doesn't look like HTML, return as is
  if (!html.includes('<') || !html.includes('>')) {
    return html;
  }
  
  // Create a DOMParser to parse the HTML string (client-side only)
  if (typeof window !== 'undefined') {
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    return doc.body.textContent || '';
  }
  
  // For server-side, do a simple regex-based extraction
  return html.replace(/<[^>]*>/g, '');
}