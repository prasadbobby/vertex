// src/lib/fallback-responses.ts
export const getFallbackResponse = (query: string) => {
    // Check for common medical queries
    if (/headache|migraine|head pain/i.test(query)) {
      return `## 🏥 Headache Information
  
  * Headaches can have various causes, from tension to more serious conditions.
  * Common types include tension headaches, migraines, and cluster headaches.
  
  ## ⚠️ When to Seek Help
  
  * Seek immediate medical attention if your headache is sudden and severe
  * Also seek help if your headache is accompanied by fever, stiff neck, confusion, seizures, double vision, weakness, numbness, or difficulty speaking
  
  ## 💊 General Advice
  
  * Rest in a quiet, dark room
  * Apply a cold pack to your forehead
  * Consider over-the-counter pain medications (following package instructions)
  * Stay hydrated
  
  *Note: This is a fallback response due to temporary connectivity issues with our main AI system. For personalized medical advice, please consult a healthcare professional.*`;
    }
    
    if (/cold|flu|fever|cough/i.test(query)) {
      return `## 🏥 Cold & Flu Information
  
  * Common symptoms include fever, cough, sore throat, body aches, and fatigue
  * Most cases resolve on their own within 7-10 days
  
  ## ⚠️ When to Seek Help
  
  * High fever (above 103°F/39.4°C)
  * Symptoms lasting more than 10 days
  * Difficulty breathing
  * Persistent chest pain
  * Severe or persistent vomiting
  
  ## 💊 General Advice
  
  * Rest and stay hydrated
  * Consider over-the-counter fever reducers
  * Use a humidifier to ease congestion
  * Gargle with salt water for sore throat
  
  *Note: This is a fallback response due to temporary connectivity issues with our main AI system. For personalized medical advice, please consult a healthcare professional.*`;
    }
    
    // Default fallback response
    return `I apologize, but I'm currently experiencing connectivity issues with our advanced analysis systems. Here's some general information that might help:
  
  ## 🏥 General Health Advice
  
  * For any severe or persistent symptoms, consult with a healthcare professional
  * Stay hydrated and get adequate rest when dealing with most illnesses
  * Over-the-counter medications can help manage many common symptoms, but follow package instructions
  * Prevention through hand washing, proper nutrition, and regular exercise helps maintain overall health
  
  I'd be happy to try answering your question again in a few minutes when our systems might be back online.
  
  *Note: This is a fallback response due to temporary connectivity issues. For personalized medical advice, please consult a healthcare professional.*`;
  };