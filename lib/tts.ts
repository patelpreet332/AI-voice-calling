import axios from "axios";




const PYTHON_TTS_API_URL = "http://localhost:8000/tts";

export async function synthesize(text: string, language: string = "en"): Promise<Buffer> {
  try {
    const response = await axios.post(
      PYTHON_TTS_API_URL,
      { 
        text,
        language
      },
      { 
        responseType: "arraybuffer" 
      }
    );
    
    return Buffer.from(response.data);
  } catch (error: any) {
    console.error(`❌ TTS Error for language ${language}:`, error.message);
    throw new Error(`Failed to synthesize TTS: ${error.message}`);
  }
}



export function checkPiperReady(): boolean {
  return true;
}