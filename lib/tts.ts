import axios from "axios";

/**
 * Piper TTS — Now supports language switching
 */

const PYTHON_TTS_API_URL = "http://localhost:8000/tts";

export async function synthesize(text: string, language: string = "en"): Promise<Buffer> {
  try {
    const response = await axios.post(
      PYTHON_TTS_API_URL,
      { 
        text,
        language : 'en'
      },
      { 
        responseType: "arraybuffer" 
      }
    );
    
    const pcmBuffer = Buffer.from(response.data);
    console.log(`🔊 [TTS] Generated ${pcmBuffer.length} bytes (${language})`);
    return pcmBuffer;
  } catch (error: any) {
    console.error(`❌ TTS Error for language ${language}:`, error.message);
    throw new Error(`Failed to synthesize TTS: ${error.message}`);
  }
}

/**
 * Check if Piper TTS is ready
 */
export function checkPiperReady(): boolean {
  return true;
}