import os
import json
import tempfile
from typing import List, Optional
from PIL import Image
import gradio as gr
from gtts import gTTS
import whisper
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ============================================================
# 🔑 PASTE YOUR GEMINI API KEY HERE:
# ============================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 1. Initialize Whisper Speech-to-Text Model (tiny version uses minimal cloud RAM)
print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")

# 2. Pydantic Schema for Structured JSON Output
class ProductMetadata(BaseModel):
    product_name: str = Field(description="Name or title of the identified/recommended product")
    category: str = Field(description="Category (e.g., Apparel, Footwear, Electronics, Grocery)")
    estimated_price: str = Field(description="Price extracted from invoice/receipt or estimated market price")
    sku: str = Field(description="Detected SKU from receipt or generated SKU (e.g., SKU-7721)")
    attributes: List[str] = Field(description="Key visual attributes: color, material, pattern, style")
    style_matches: List[str] = Field(description="Complementary clothing, matching accessories, or pairing tips")
    spoken_summary: str = Field(description="Concise, natural summary intended for speech synthesis")

# 3. Pipeline Functions
def transcribe_audio(audio_path: Optional[str]) -> str:
    if not audio_path:
        return ""
    try:
        result = whisper_model.transcribe(audio_path)
        return result.get("text", "").strip()
    except Exception as e:
        print(f"Whisper Error: {e}")
        return ""

def call_gemini_vision(image: Optional[Image.Image], combined_query: str) -> tuple[dict, str]:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        raise ValueError("Please set your Gemini API key in app.py.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    system_instruction = (
        "You are an expert AI E-Commerce Assistant and Shopping Stylist. "
        "Analyze the provided image (which may be a product, clothing, invoice, or store receipt) "
        "along with the user's instructions. "
        "Extract key attributes, detect prices/SKUs, suggest matching styles/accessories, "
        "and draft a natural, warm summary for voice output."
    )

    contents = []
    if image is not None:
        contents.append(image)

    prompt = f"Customer Query Context:\n{combined_query if combined_query else 'Analyze this item and provide recommendations.'}"
    contents.append(prompt)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=ProductMetadata,
            temperature=0.3,
        ),
    )

    parsed_data = json.loads(response.text)
    return parsed_data, parsed_data.get("spoken_summary", "")

def generate_speech(text: str) -> Optional[str]:
    if not text:
        return None
    try:
        tts = gTTS(text=text, lang="en", tld="com", slow=False)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        print(f"gTTS Error: {e}")
        return None

def run_pipeline(image, text_query, audio_path):
    transcribed_text = transcribe_audio(audio_path)

    combined_parts = []
    if text_query and text_query.strip():
        combined_parts.append(f"Text Input: {text_query.strip()}")
    if transcribed_text:
        combined_parts.append(f"Voice Request: {transcribed_text}")

    final_query = "\n".join(combined_parts)
    if not final_query and image is None:
        return "⚠️ Please upload an image, type a question, or record your voice.", {}, None, ""

    try:
        data, spoken_text = call_gemini_vision(image, final_query)
        audio_output_path = generate_speech(spoken_text)

        markdown_result = f"""
### 🛍️ Assistant Recommendation
{spoken_text}

---
* **Product Name:** {data.get('product_name', 'N/A')}
* **Category:** {data.get('category', 'N/A')}
* **Price:** `{data.get('estimated_price', 'N/A')}`
* **SKU:** `{data.get('sku', 'N/A')}`
* **Attributes:** {', '.join(data.get('attributes', []))}
* **Style Pairings:** {', '.join(data.get('style_matches', []))}
"""
        return (
            markdown_result,
            data,
            audio_output_path,
            transcribed_text if transcribed_text else "(No voice input recorded)"
        )
    except Exception as err:
        return f"❌ Error: {str(err)}", {}, None, transcribed_text

# 4. Gradio Interface
with gr.Blocks(title="Multimodal E-Commerce Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛒 Multimodal Smart E-Commerce & Product Recommendation Assistant")

    with gr.Row():
        with gr.Column(scale=5):
            image_input = gr.Image(label="1. Upload Product Photo or Receipt", type="pil")
            text_input = gr.Textbox(label="2. Customer Text Query", placeholder="e.g. Find matching shoes or summarize this receipt")
            audio_input = gr.Audio(label="3. Spoken Voice Request", sources=["microphone", "upload"], type="filepath")
            submit_btn = gr.Button("🚀 Process & Recommend", variant="primary")

        with gr.Column(scale=6):
            voice_transcript_output = gr.Textbox(label="Transcribed Voice (Whisper)", interactive=False)
            audio_response_output = gr.Audio(label="Spoken Recommendation (gTTS)", autoplay=True)
            markdown_output = gr.Markdown(label="Recommendation Summary")
            json_output = gr.JSON(label="Structured Metadata (Price, SKU, Category)")

    submit_btn.click(
        fn=run_pipeline,
        inputs=[image_input, text_input, audio_input],
        outputs=[markdown_output, json_output, audio_response_output, voice_transcript_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
