import cv2
import os
import time
import asyncio
import edge_tts
import subprocess
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- CONFIGURATION ---
RECORD_SECONDS = 10       # Record 10 seconds
FRAME_INTERVAL = 20       # Take every 20th frame
VOICE = "en-US-ChristopherNeural"
MODEL_NAME = "gemini-2.0-flash" 

# --- SETUP ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
CLIENT = genai.Client(api_key=API_KEY)

# --- 1. VIDEO CAPTURE ---
def capture_live_video(output_path, duration=5):
    """Records video using Pi Hardware"""
    print(f"🎥 Recording context ({duration}s)...")
    command = [
        "rpicam-vid", "-t", str(duration * 1000), "-o", output_path,
        "--width", "640", "--height", "480", "--framerate", "30", "--nopreview"
    ]
    try:
        subprocess.run(command, check=True)
        return True
    except:
        print("❌ Camera Error.")
        return False

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    images = []
    frame_idx = 0
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx % FRAME_INTERVAL == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            pil_img.thumbnail((320, 320)) # Resize for speed
            images.append(pil_img)
            count += 1
            if count >= 15: break
        frame_idx += 1
    
    cap.release()
    return images

def get_ai_response(pil_images, mode):
    print(f"⚡ Analyzing in mode: {mode}...")

    if mode == "1":
        prompt = (
            "Describe the scene immediately without any greetings or introductory filler. "
            "First, briefly mention the setting and lighting. Also capture the emotion of the environment. "
            "Next, describe the person closest to the camera: their outfit and specifically what they are holding or showing. "
            "Finally, briefly describe the other people in the background. Explain if there is an danger to the person or not."
            "Do not say 'Welcome' and do not ask questions. Just provide the visual description in a natural, conversational flow. "
        )

    elif mode == "2":
        prompt = (
            "You are a Blind Sonar assistant. The user is holding an object. "
            "Identify the specific object in the center of the frame or closest to the camera."
            "Describe its color, text (if any), and what it is. Be extremely brief (1 sentence)."
        )

    elif mode == "3":
        prompt = (
            "You are a Blind Sonar assistant. The user asked: 'Who is in front of me?'. "
            "Describe the person closest to the camera. Mention their gender, clothing color, "
            "and what they are doing. Also capture the emotion of the person. Ignore the background crowd."
        )

    try:
        response = CLIENT.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, *pil_images]
        )
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return "I had trouble seeing that."

# --- 3. OUTPUT ---
async def speak(text):
    print(f"\n🤖 AI: {text}\n")
    # Generate MP3
    communicate = edge_tts.Communicate(text, VOICE, rate="+10%", volume="+50%")
    await communicate.save("response.mp3")
    # Play MP3
    subprocess.run(["mpg123", "-q", "-f", "60000", "response.mp3"])

def main():
    os.makedirs("data_cache", exist_ok=True)
    video_file = "data_cache/context.mp4"

    print("\n=== 🦇 BLIND SONAR (DEMO MODE) ===")
    print("-----------------------------------")
    print(" [1] SCENE  -> 'Where am I?'")
    print(" [2] OBJECT -> 'What is this?'")
    print(" [3] PERSON -> 'Who is this?'")
    print(" [Q] QUIT")
    print("-----------------------------------")

    while True:
        key = input("\n👉 Select Mode (1/2/3): ").strip().lower()

        if key == 'q':
            print("👋 Exiting...")
            break
        
        if key not in ['1', '2', '3']:
            print("❌ Invalid key. Use 1, 2, or 3.")
            continue

        capture_live_video(video_file, duration=RECORD_SECONDS)
        images = extract_frames(video_file, "data_cache")
        if not images: continue

        answer = get_ai_response(images, key)
        asyncio.run(speak(answer))

if __name__ == "__main__":
    main()