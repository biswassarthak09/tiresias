import cv2
import os
import time
import json
import glob
import asyncio
import argparse
import pygame
import edge_tts
import time

from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- CONFIGURATION ---
FRAME_INTERVAL = 90       # Take every 90th frame (every 3 sec @ 30fps)
MAX_FRAMES = 3            # Limit to 3 frames for ultra-fast mode
VOICE = "en-US-ChristopherNeural"

# --- SETUP ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ Error: GEMINI_API_KEY not found in .env file.")
    exit(1)

CLIENT = genai.Client(api_key=API_KEY)

def get_video_file(input_folder):
    """Finds the first MP4 file in the inputs folder."""
    files = glob.glob(os.path.join(input_folder, "*.mp4"))
    if not files:
        return None
    print(f"🎥 Found video file: {os.path.basename(files[0])}")
    return files[0] # Returns the first video found

def extract_nth_frames(video_path, output_folder):
    """
    Extracts every Nth frame, saves it to disk, and returns PIL images for Gemini.
    """
    print(f"🎬 Processing video: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   Stats: {total_frames} frames total @ {fps:.2f} FPS")
    print(f"   Strategy: Extracting every {FRAME_INTERVAL}th frame...")

    extracted_images = []
    # saved_paths = []
    
    frame_idx = 0
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # LOGIC: Only take every Nth frame
        if frame_idx % FRAME_INTERVAL == 0:
            # 1. Convert BGR (OpenCV) to RGB (PIL)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            
            # 2. Resize if huge (Optional, speeds up upload)
            pil_img.thumbnail((256, 256)) 
            
            # 3. Save to output folder for user inspection
            filename = f"frame_{count:03d}_(idx_{frame_idx}).jpg"
            save_path = os.path.join(output_folder, filename)
            pil_img.save(save_path)
            
            extracted_images.append(pil_img)
            # saved_paths.append(save_path)
            count += 1
            
            # Stop if we hit our limit
            if count >= MAX_FRAMES:
                print(f"   ⚠️ Hit limit of {MAX_FRAMES} frames.")
                break

        frame_idx += 1
    
    cap.release()
    print(f"✅ Extracted {len(extracted_images)} frames.")
    return extracted_images

def ask_gemini(pil_images, prompt):
    """Sends the list of images + text to Gemini."""
    print("⚡ Sending to Gemini 1.5 Flash...")
    start_time = time.time()
    
    response = CLIENT.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=[prompt, *pil_images] # <--- Sending list of images
    )
    
    latency = time.time() - start_time
    print(f"📝 Answer received in {latency:.2f}s")
    return response.text

async def generate_audio(text, output_file):
    """Generates MP3 using EdgeTTS."""
    print("🗣️ Generating MP3...")
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(output_file)

def main():
    # 1. Setup Folders
    input_folder = "inputs"

    # Start the timer
    start_time = time.perf_counter()
    
    # Create output folder with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_folder = os.path.join("outputs", f"run_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    # 2. Find Video
    video_path = get_video_file(input_folder)
    if not video_path:
        print(f"❌ No .mp4 file found in '{input_folder}' folder!")
        print("   -> Please create a folder named 'inputs' and put your video there.")
        return

    # 3. Extract Frames (The "Nth Frame" Logic)
    images = extract_nth_frames(video_path, output_folder)
    
    if not images:
        print("❌ No frames extracted. Check if video is valid.")
        return

    # 4. Get Text Description
    prompt = "Look at this sequence of frames from a video. Where am I and what is happening? Keep it conversational."
    text_resp = ask_gemini(images, prompt)
    print(f"\n--- AI Output ---\n{text_resp}\n-----------------")

    # 5. Save the AI text next to the last saved frame (same basename, .txt)
    txt_path = os.path.join(output_folder, "response.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_resp)
    print(f"💾 Saved AI response to: {txt_path}")
        
    # 6. Generate & Play Audio
    audio_path = os.path.join(output_folder, "speech.mp3")
    asyncio.run(generate_audio(text_resp, audio_path))
    
    # print("▶️ Playing audio...")
    # pygame.mixer.init()
    # pygame.mixer.music.load(audio_path)
    # pygame.mixer.music.play()
    # while pygame.mixer.music.get_busy():
    #     pygame.time.Clock().tick(10)

    end_time = time.perf_counter()
    print(f"⏱️ Total runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()