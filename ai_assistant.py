import cv2
import speech_recognition as sr
import os
import time
import google.generativeai as genai
import datetime
from gtts import gTTS
from PIL import Image
import pygame
import numpy as np 


# ================= CONFIGURATION =================
# 🔑 PASTE YOUR GEMINI KEY HERE
GEMINI_API_KEY = "" 
MIC_INDEX = None 
# =================================================

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    print(f"❌ Error configuring Gemini: {e}")
    exit()

recognizer = sr.Recognizer()
pygame.mixer.init()

def speak_text(text):
    """Convert text to audio and play it locally"""
    print(f"🤖 Gemini: {text}")
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("response.mp3")
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"❌ Audio Error: {e}")

def listen_to_mic():
    """Listen for user question"""
    with sr.Microphone(device_index=MIC_INDEX) as source:
        print("🎤 Listening... (Ask your question!)")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("⏳ Converting speech to text...")
            text = recognizer.recognize_google(audio)
            print(f"🗣️ You said: '{text}'")
            return text
        except sr.UnknownValueError:
            print("❌ Didn't catch that.")
            return None
        except Exception as e:
            print(f"❌ Mic Error: {e}")
            return None

def analyze_with_gemini(cv2_frame, user_prompt):
    print("🧠 Gemini is thinking...")
    
    try:
        h, w = cv2_frame.shape[:2]
        if w > 10000 or h == 1:
            print("🔧 Detected flat buffer. Reshaping to 640x480...")
            try:
                cv2_frame = cv2_frame.reshape((480, 640, 3))
            except:
                side = int(cv2_frame.size**0.5)
                cv2_frame = cv2_frame.reshape((side, -1, 3))

        h, w = cv2_frame.shape[:2]
        if w > 1024 or h > 1024:
            print("⚠️ Image too big, resizing...")
            cv2_frame = cv2.resize(cv2_frame, (1024, 768))
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        cv2.imwrite(filename, cv2_frame)
        print(f"💾 Saved image to: {filename}")

        if len(cv2_frame.shape) == 2:
            color_converted = cv2.cvtColor(cv2_frame, cv2.COLOR_GRAY2RGB)
        else:
            color_converted = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(color_converted)

        response = model.generate_content([user_prompt, pil_image])
        return response.text

    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return "I couldn't process the image data."

def main():
    # Force the camera to use the correct backend for Pi
    cap = cv2.VideoCapture(0)
    
    # Try to set resolution (Note: libcamerify sometimes ignores this)
    cap.set(3, 640)
    cap.set(4, 480)

    print("=======================================")
    print("  ✨ GEMINI VISION (SAFE MODE) ✨")
    print("  Press [ENTER] in this terminal to speak.")
    print("  Press [Ctrl+C] to quit.")
    print("=======================================")

    try:
        while True:
            input("👉 Press Enter to start listening...")

            # Clear buffer to get fresh image
            for _ in range(5):
                cap.read()
            
            ret, frame = cap.read()
            if not ret or frame is None:
                print("❌ Camera failed to grab frame! (Is it plugged in?)")
                continue

            # Check if we captured a "Ghost" frame (0 size)
            if frame.shape[0] == 0 or frame.shape[1] == 0:
                print("❌ Empty frame detected. Retrying...")
                continue

            question = listen_to_mic()
            
            if question:
                answer = analyze_with_gemini(frame, question)
                speak_text(answer)
            
            print("---------------------------------------")

    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    finally:
        cap.release()

if __name__ == "__main__":
    main()