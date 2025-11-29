import pygame
import time
import os

# Initialize the mixer (the audio engine)
pygame.mixer.init()

# Check if file exists
sound_file = "alert.mp3"
if not os.path.exists(sound_file):
    print("❌ Error: mp3 file not found.")
    exit()

print("🎵 Playing sound...")

# Load and play
pygame.mixer.music.load(sound_file)
pygame.mixer.music.play()

# Keep script running while sound plays
while pygame.mixer.music.get_busy():
    time.sleep(1)

print("✅ Done!")