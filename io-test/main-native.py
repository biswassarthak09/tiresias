import time
import cv2
from picamera2 import Picamera2

# Configuration
FILENAME = "captured_feed.mp4"
DURATION_SEC = 10
# Picamera2 setup
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "XRGB8888"})
picam2.configure(config)
picam2.start()

print(f"🎥 Recording for {DURATION_SEC} seconds...")

# Setup OpenCV Video Writer
# Note: Picamera2 effectively gives us 30fps usually
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(FILENAME, fourcc, 30.0, (640, 480))

start_time = time.time()

try:
    while (time.time() - start_time) < DURATION_SEC:
        # 1. Grab the array (image) directly from the camera hardware
        frame = picam2.capture_array()

        # 2. Convert from RGB (Camera) to BGR (OpenCV standard)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 3. Write to file
        out.write(frame)

except KeyboardInterrupt:
    pass

# Cleanup
picam2.stop()
out.release()
print(f"✅ Video saved as {FILENAME}")