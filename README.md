## 👁️‍🗨️ Assistive Vision System – Workflow Overview

This project uses **Raspberry Pi 5 + Hailo AI accelerator + camera + speaker** to help a blind or visually impaired person understand their surroundings in real time.  
Edge AI (YOLOv11) detects what’s happening in the scene, and **Gemini 2.5** explains it in natural language, which is then spoken out loud via **text-to-speech**.

---

## 🧩 System Components

- **Hardware**
  - Raspberry Pi 5 (main controller)
  - Hailo AI accelerator (edge inference for YOLOv11)
  - Camera (live video feed)
  - Speaker / audio output device (voice feedback)

- **On-Device Software**
  - Python runtime
  - YOLOv11 model for:
    - Object detection  
    - Basic scene understanding (people, objects, positions, etc.)

- **Cloud / AI Services**
  - **Gemini 2.5**  
    - Takes detected objects, context, and optionally frame snapshots  
    - Generates a natural-language explanation of the scene:
      - Surroundings (what’s around the user)
      - Emotions / social cues (if inferable)
      - Potential dangers (e.g. cars, obstacles, moving objects)
  - **Text-to-Speech (TTS) engine**
    - Converts the Gemini-generated explanation into audio
    - Audio is streamed back to the Raspberry Pi

---

## 🔄 End-to-End Workflow

1. **Video Capture**
   - The camera continuously streams video frames to the **Raspberry Pi 5**.

2. **Edge Inference with YOLOv11 + Hailo AI**
   - Frames are sent from Raspberry Pi to the **Hailo AI accelerator**.
   - **YOLOv11** runs on the Hailo edge hardware to:
     - Detect objects in each frame.
     - Optionally track objects across frames.
   - The output includes:
     - Object labels (e.g. *person, car, door, stairs*)
     - Bounding boxes / positions
     - Confidence scores

3. **Context Packaging (Python on Raspberry Pi)**
   - A Python script:
     - Filters and aggregates the YOLOv11 detection results.
     - Optionally adds metadata such as:
       - Time, movement patterns, proximity to user, etc.
     - Builds a **compact scene summary**:
       - e.g. “Two people standing to your left, a car approaching from the front, stairs ahead.”

4. **Scene Explanation with Gemini 2.5**
   - The Raspberry Pi sends:
     - The structured detection data (and optionally an image/frame)  
     - A prompt describing what we want (e.g. *“Explain the scene to a blind person. Mention surroundings, emotions, and dangers in simple language.”*)
   - **Gemini 2.5**:
     - Interprets the context.
     - Generates a **natural language explanation**, e.g.  
       *“You are standing near a road. A car is approaching from your right. Two people are talking in front of you. There are stairs slightly ahead, so walk carefully.”*

5. **Text-to-Speech Generation**
   - The generated explanation is passed to a **TTS model**.
   - The TTS engine returns an **audio waveform** (e.g. WAV/MP3 stream).

6. **Audio Playback on Device**
   - The Raspberry Pi receives the audio from TTS.
   - Audio is played through the **speaker**.
   - The blind user hears real-time updates about:
     - Surroundings
     - People and emotions (if detectable)
     - Potential dangers / obstacles

7. **Continuous Loop**
   - Steps 1–6 repeat continuously (or at a fixed interval / on demand):
     - Ensuring **near real-time** assistive feedback.

---

## 📊 Data Flow Diagram (Mermaid)

```mermaid
flowchart LR
    subgraph Hardware
        Cam[Camera] --> RPi[Raspberry Pi 5]
        RPi --> Hailo[Hailo AI Accelerator]
        RPi --> Speaker[Speaker]
    end

    subgraph Edge AI
        Hailo --> YOLO[YOLOv11 Object Detection]
        YOLO -->|Detections & context| RPi
    end

    RPi -->|Detections + prompt| Gemini[Gemini 2.5]
    Gemini -->|Scene explanation (text)| TTS[Text-to-Speech Service]
    TTS -->|Audio stream| RPi
    RPi -->|Spoken output| Speaker
```mermaid

## 🧮 Summary
**Edge (Raspberry Pi + Hailo + YOLOv11)**
Handles **real-time, low-latency** object detection and context extraction.

Cloud (Gemini 2.5 + TTS)
Transforms structured detections into a **human-friendly narrative**, then turns it into speech.

User Experience
A blind or visually impaired user receives **continuous spoken descriptions** about:

What’s happening around them

People and emotions

Dangers and obstacles
in a way that’s understandable, timely, and helpful.
