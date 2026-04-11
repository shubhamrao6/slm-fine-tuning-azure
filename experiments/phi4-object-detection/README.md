# Phi-4-Multimodal Object Detection

Object detection in images and videos using Microsoft Phi-4-multimodal-instruct's vision capabilities via the Azure serverless API.

## How It Works

Phi-4-multimodal is a generative vision-language model, not a traditional object detector like YOLO. The approach here is:

1. Send the image to the model with a structured prompt asking for object labels and bounding box coordinates
2. Parse the JSON response containing detected objects and their locations
3. Draw bounding boxes on the image/video using OpenCV

## Important: Capabilities & Limitations

| Aspect | Phi-4-multimodal | Traditional Detectors (YOLO, etc.) |
|--------|-----------------|-----------------------------------|
| Object identification | Excellent — understands context, can describe objects in natural language | Limited to trained classes |
| Bounding box precision | Approximate — gives rough percentage-based locations | Pixel-precise |
| Speed | ~2-5 sec per image (API call) | ~10-50ms per image (local GPU) |
| Real-time video | Not feasible (too slow per frame) | Yes, 30+ FPS |
| Custom objects | Can detect anything it can describe (zero-shot) | Needs training data per class |
| Scene understanding | Can reason about relationships between objects | No scene understanding |

**Bottom line**: Phi-4-multimodal is great for understanding *what's in an image* and giving approximate locations. It's not a replacement for YOLO/SSD for real-time precise detection, but it excels at zero-shot detection of arbitrary objects with contextual understanding.

## Setup

```bash
# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies (already done if you used the setup)
pip install -r requirements.txt

# Copy env and add your API key
cp .env.example .env
```

## Usage

### Image Object Detection
```bash
python detect_objects.py sample_dog.jpg
python detect_objects.py your_image.png --output output/result.jpg
```

### Video Object Detection
```bash
# Process video at 2 frames per second (default)
python detect_video.py video.mp4

# Process at 1 fps for fewer API calls
python detect_video.py video.mp4 --fps 1 --output output/annotated.mp4
```

## Sample Results

```
$ python detect_objects.py sample_dog.jpg
Detecting objects in: sample_dog.jpg
Detected 1 objects:
  - dog (bbox: [40.0, 34.0, 60.0, 70.0])
Saved annotated image to: output/dog_detected.jpg
```

## Real-Time Detection Discussion

For true real-time object detection (30+ FPS), you'd want a hybrid approach:

1. **YOLO/SSD for speed**: Use YOLOv8 or similar for real-time bounding boxes at 30+ FPS locally
2. **Phi-4-multimodal for understanding**: Use the VLM periodically (every N seconds) to:
   - Identify unusual or unknown objects YOLO missed
   - Understand scene context and object relationships
   - Answer natural language queries about the scene
   - Describe activities or events happening in the video

This hybrid architecture gives you the best of both worlds — speed from traditional detectors and intelligence from the VLM.

## Files

| File | Description |
|------|-------------|
| `detect_objects.py` | Image object detection with bounding box visualization |
| `detect_video.py` | Video object detection (frame-by-frame via API) |
| `sample_dog.jpg` | Sample image for testing |
| `sample_image.jpg` | Landscape sample image |
| `output/` | Generated annotated images/videos |
