"""
Video Object Detection using Phi-4-multimodal-instruct.

Extracts frames from a video, resizes each to 500x500, runs object detection
via the Phi-4-multimodal API, and produces an annotated output video.

Usage:
    python detect_video.py <video_path> [--fps 2] [--output output.mp4]
"""
import os
import sys
import json
import re
import base64
import cv2
import numpy as np
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    UserMessage, TextContentItem, ImageContentItem, ImageUrl,
)
from azure.core.credentials import AzureKeyCredential

load_dotenv()

IMG_SIZE = 500

SYSTEM_PROMPT = "You are a helpful assistant that can analyze images."

DETECTION_PROMPT = f"""This {IMG_SIZE}x{IMG_SIZE} pixel image is a video frame. What objects are in it and where?

For each object: {{"label":"name", "confidence":"high/medium/low", "bbox":[x_min, y_min, x_max, y_max]}} in pixel coords (0 to {IMG_SIZE}).

Example: [{{"label":"person","confidence":"high","bbox":[100,50,200,400]}}]"""

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]


def frame_to_base64(frame) -> str:
    """Resize frame to IMG_SIZExIMG_SIZE and encode as base64 JPEG."""
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def detect_frame(client, frame) -> list[dict]:
    b64 = frame_to_base64(frame)
    try:
        response = client.complete(
            messages=[
                UserMessage(content=[
                    TextContentItem(text=DETECTION_PROMPT),
                    ImageContentItem(image_url=ImageUrl(url=f"data:image/jpeg;base64,{b64}")),
                ]),
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        # Try full JSON parse first (handles {"objects": [...]})
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break
            if isinstance(parsed, list):
                dets = [d for d in parsed if isinstance(d, dict) and "label" in d]
                return normalize_bboxes(dets)
        except json.JSONDecodeError:
            pass

        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            detections = json.loads(match.group())
            dets = [d for d in detections if isinstance(d, dict) and "label" in d]
            return normalize_bboxes(dets)
    except Exception as e:
        print(f"  Warning: API error on frame: {e}")
    return []


def normalize_bboxes(dets):
    for d in dets:
        bbox = d.get("bbox", [0, 0, 0, 0])
        mx = max(bbox) if bbox else 0
        if mx <= 1.0:
            d["bbox"] = [v * IMG_SIZE for v in bbox]
        elif mx <= 100:
            d["bbox"] = [v / 100 * IMG_SIZE for v in bbox]
    return dets


def draw_on_frame(frame, detections):
    """Draw detections on a 500x500 frame using pixel coordinates."""
    for i, det in enumerate(detections):
        label = det.get("label", "?")
        bbox = det.get("bbox", [0, 0, 0, 0])
        color = COLORS[i % len(COLORS)]

        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(IMG_SIZE, int(bbox[2]))
        y2 = min(IMG_SIZE, int(bbox[3]))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_video.py <video_path> [--fps 2] [--output output.mp4]")
        sys.exit(1)

    video_path = sys.argv[1]
    sample_fps = 2
    output_path = "output/detected_video.mp4"

    args = sys.argv[2:]
    for i, arg in enumerate(args):
        if arg == "--fps" and i + 1 < len(args):
            sample_fps = float(args[i + 1])
        elif arg == "--output" and i + 1 < len(args):
            output_path = args[i + 1]

    if not os.path.exists(video_path):
        print(f"Error: File not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / orig_fps if orig_fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"  Original: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
          f"FPS: {orig_fps:.1f}, Duration: {duration:.1f}s")
    print(f"  Output: {IMG_SIZE}x{IMG_SIZE}, sampling at {sample_fps} fps → ~{int(duration * sample_fps)} frames")

    frame_interval = int(orig_fps / sample_fps) if sample_fps < orig_fps else 1

    client = ChatCompletionsClient(
        endpoint=os.environ["AZURE_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_API_KEY"]),
    )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, sample_fps, (IMG_SIZE, IMG_SIZE))

    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            processed += 1
            resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            print(f"  Frame {processed} (#{frame_idx})...", end=" ", flush=True)
            detections = detect_frame(client, frame)
            print(f"{len(detections)} objects")
            annotated = draw_on_frame(resized, detections)
            out.write(annotated)

        frame_idx += 1

    cap.release()
    out.release()
    print(f"\nDone. {processed} frames → {output_path}")


if __name__ == "__main__":
    main()
