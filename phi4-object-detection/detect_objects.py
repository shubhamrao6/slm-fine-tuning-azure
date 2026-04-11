"""
Object Detection using Phi-4-multimodal-instruct.

Images are resized to 500x500 before sending to the model.

Usage:
    python detect_objects.py <image_path> [--output output.jpg]
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
from azure.ai.inference.models import UserMessage, TextContentItem, ImageContentItem, ImageUrl
from azure.core.credentials import AzureKeyCredential

load_dotenv()

IMG_SIZE = 500

DETECTION_PROMPT = f"""This image is {IMG_SIZE}x{IMG_SIZE} pixels. Detect at most 4 objects. For each object return label and bounding box as pixel coordinates.

Return ONLY a JSON array like this (values are for reference only, use actual values from the image but keep the structure the same):
[{{"label":"dog","bbox":[150,100,350,400]}},{{"label":"ball","bbox":[390,340,450,400]}}]"""


def resize_image(image_path: str) -> bytes:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def parse_detections(raw: str) -> list[dict]:
    """Extract detections from model response, handling malformed JSON."""
    # 1. Try clean JSON
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
        if isinstance(parsed, list):
            dets = [d for d in parsed if isinstance(d, dict) and "label" in d and "bbox" in d]
            if dets:
                return normalize(dets)
    except json.JSONDecodeError:
        pass

    # 2. Try extracting [...] substring
    m = re.search(r'\[.*\]', raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            dets = [d for d in parsed if isinstance(d, dict) and "label" in d and "bbox" in d]
            if dets:
                return normalize(dets)
        except json.JSONDecodeError:
            pass

    # 3. Regex fallback: pull label+bbox pairs from broken JSON
    pairs = re.findall(r'"label"\s*:\s*"([^"]+)".*?"bbox"\s*:\s*\[([\d\s,.]+)\]', raw, re.DOTALL)
    if pairs:
        seen, dets = set(), []
        for label, bbox_str in pairs:
            if label in seen:
                continue
            nums = [float(x.strip()) for x in bbox_str.split(",") if x.strip()]
            if len(nums) == 4:
                seen.add(label)
                dets.append({"label": label, "bbox": nums})
        if dets:
            return normalize(dets)

    return []


def normalize(dets: list[dict]) -> list[dict]:
    """Auto-convert bbox coords to pixels on IMG_SIZE."""
    for d in dets:
        bbox = d["bbox"]
        mx = max(bbox) if bbox else 0
        if mx <= 1.0:
            d["bbox"] = [v * IMG_SIZE for v in bbox]
        elif mx <= 100:
            d["bbox"] = [v / 100 * IMG_SIZE for v in bbox]
        # Clamp
        d["bbox"] = [max(0, min(IMG_SIZE, v)) for v in d["bbox"]]
    return dets


def detect(image_path: str, max_retries: int = 3) -> list[dict]:
    client = ChatCompletionsClient(
        endpoint=os.environ["AZURE_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_API_KEY"]),
    )
    b64 = base64.b64encode(resize_image(image_path)).decode("utf-8")

    for attempt in range(1, max_retries + 1):
        response = client.complete(
            messages=[UserMessage(content=[
                TextContentItem(text=DETECTION_PROMPT),
                ImageContentItem(image_url=ImageUrl(url=f"data:image/jpeg;base64,{b64}")),
            ])],
            max_tokens=512,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        print(f"[Attempt {attempt}] {raw[:150]}")
        dets = parse_detections(raw)
        if dets:
            return dets[:4]
        if attempt < max_retries:
            print("  Retrying...")

    print("No valid detections after retries.")
    return []


COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]


def draw_detections(image_path: str, detections: list[dict], output_path: str):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(det["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, det["label"], (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"\nSaved to: {output_path}")
    for det in detections:
        print(f"  - {det['label']} bbox: {[int(v) for v in det['bbox']]}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_objects.py <image_path> [--output output.jpg]")
        sys.exit(1)
    image_path = sys.argv[1]
    output_path = "output/detected.jpg"
    for i, arg in enumerate(sys.argv[2:]):
        if arg == "--output" and i + 1 < len(sys.argv) - 2:
            output_path = sys.argv[i + 3]
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found"); sys.exit(1)
    print(f"Detecting in: {image_path} ({IMG_SIZE}x{IMG_SIZE})\n")
    dets = detect(image_path)
    if dets:
        draw_detections(image_path, dets, output_path)
    else:
        print("No objects detected.")

if __name__ == "__main__":
    main()
