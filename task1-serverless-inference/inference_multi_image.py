"""
Phi-4-multimodal-instruct: Multi-image inference example.
Sends multiple images with a prompt — useful for comparison or video frame analysis.

Usage:
    python inference_multi_image.py <image1> <image2> [image3 ...] --prompt "Your question"

Examples:
    python inference_multi_image.py before.jpg after.jpg --prompt "What changed between these two images?"
    python inference_multi_image.py frame1.jpg frame2.jpg frame3.jpg --prompt "Describe what is happening across these frames"
"""
import os
import sys
import base64
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    UserMessage,
    TextContentItem,
    ImageContentItem,
    ImageUrl,
)
from azure.core.credentials import AzureKeyCredential

load_dotenv()


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}.get(ext, "image/jpeg")


def main():
    images = []
    prompt = "Compare these images and describe what you see."

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--prompt" and i + 1 < len(args):
            prompt = args[i + 1]
            i += 2
        elif os.path.exists(args[i]):
            images.append(args[i])
            i += 1
        else:
            # Treat non-file arguments as the prompt
            prompt = args[i]
            i += 1

    if not images:
        print("Usage: python inference_multi_image.py <img1> <img2> [prompt or --prompt 'question']")
        sys.exit(1)

    client = ChatCompletionsClient(
        endpoint=os.environ["AZURE_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_API_KEY"]),
    )

    content = [TextContentItem(text=prompt)]
    for img_path in images:
        if not os.path.exists(img_path):
            print(f"Warning: skipping {img_path} (not found)")
            continue
        b64 = encode_image(img_path)
        mime = get_mime(img_path)
        content.append(ImageContentItem(image_url=ImageUrl(url=f"data:{mime};base64,{b64}")))

    response = client.complete(
        messages=[UserMessage(content=content)],
        max_tokens=1024,
        temperature=0.7,
    )

    print(response.choices[0].message.content)
    print(f"\n--- Usage: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output tokens ---")


if __name__ == "__main__":
    main()
