"""
Phi-4-multimodal-instruct: Image + Text inference example.
Sends an image along with a text prompt for visual understanding.

Usage:
    python inference_image.py <path-to-image> [optional prompt]

Examples:
    python inference_image.py photo.jpg
    python inference_image.py chart.png "Summarize the data in this chart"
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


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "image/jpeg")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inference_image.py <image_path> [prompt]")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe what you see in this image in detail."

    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    client = ChatCompletionsClient(
        endpoint=os.environ["AZURE_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_API_KEY"]),
    )

    b64 = encode_image(image_path)
    mime = get_mime_type(image_path)

    response = client.complete(
        messages=[
            UserMessage(
                content=[
                    TextContentItem(text=prompt),
                    ImageContentItem(
                        image_url=ImageUrl(url=f"data:{mime};base64,{b64}")
                    ),
                ]
            )
        ],
        max_tokens=1024,
        temperature=0.7,
    )

    print(response.choices[0].message.content)
    print(f"\n--- Usage: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output tokens ---")


if __name__ == "__main__":
    main()
