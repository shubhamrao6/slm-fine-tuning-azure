"""
Phi-4-multimodal-instruct: Text-only inference example.
Sends a text prompt and prints the model's response.
"""
import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential

load_dotenv()

client = ChatCompletionsClient(
    endpoint=os.environ["AZURE_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["AZURE_API_KEY"]),
)

response = client.complete(
    messages=[UserMessage(content="Explain what a small language model is in 3 sentences.")],
    max_tokens=256,
    temperature=0.7,
)

print(response.choices[0].message.content)
print(f"\n--- Usage: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output tokens ---")
