# Phi-4-Multimodal Serverless Inference

Inference SDK for Microsoft Phi-4-multimodal-instruct deployed as a serverless API on Azure AI Foundry.

## Endpoint Details

| Property | Value |
|----------|-------|
| Model | Phi-4-multimodal-instruct (5.6B params) |
| Endpoint | `https://phi4-mm-serverless.eastus2.models.ai.azure.com` |
| Region | East US 2 |
| Workspace | phi4-workspace (resource group: CashAPI) |
| Billing | Pay-per-token (no idle cost) |
| Input | Text, Images, Audio |
| Output | Text |

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Copy the env file and fill in your key:
```bash
cp .env.example .env
```
Edit `.env` with your API key. To retrieve the key via CLI:
```bash
az ml serverless-endpoint get-credentials \
  --name phi4-mm-serverless \
  --resource-group CashAPI \
  --workspace-name phi4-workspace
```

## Usage

### Text Inference
```bash
python inference_text.py
```
Edit the prompt inside the script or extend it for your use case.

### Image + Text Inference
```bash
python inference_image.py photo.jpg
python inference_image.py chart.png "Summarize the data in this chart"
```

### Multi-Image / Video Frame Inference
```bash
python inference_multi_image.py before.jpg after.jpg --prompt "What changed?"
python inference_multi_image.py frame1.jpg frame2.jpg frame3.jpg --prompt "Describe the sequence"
```

## How This Was Deployed (CLI Steps)

The endpoint was created entirely via Azure CLI:

```bash
# 1. Install the ML extension
az extension add --name ml -y

# 2. Create a workspace in a supported region (eastus2)
az ml workspace create \
  --name phi4-workspace \
  --resource-group CashAPI \
  --location eastus2

# 3. Create the serverless endpoint using the YAML config
az ml serverless-endpoint create \
  --file deploy-endpoint.yml \
  --resource-group CashAPI \
  --workspace-name phi4-workspace

# 4. Get the API key
az ml serverless-endpoint get-credentials \
  --name phi4-mm-serverless \
  --resource-group CashAPI \
  --workspace-name phi4-workspace
```

The YAML config (`deploy-endpoint.yml`):
```yaml
name: phi4-mm-serverless
model_id: azureml://registries/azureml/models/Phi-4-multimodal-instruct
```

## Pricing

| | Rate |
|---|---|
| Input (text + image) | $0.00008 / 1K tokens |
| Output | $0.00032 / 1K tokens |
| Input (audio) | $0.004 / 1K tokens |

## Supported Regions for Serverless Deployment

East US, East US 2, North Central US, South Central US, Sweden Central, West US, West US 3

## Manage the Endpoint

```bash
# Check endpoint status
az ml serverless-endpoint show \
  --name phi4-mm-serverless \
  --resource-group CashAPI \
  --workspace-name phi4-workspace

# List all serverless endpoints
az ml serverless-endpoint list \
  --resource-group CashAPI \
  --workspace-name phi4-workspace

# Regenerate keys
az ml serverless-endpoint regenerate-keys \
  --name phi4-mm-serverless \
  --resource-group CashAPI \
  --workspace-name phi4-workspace

# Delete endpoint (stops billing)
az ml serverless-endpoint delete \
  --name phi4-mm-serverless \
  --resource-group CashAPI \
  --workspace-name phi4-workspace
```

## Files

| File | Description |
|------|-------------|
| `deploy-endpoint.yml` | YAML config used to create the serverless endpoint |
| `inference_text.py` | Text-only inference example |
| `inference_image.py` | Single image + text inference |
| `inference_multi_image.py` | Multi-image comparison / video frame analysis |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for environment variables |
| `.env` | Your actual credentials (gitignored) |
