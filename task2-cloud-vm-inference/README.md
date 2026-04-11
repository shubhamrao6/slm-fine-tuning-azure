# Task 2: Cloud Provisioned VM Inference

Deploy all 4 models (Florence-2-large, Qwen2.5-VL-3B, Qwen2.5-VL-7B, Phi-4-multimodal) on Azure ML compute instances for controlled, consistent testing.

## Provisioned Resources

Workspace: `slm-workspace` (West US, resource group: CashAPI)

| Compute Name | VM Size | GPU | VRAM | Cost/hr | Purpose |
|---|---|---|---|---|---|
| slm-workbench | Standard_NC12s_v3 | 2x V100 | 32 GB | $6.12 | Main workbench — inference, fine-tuning, benchmarking |
| slm-edge-sim | Standard_NC4as_T4_v3 | 1x T4 | 16 GB | $0.53 | Edge simulation (matches Jetson Orin NX 16GB) |

### Important: Stop when not using (saves money)

```bash
# Stop both
az ml compute stop --name slm-workbench --resource-group CashAPI --workspace-name slm-workspace
az ml compute stop --name slm-edge-sim --resource-group CashAPI --workspace-name slm-workspace

# Start when needed
az ml compute start --name slm-workbench --resource-group CashAPI --workspace-name slm-workspace
az ml compute start --name slm-edge-sim --resource-group CashAPI --workspace-name slm-workspace
```

### Access

1. Go to https://ml.azure.com → select `slm-workspace`
2. Compute → Compute instances → click on the instance
3. Use Jupyter, Terminal, or VS Code to work on it

## Setup

SSH into the compute instance (or use the Jupyter terminal) and run:
```bash
bash setup.sh
```

This installs all dependencies and downloads all 4 models.
