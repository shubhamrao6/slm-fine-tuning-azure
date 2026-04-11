#!/bin/bash
# Task 2: Provision scripts — ALREADY DONE
# These compute instances are already created in slm-workspace (West US)

RESOURCE_GROUP="CashAPI"
WORKSPACE="slm-workspace"

# === Already provisioned ===
# slm-workbench: Standard_NC12s_v3 (2x V100, 32GB VRAM, $6.12/hr)
# slm-edge-sim:  Standard_NC4as_T4_v3 (1x T4, 16GB VRAM, $0.53/hr)

# === Stop both (save money) ===
echo "Stopping compute instances..."
az ml compute stop --name slm-workbench --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE --no-wait
az ml compute stop --name slm-edge-sim --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE --no-wait
echo "Stop initiated."

# === Start both ===
# az ml compute start --name slm-workbench --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE --no-wait
# az ml compute start --name slm-edge-sim --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE --no-wait
