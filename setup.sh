#!/bin/bash

echo "[INFO] Setting up SelenoMatch environment..."

# 1. Clone the LoFTR repository
if [ ! -d "LoFTR" ]; then
    echo "[INFO] Cloning LoFTR repository..."
    git clone https://github.com/zju3dv/LoFTR.git
else
    echo "[OK] LoFTR directory already exists."
fi

# 2. Create the weights directory
mkdir -p LoFTR/weights

if [ ! -f "LoFTR/weights/outdoor_ds.ckpt" ]; then
    echo "[INFO] Downloading outdoor_ds.ckpt..."
    
    wget -O LoFTR/weights/outdoor_ds.ckpt "https://github.com/gogoiprajesh-arch/SelenoMatch/releases/download/v1.0-weights/outdoor_ds.ckpt"
else
    echo "[OK] LoFTR weights already downloaded."
fi

echo "[DONE] Setup complete! You are ready to run SelenoMatch."
