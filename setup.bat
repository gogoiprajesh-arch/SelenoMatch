@echo off
echo [INFO] Setting up SelenoMatch environment...

:: 1. Clone the LoFTR repository
if not exist "LoFTR\" (
    echo [INFO] Cloning LoFTR repository...
    git clone https://github.com/zju3dv/LoFTR.git
) else (
    echo [OK] LoFTR directory already exists.
)

:: 2. Create the weights directory
if not exist "LoFTR\weights" mkdir "LoFTR\weights"

:: 3. Download the weights
if not exist "LoFTR\weights\outdoor_ds.ckpt" (
    echo [INFO] Downloading outdoor_ds.ckpt...
    curl -L -o "LoFTR\weights\outdoor_ds.ckpt" "https://github.com/gogoiprajesh-arch/SelenoMatch/releases/download/v1.0-weights/outdoor_ds.ckpt"
) else (
    echo [OK] LoFTR weights already downloaded.
)

echo [DONE] Setup complete! You are ready to run SelenoMatch.
