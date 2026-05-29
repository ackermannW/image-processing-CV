Write-Host "========================================"
Write-Host "Digital Image Processing Setup (Windows)"
Write-Host "========================================"

$CondaDir = "$env:USERPROFILE\miniconda3"
$EnvName = "image-processing-cv"

# -----------------------------
# CHECK / INSTALL MINICONDA
# -----------------------------
Write-Host "`n[1/4] Checking Conda..."

$condaExists = Get-Command conda -ErrorAction SilentlyContinue

if (-not $condaExists) {
    Write-Host "Conda not found. Installing Miniconda..."

    $installer = "miniconda.exe"

    Invoke-WebRequest `
        -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" `
        -OutFile $installer

    Start-Process -FilePath $installer -ArgumentList "/S /D=$CondaDir" -Wait

    $env:Path += ";$CondaDir\Scripts;$CondaDir\Library\bin;$CondaDir"
}

# -----------------------------
# INITIALIZE CONDA FOR SCRIPT
# -----------------------------
Write-Host "`n[2/4] Initializing Conda..."

& "$CondaDir\Scripts\conda.exe" "shell.powershell" "hook" | Out-String | Invoke-Expression

# -----------------------------
# CREATE ENVIRONMENT
# -----------------------------
Write-Host "`n[3/4] Creating environment..."

conda env create -f environment.yml

# -----------------------------
# TEST INSTALLATION
# -----------------------------
Write-Host "`n[4/4] Testing installation..."

conda run -n $EnvName python -c "import numpy, cv2; print('Environment OK')"

# -----------------------------
# CLEANUP
# -----------------------------
Write-Host "`nCleaning installer..."
Remove-Item -Force "miniconda.exe" -ErrorAction SilentlyContinue

Write-Host "`n========================================"
Write-Host "Setup complete!"
Write-Host "========================================"

Write-Host "`nTo activate later:"
Write-Host "conda activate $EnvName"
