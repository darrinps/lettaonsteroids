# Development Setup Script for Letta Hybrid POC (Windows)

param(
    [switch]$SkipOllama,
    [switch]$SkipPoetry,
    [switch]$SkipIndex
)

$ErrorActionPreference = "Stop"

Write-Host "=== Letta Hybrid POC - Windows Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check if running in correct directory
if (-not (Test-Path "pyproject.toml")) {
    Write-Host "Error: Please run this script from the letta_hybrid_poc directory" -ForegroundColor Red
    exit 1
}

# Step 1: Check/Install Ollama
if (-not $SkipOllama) {
    Write-Host "[1/5] Checking Ollama installation..." -ForegroundColor Yellow

    $ollamaInstalled = Get-Command ollama -ErrorAction SilentlyContinue

    if (-not $ollamaInstalled) {
        Write-Host "Ollama not found. Please install it:" -ForegroundColor Yellow
        Write-Host "  Option 1: Download from https://ollama.ai/download" -ForegroundColor White
        Write-Host "  Option 2: Run 'winget install Ollama.Ollama'" -ForegroundColor White
        Write-Host ""
        $continue = Read-Host "Have you installed Ollama? (y/n)"
        if ($continue -ne "y") {
            exit 1
        }
    } else {
        Write-Host "  ✓ Ollama found: $($ollamaInstalled.Source)" -ForegroundColor Green
    }

    # Check if Ollama is running
    Write-Host "  Checking Ollama service..." -ForegroundColor White
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 5 -ErrorAction Stop
        Write-Host "  ✓ Ollama service is running" -ForegroundColor Green
    } catch {
        Write-Host "  ! Ollama service not responding, attempting to start..." -ForegroundColor Yellow
        Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 3

        try {
            $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 5
            Write-Host "  ✓ Ollama service started" -ForegroundColor Green
        } catch {
            Write-Host "  ! Could not start Ollama service automatically" -ForegroundColor Yellow
            Write-Host "  Please start it manually: 'ollama serve'" -ForegroundColor White
        }
    }

    # Pull required model
    Write-Host "  Pulling llama3.1:8b model..." -ForegroundColor White
    $models = ollama list
    if ($models -match "llama3.1:8b") {
        Write-Host "  ✓ Model llama3.1:8b already exists" -ForegroundColor Green
    } else {
        Write-Host "  Downloading llama3.1:8b (this may take several minutes)..." -ForegroundColor Yellow
        ollama pull llama3.1:8b
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ Model downloaded successfully" -ForegroundColor Green
        } else {
            Write-Host "  ! Failed to download model" -ForegroundColor Red
            exit 1
        }
    }

    Write-Host ""
}

# Step 2: Check/Install Poetry
if (-not $SkipPoetry) {
    Write-Host "[2/5] Checking Poetry installation..." -ForegroundColor Yellow

    $poetryInstalled = Get-Command poetry -ErrorAction SilentlyContinue

    if (-not $poetryInstalled) {
        Write-Host "Poetry not found. Installing via pip..." -ForegroundColor Yellow
        python -m pip install --user poetry
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ Poetry installed" -ForegroundColor Green
            # Refresh PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + [System.Environment]::GetEnvironmentVariable("Path","Machine")
        } else {
            Write-Host "  ! Failed to install Poetry" -ForegroundColor Red
            Write-Host "  Please install manually: https://python-poetry.org/docs/#installation" -ForegroundColor White
            exit 1
        }
    } else {
        Write-Host "  ✓ Poetry found: $($poetryInstalled.Source)" -ForegroundColor Green
    }

    Write-Host ""
}

# Step 3: Install Python Dependencies
Write-Host "[3/5] Installing Python dependencies..." -ForegroundColor Yellow
Write-Host "  This may take 5-10 minutes..." -ForegroundColor White

try {
    poetry install
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Dependencies installed successfully" -ForegroundColor Green
    } else {
        Write-Host "  ! Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "  ! Error during installation: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 4: Create necessary directories
Write-Host "[4/5] Creating directories..." -ForegroundColor Yellow

$directories = @("data/index", "data/cache")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  ✓ Created $dir" -ForegroundColor Green
    } else {
        Write-Host "  ✓ $dir already exists" -ForegroundColor Green
    }
}

Write-Host ""

# Step 5: Build Index
if (-not $SkipIndex) {
    Write-Host "[5/5] Building hybrid index..." -ForegroundColor Yellow

    if (Test-Path "data/index/config.json") {
        Write-Host "  Index already exists. Rebuild? (y/n)" -ForegroundColor White
        $rebuild = Read-Host
        if ($rebuild -ne "y") {
            Write-Host "  Skipping index build" -ForegroundColor Yellow
            Write-Host ""
        } else {
            Write-Host "  Building index from corpus..." -ForegroundColor White
            poetry run python -m src.cli.ingest build --corpus-path data/raw/corpus.json
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  ✓ Index built successfully" -ForegroundColor Green
            } else {
                Write-Host "  ! Failed to build index" -ForegroundColor Red
            }
            Write-Host ""
        }
    } else {
        Write-Host "  Building index from corpus..." -ForegroundColor White
        poetry run python -m src.cli.ingest build --corpus-path data/raw/corpus.json
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ Index built successfully" -ForegroundColor Green
        } else {
            Write-Host "  ! Failed to build index" -ForegroundColor Red
            Write-Host "  You can build it later with:" -ForegroundColor White
            Write-Host "    poetry run python -m src.cli.ingest build" -ForegroundColor White
        }
        Write-Host ""
    }
}

# Summary
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Quick Start Commands:" -ForegroundColor White
Write-Host "  1. Activate environment:  poetry shell" -ForegroundColor Yellow
Write-Host "  2. Start chat:            poetry run python -m src.cli.chat run --mode augmented" -ForegroundColor Yellow
Write-Host "  3. Run evaluation:        poetry run python -m src.eval.eval run" -ForegroundColor Yellow
Write-Host ""
Write-Host "For more info, see README.md" -ForegroundColor White
Write-Host ""
