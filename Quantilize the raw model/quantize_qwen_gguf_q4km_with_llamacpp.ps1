param(
    [string]$SourceModelDir = "E:\Learning_journal_at_CUHK\CUHK_MSc_Project-32\Important_number_extraction\model\SFT model\raw_model\Qwen2.5-7B-Instruct-merged",
    [string]$OutputGGUFPath = "E:\Learning_journal_at_CUHK\CUHK_MSc_Project-32\Important_number_extraction\model\SFT model\gguf model\Qwen2.5-7B-Instruct-merged-Q4_K_M.gguf",
    [string]$OllamaExe = "E:\ollama\ollama.exe",
    [string]$OllamaModelsDir = "E:\ollama",
    [string]$OllamaModelName = "Law-Qwen-4bit",
    [string]$WorkRoot = "D:\llama_tools",
    [switch]$CleanupTemp
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Path([string]$PathToCheck, [string]$Label) {
    if (-not (Test-Path -LiteralPath $PathToCheck)) {
        throw "$Label not found: $PathToCheck"
    }
}

Ensure-Path -PathToCheck $SourceModelDir -Label "Source model dir"
Ensure-Path -PathToCheck $OllamaExe -Label "Ollama executable"

New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
$binDir = Join-Path $WorkRoot "llama-b8508-bin-win-cpu-x64"
$srcRoot = Join-Path $WorkRoot "src"
$srcDir = Join-Path $srcRoot "llama.cpp-b8508"
$workDir = Join-Path $WorkRoot "work"
New-Item -ItemType Directory -Force -Path $workDir | Out-Null

$quantExe = Join-Path $binDir "llama-quantize.exe"
$convertScript = Join-Path $srcDir "convert_hf_to_gguf.py"

if (-not (Test-Path -LiteralPath $quantExe)) {
    Write-Host "[Info] Downloading llama.cpp win-cpu binaries..."
    $zip = Join-Path $WorkRoot "llama-b8508-bin-win-cpu-x64.zip"
    Invoke-WebRequest -Uri "https://github.com/ggml-org/llama.cpp/releases/download/b8508/llama-b8508-bin-win-cpu-x64.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $binDir -Force
}

if (-not (Test-Path -LiteralPath $convertScript)) {
    Write-Host "[Info] Downloading llama.cpp source for conversion script..."
    New-Item -ItemType Directory -Force -Path $srcRoot | Out-Null
    $srcZip = Join-Path $srcRoot "llama.cpp-b8508.zip"
    Invoke-WebRequest -Uri "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b8508.zip" -OutFile $srcZip
    Expand-Archive -Path $srcZip -DestinationPath $srcRoot -Force
}

Write-Host "[Info] Ensuring Python dependency: sentencepiece"
python -m pip install --upgrade sentencepiece | Out-Null

$f16Gguf = Join-Path $workDir "Qwen2.5-7B-Instruct-merged-F16.gguf"
if (Test-Path -LiteralPath $f16Gguf) {
    Remove-Item -LiteralPath $f16Gguf -Force
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputGGUFPath) | Out-Null
if (Test-Path -LiteralPath $OutputGGUFPath) {
    Remove-Item -LiteralPath $OutputGGUFPath -Force
}

Write-Host "[Info] Converting HF -> GGUF (F16)..."
& python $convertScript $SourceModelDir --outfile $f16Gguf --outtype f16
if ($LASTEXITCODE -ne 0) {
    throw "convert_hf_to_gguf failed with exit code $LASTEXITCODE"
}

Write-Host "[Info] Quantizing GGUF to Q4_K_M..."
& $quantExe $f16Gguf $OutputGGUFPath Q4_K_M
if ($LASTEXITCODE -ne 0) {
    throw "llama-quantize failed with exit code $LASTEXITCODE"
}

# Validate GGUF magic from first 4 bytes without loading the whole file.
$fs = [System.IO.File]::OpenRead($OutputGGUFPath)
try {
    $buf = New-Object byte[] 4
    $null = $fs.Read($buf, 0, 4)
    $magic = [System.Text.Encoding]::ASCII.GetString($buf)
} finally {
    $fs.Dispose()
}
if ($magic -ne "GGUF") {
    throw "Output is not a valid GGUF file: $OutputGGUFPath"
}

$env:OLLAMA_MODELS = $OllamaModelsDir
$modelfile = Join-Path (Split-Path -Parent $OutputGGUFPath) "Modelfile.from-gguf"
@"
FROM "$($OutputGGUFPath -replace '\\', '/')"
PARAMETER temperature 0
PARAMETER top_p 1
"@ | Set-Content -LiteralPath $modelfile -Encoding UTF8

Write-Host "[Info] Importing GGUF into Ollama model: $OllamaModelName"
& $OllamaExe create $OllamaModelName -f $modelfile
if ($LASTEXITCODE -ne 0) {
    throw "ollama create failed with exit code $LASTEXITCODE"
}

$sizeBytes = (Get-Item -LiteralPath $OutputGGUFPath).Length
$sizeGiB = [math]::Round($sizeBytes / 1GB, 3)
Write-Host "[Done] GGUF: $OutputGGUFPath"
Write-Host "[Done] Size: $sizeBytes bytes (~$sizeGiB GiB)"
Write-Host "[Done] Ollama model: $OllamaModelName"

if ($CleanupTemp -and (Test-Path -LiteralPath $f16Gguf)) {
    Remove-Item -LiteralPath $f16Gguf -Force
    Write-Host "[Info] Removed temp file: $f16Gguf"
}

