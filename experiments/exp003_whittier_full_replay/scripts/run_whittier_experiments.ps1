[CmdletBinding()]
param(
    [ValidateSet('Check', 'Cold10Legacy', 'Cold10Cached', 'FullBaseline', 'LowThreshold', 'HighResolution', 'Combined', 'All')]
    [string]$Stage = 'Check',
    [string]$Python = 'D:\AIC\pyronear_lab\.venv\Scripts\python.exe',
    [string]$Model = 'D:\AIC\pyronear_lab\models\v0.6.0\model.zip',
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')]
    [string]$RunId = (Get-Date -Format 'yyyyMMdd_HHmmss')
)

$ErrorActionPreference = 'Stop'
$experimentDirectory = Split-Path -Parent $PSScriptRoot
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$sequenceDirectory = Join-Path $projectRoot 'data\figlib_whittier_full_81'
$oldSequenceDirectory = Join-Path $projectRoot 'data\figlib_whittier_smoke_10'
$replayScript = Join-Path $PSScriptRoot 'replay_sequence.py'
$downloadScript = Join-Path $PSScriptRoot 'download_figlib_whittier.py'
$runDirectory = Join-Path $experimentDirectory "outputs\exp003_$RunId"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "Model not found: $Model" }

# Process-local cache/temp locations keep runtime files with the D: project.
$cacheDirectory = Join-Path $projectRoot '.cache\whittier_replay'
foreach ($cacheName in @('temp', 'ultralytics', 'torch', 'huggingface', 'matplotlib')) {
    $cacheTarget = Join-Path $cacheDirectory $cacheName
    New-Item -ItemType Directory -Path $cacheTarget -Force | Out-Null
}
$env:TEMP = Join-Path $cacheDirectory 'temp'
$env:TMP = $env:TEMP
$env:YOLO_CONFIG_DIR = Join-Path $cacheDirectory 'ultralytics'
$env:TORCH_HOME = Join-Path $cacheDirectory 'torch'
$env:HF_HOME = Join-Path $cacheDirectory 'huggingface'
$env:MPLCONFIGDIR = Join-Path $cacheDirectory 'matplotlib'
$env:XDG_CACHE_HOME = $cacheDirectory
$env:PYTHONDONTWRITEBYTECODE = '1'

function Invoke-Preflight {
    & $Python -B $downloadScript --output-dir $sequenceDirectory --compare-ten $oldSequenceDirectory --verify-only
    if ($LASTEXITCODE -ne 0) { throw 'Dataset verification failed.' }
    & $Python -B $replayScript --sequence $sequenceDirectory --model $Model --window 20 --dry-run
    if ($LASTEXITCODE -ne 0) { throw 'Replay input validation failed.' }
    $checkCode = @'
import sys
import torch
import temporal_model.core.model as core_model
print('Python:', sys.executable)
print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('Temporal model module:', core_model.__file__)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
if sys.argv[1] == 'cuda' and not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable. Repair the environment or explicitly use -Device cpu.')
'@
    & $Python -B -c $checkCode $Device
    if ($LASTEXITCODE -ne 0) { throw 'Model environment check failed.' }
    Write-Host 'Preflight passed. No model weights were loaded and no inference was run.'
}

function Invoke-ReplayStage {
    param([string]$Name, [double]$Confidence, [int]$ImageSize,
          [string]$DetectionMode = 'per-frame-cache', [switch]$ColdTen)
    $destination = Join-Path $runDirectory $Name
    if (Test-Path -LiteralPath $destination) {
        throw "Output already exists: $destination. Choose a new -RunId to preserve results."
    }
    $replayArguments = @('-B', $replayScript, '--sequence', $sequenceDirectory,
        '--model', $Model, '--output-dir', $destination, '--device', $Device,
        '--window', '20', '--confidence', $Confidence.ToString([Globalization.CultureInfo]::InvariantCulture),
        '--image-size', [string]$ImageSize, '--detection-mode', $DetectionMode, '--warmup', '1')
    if ($ColdTen) { $replayArguments += @('--offset-min', '0', '--offset-max', '540') }
    Write-Host "Running $Name -> $destination"
    & $Python @replayArguments
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Name. Inspect run.json and partial CSV/JSONL; results are preserved." }
}

if ($Stage -eq 'Check') {
    Invoke-Preflight
    return
}
if ($Stage -eq 'All') { Invoke-Preflight }
switch ($Stage) {
    'Cold10Legacy' { Invoke-ReplayStage -Name 'cold10_legacy' -Confidence 0.1 -ImageSize 1024 -DetectionMode 'window-batch' -ColdTen }
    'Cold10Cached' { Invoke-ReplayStage -Name 'cold10_cached' -Confidence 0.1 -ImageSize 1024 -ColdTen }
    'FullBaseline' { Invoke-ReplayStage -Name 'full_baseline' -Confidence 0.1 -ImageSize 1024 }
    'LowThreshold' { Invoke-ReplayStage -Name 'low_threshold' -Confidence 0.05 -ImageSize 1024 }
    'HighResolution' { Invoke-ReplayStage -Name 'high_resolution' -Confidence 0.1 -ImageSize 1536 }
    'Combined' { Invoke-ReplayStage -Name 'combined' -Confidence 0.05 -ImageSize 1536 }
    'All' {
        Invoke-ReplayStage -Name 'cold10_legacy' -Confidence 0.1 -ImageSize 1024 -DetectionMode 'window-batch' -ColdTen
        Invoke-ReplayStage -Name 'cold10_cached' -Confidence 0.1 -ImageSize 1024 -ColdTen
        Invoke-ReplayStage -Name 'full_baseline' -Confidence 0.1 -ImageSize 1024
        Invoke-ReplayStage -Name 'low_threshold' -Confidence 0.05 -ImageSize 1024
        Invoke-ReplayStage -Name 'high_resolution' -Confidence 0.1 -ImageSize 1536
        Invoke-ReplayStage -Name 'combined' -Confidence 0.05 -ImageSize 1536
    }
}
