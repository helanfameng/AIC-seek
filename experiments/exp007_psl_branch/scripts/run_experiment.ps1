[CmdletBinding()]
param(
    [ValidateSet('Check','Tests','Prepare','Extract','Setup','Review','Prelabel','ImportVia','Status','Smoke','Train','Calibrate','Evaluate','Replay','Aggregate','Run')]
    [string]$Stage = 'Check',
    [string]$Python = '',
    [string]$Packages = 'D:\AIC\pyronear_lab\.venv\Lib\site-packages',
    [string]$Model = 'D:\AIC\pyronear_lab\models\v0.6.0\model.zip',
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,51}$')][string]$RunId = 'exp007_main_20261023',
    [Alias('Seed')][int[]]$Seeds = @(42,43,44),
    [ValidateSet('C0','P1','P2','P3','P4','P5','C1')][string[]]$Variants,
    [ValidateSet('develop','train','val','test','all')][string]$ReviewSplit = 'test',
    [ValidateRange(1024,65535)][int]$Port = 8767,
    [string]$Via
)
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $runtimeCandidates = @(
        'D:\AIC\pyronear_lab\.venv\Scripts\python.exe',
        (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
    )
    foreach ($candidate in $runtimeCandidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            try {
                $probe = & $candidate -B -c 'import sys; print(sys.version_info.major)' 2>&1
                if ($LASTEXITCODE -eq 0) { $Python = $candidate; break }
            } catch { }
        }
    }
}
if (-not $Python -or -not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'No working Python launcher found. Pass -Python and -Packages with the existing compatible environment.'
}
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$action = if ($Stage -eq 'ImportVia') { 'import-via' } else { $Stage.ToLowerInvariant() }
$arguments = @('-B','-X','utf8','-u',(Join-Path $PSScriptRoot 'bootstrap.py'),'--packages',$Packages,
    $action,'--run-id',$RunId,'--model',$Model,'--device',$Device,'--port',[string]$Port,'--review-split',$ReviewSplit,'--seeds')
foreach ($seedValue in $Seeds) { $arguments += [string]$seedValue }
if ($Variants) { $arguments += '--variants'; foreach ($variantValue in $Variants) { $arguments += $variantValue } }
if ($Via) { $arguments += @('--via',$Via) }
Write-Host "Experiment 7 $Stage | $RunId | Python: $Python"
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Experiment 7 $Stage stopped (exit $LASTEXITCODE). Read the missing-item/error list above; prior outputs are preserved." }
