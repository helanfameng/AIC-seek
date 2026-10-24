[CmdletBinding()]
param(
    [ValidateSet('Check','Smoke','Data','Prepare','Review','ImportVia','Extract','Prelabel','Predictor','Fusion','Train','Calibrate','Evaluate','Tests')]
    [string]$Stage = 'Check',
    [string]$Python = 'D:\AIC\pyronear_lab\.venv\Scripts\python.exe',
    [string]$Model = 'D:\AIC\pyronear_lab\models\v0.6.0\model.zip',
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [ValidateSet('baseline','high_resolution')][string]$Setting = 'baseline',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')]
    [string]$RunId = ('exp006_' + (Get-Date).ToString('yyyyMMdd_HHmmss')),
    [int]$Seed = 42,
    [ValidateSet('all','figlib','hpwren')][string]$Source = 'all',
    [string[]]$Sequence,
    [ValidateRange(0,100000)][int]$LimitSequences = 0,
    [ValidateRange(0,100000)][int]$LimitFrames = 0,
    [ValidateRange(1,16)][int]$Workers = 4,
    [ValidateRange(1024,65535)][int]$Port = 8766,
    [ValidateSet('develop','train','val','test','all')][string]$ReviewSplit = 'develop',
    [string]$Via
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$action = if ($Stage -eq 'ImportVia') { 'import-via' } else { $Stage.ToLowerInvariant() }
$arguments = @('-B','-X','utf8','-u',(Join-Path $PSScriptRoot 'run_experiment.py'),$action,
    '--run-id',$RunId,'--setting',$Setting,'--model',$Model,'--device',$Device,
    '--seed',[string]$Seed,'--source',$Source,'--limit-sequences',[string]$LimitSequences,
    '--limit-frames',[string]$LimitFrames,'--workers',[string]$Workers,'--port',[string]$Port,
    '--review-split',$ReviewSplit)
foreach ($item in $Sequence) { $arguments += @('--sequence',$item) }
if ($Via) { $arguments += @('--via',$Via) }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "exp006 $Stage failed (exit $LASTEXITCODE). Existing data have been preserved." }
