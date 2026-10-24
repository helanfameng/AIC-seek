[CmdletBinding()]
param(
    [ValidateSet('Check', 'Plan', 'Crop', 'Oracle', 'Association', 'Calibration', 'All', 'Summary', 'Tests')]
    [string]$Stage = 'Check',
    [ValidateSet('All', 'Alison', 'West', 'Cranston')]
    [string]$Event = 'All',
    [ValidateSet('Default', 'All', 'Full_Baseline', 'Low_Threshold', 'High_Resolution', 'Combined')]
    [string]$Setting = 'Default',
    [string]$Python = 'D:\AIC\pyronear_lab\.venv\Scripts\python.exe',
    [string]$Model = 'D:\AIC\pyronear_lab\models\v0.6.0\model.zip',
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')]
    [string]$RunId = ((Get-Date).ToUniversalTime().AddHours(8).ToString('yyyyMMdd_HHmmss')),
    [Nullable[int]]$OffsetMin,
    [Nullable[int]]$OffsetMax,
    [ValidateRange(1, 10000)]
    [int]$Limit
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
if ($Stage -eq 'Tests') {
    $arguments = @('-X', 'utf8', '-B', '-m', 'unittest', 'discover', '-s', (Join-Path $PSScriptRoot '..\tests'), '-p', 'test_*.py', '-v')
} else {
    $action = 'run'
    $part = $Stage.ToLowerInvariant()
    if ($Stage -in @('Check', 'Plan', 'Summary')) { $action = $part; $part = 'all' }
    $arguments = @('-X', 'utf8', '-B', (Join-Path $PSScriptRoot 'run_experiment.py'), $action,
        '--stage', $part, '--event', $Event.ToLowerInvariant(), '--setting', $Setting.ToLowerInvariant(),
        '--model', $Model, '--device', $Device, '--run-id', $RunId)
    if ($PSBoundParameters.ContainsKey('OffsetMin')) { $arguments += @('--offset-min', [string]$OffsetMin) }
    if ($PSBoundParameters.ContainsKey('OffsetMax')) { $arguments += @('--offset-max', [string]$OffsetMax) }
    if ($PSBoundParameters.ContainsKey('Limit')) { $arguments += @('--limit', [string]$Limit) }
}
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "exp005 $Stage failed (exit $LASTEXITCODE). Existing outputs have been preserved." }
