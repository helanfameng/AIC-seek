[CmdletBinding()]
param(
    [ValidateSet('Download', 'Verify', 'Plan', 'Check', 'FullBaseline', 'LowThreshold', 'HighResolution', 'Combined', 'All', 'Summary')]
    [string]$Stage = 'Check',
    [ValidateSet('All', 'Alison', 'West', 'Cranston')]
    [string]$Event = 'All',
    [string]$Python = 'D:\AIC\pyronear_lab\.venv\Scripts\python.exe',
    [string]$Model = 'D:\AIC\pyronear_lab\models\v0.6.0\model.zip',
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')]
    [string]$RunId = (Get-Date -Format 'yyyyMMdd_HHmmss')
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$eventKey = $Event.ToLowerInvariant()

if ($Stage -eq 'Download' -or $Stage -eq 'Verify') {
    $arguments = @('-X', 'utf8', '-B', (Join-Path $PSScriptRoot 'download_events.py'), '--event', $eventKey)
    if ($Stage -eq 'Verify') { $arguments += '--verify-only' }
} else {
    $action = 'run'
    $setting = 'all'
    switch ($Stage) {
        'Plan' { $action = 'plan' }
        'Check' { $action = 'check' }
        'Summary' { $action = 'summary' }
        'FullBaseline' { $setting = 'full_baseline' }
        'LowThreshold' { $setting = 'low_threshold' }
        'HighResolution' { $setting = 'high_resolution' }
        'Combined' { $setting = 'combined' }
    }
    $arguments = @('-X', 'utf8', '-B', (Join-Path $PSScriptRoot 'run_experiments.py'), $action,
        '--event', $eventKey, '--stage', $setting, '--model', $Model, '--device', $Device, '--run-id', $RunId)
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "exp004 $Stage failed (exit $LASTEXITCODE). Existing results have been preserved."
}
