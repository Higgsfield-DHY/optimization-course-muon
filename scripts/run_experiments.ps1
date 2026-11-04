param(
    [ValidateSet('matrix','language','all')][string]$Task = 'all',
    [int[]]$Seeds = @(0,1,2),
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run scripts/setup.ps1 first.' }
$configs = @()
if ($Task -eq 'matrix' -or $Task -eq 'all') {
    $configs += @('configs/matrix_adamw.toml','configs/matrix_muon.toml')
}
if ($Task -eq 'language' -or $Task -eq 'all') {
    $configs += @('configs/language_adamw.toml','configs/language_muon.toml')
}
$arguments = @('-m','optcourse','batch','--configs') + $configs + @('--seeds') + @($Seeds | ForEach-Object { [string]$_ })
if ($Execute) { $arguments += '--execute' }
Push-Location $projectRoot
try {
    & $pythonPath @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Experiment command failed; inspect error and run metadata.' }
} finally { Pop-Location }
