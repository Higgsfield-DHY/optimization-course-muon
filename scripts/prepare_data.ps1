param([string]$InputPath, [switch]$DownloadShakespeare)
$ErrorActionPreference = 'Stop'
if ((-not $InputPath -and -not $DownloadShakespeare) -or ($InputPath -and $DownloadShakespeare)) {
    throw 'Choose exactly one: -InputPath <local text> or -DownloadShakespeare.'
}
if ($InputPath) { $InputPath = (Resolve-Path -LiteralPath $InputPath).Path }
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$arguments = @('-m','optcourse','prepare-data','--output','data/shakespeare_byte')
if ($DownloadShakespeare) { $arguments += '--download-shakespeare' } else { $arguments += @('--input',$InputPath) }
Push-Location $projectRoot
try {
    & $pythonPath @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed.' }
} finally { Pop-Location }
