$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dependencyCache = Join-Path $projectRoot 'work\uv_cache'
Push-Location $projectRoot
try {
    & uv --cache-dir $dependencyCache sync --extra dev --locked
    if ($LASTEXITCODE -ne 0) { throw 'Environment setup failed.' }
    & '.\.venv\Scripts\python.exe' -m optcourse doctor
    if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
} finally { Pop-Location }
