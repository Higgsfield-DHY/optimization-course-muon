param(
    [ValidateSet('all','matrix','pilot','tuning','final','evaluate','report')][string]$Stage='all',
    [switch]$Execute
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$arguments=@('-m','optcourse','study','--stage',$Stage)
if ($Execute) { $arguments+='--execute' }
Push-Location $projectRoot
try {
    & '.\.venv\Scripts\python.exe' @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Study stopped; inspect manifest and budget ledger.' }
} finally { Pop-Location }
