param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectPath,

    [Parameter(Mandatory = $true)]
    [string]$SpaceUrl,

    [string]$TempDirName = ".hf-space-deploy-temp",
    [string]$CommitMessage = "Deploy Space snapshot"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ProjectPath)) {
    throw "ProjectPath not found: $ProjectPath"
}

$repoRoot = (Get-Location).Path
$tempDir = Join-Path $repoRoot $TempDirName

if (Test-Path $tempDir) {
    Remove-Item -Recurse -Force $tempDir
}

New-Item -ItemType Directory -Path $tempDir | Out-Null

# Copy a clean snapshot of the project content.
robocopy $ProjectPath $tempDir /E /XD .git __pycache__ .venv venv env data\processed data\clean model | Out-Null

Push-Location $tempDir
try {
    git init -b main | Out-Null
    git config user.name "Marcilio"
    git config user.email "marcilio.dfn@gmail.com"
    git add .
    git commit -m $CommitMessage | Out-Null
    git remote add origin $SpaceUrl
    git push origin main --force
}
finally {
    Pop-Location
    if (Test-Path $tempDir) {
        Remove-Item -Recurse -Force $tempDir
    }
}

Write-Host "Deploy finished successfully to $SpaceUrl"
