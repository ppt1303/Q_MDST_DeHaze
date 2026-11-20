$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
$pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create .venv with Python 3.12 and install requirements-gpu.txt first.'
}
foreach ($name in @('Sen.pth', 'Tde.pth')) {
    $checkpointPath = Join-Path $repoRoot "results\checkpoints\example\$name"
    if (-not (Test-Path -LiteralPath $checkpointPath)) {
        throw "Missing checkpoint: $checkpointPath"
    }
}
Push-Location (Join-Path $repoRoot 'scripts')
try {
    & $pythonPath inference_H2C.py
    if ($LASTEXITCODE -ne 0) { throw "Inference failed (exit $LASTEXITCODE)." }
} finally {
    Pop-Location
}
