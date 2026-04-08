param(
    [Parameter(Mandatory = $true)]
    [string]$DataPath,
    [string]$ModelName = "gpt2",
    [string]$OutputRoot = "outputs/decoder_phase5",
    [double]$FrocEps = 0.02,
    [switch]$SkipPlots
)

$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd_HHmm"
$archiveDir = Join-Path "results/archive" $timestamp

if (Test-Path $OutputRoot) {
    New-Item -ItemType Directory -Force -Path $archiveDir | Out-Null
    Copy-Item -Path $OutputRoot -Destination (Join-Path $archiveDir "decoder_phase5") -Recurse -Force
}

$plotArg = ""
if ($SkipPlots.IsPresent) {
    $plotArg = "--skip_plots"
}

$cmd = @(
    "scripts/run_decoder_phase5.py",
    "--data_path", "$DataPath",
    "--model_name", "$ModelName",
    "--froc-mode", "both",
    "--froc-eps", "$FrocEps",
    "--output_dir", "$OutputRoot"
)

if ($plotArg -ne "") {
    $cmd += $plotArg
}

$commandText = "python " + ($cmd -join " ")
Write-Host "Running: $commandText"

$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "."
try {
    & python @cmd
}
finally {
    $env:PYTHONPATH = $oldPythonPath
}

$required = @(
    (Join-Path $OutputRoot "phase23_strict/phase23_verification_report.md"),
    (Join-Path $OutputRoot "phase23_strict/transport_diagnostics.json"),
    (Join-Path $OutputRoot "phase23_strict/threshold_invariance.csv"),
    (Join-Path $OutputRoot "phase23_pragmatic/phase23_verification_report.md"),
    (Join-Path $OutputRoot "phase23_pragmatic/transport_diagnostics.json"),
    (Join-Path $OutputRoot "phase23_pragmatic/threshold_invariance.csv")
)

$missing = @()
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        $missing += $path
    }
}

if ($missing.Count -gt 0) {
    Write-Error "Pipeline completed but required artifacts are missing:`n$($missing -join "`n")"
}

Write-Host "FROC phase23 pipeline finished successfully."
Write-Host "Strict outputs: $(Join-Path $OutputRoot "phase23_strict")"
Write-Host "Pragmatic outputs: $(Join-Path $OutputRoot "phase23_pragmatic")"
