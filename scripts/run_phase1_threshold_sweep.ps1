param(
    [ValidateSet("both", "manual", "gpt")]
    [string]$Source = "both",
    [int[]]$Seeds = @(42, 123, 456, 789, 999),
    [string]$TestManifestDir = "data/manifests/manual",
    [string[]]$TestManifests = @("FINAL_TEST.csv"),
    [switch]$Force,
    [switch]$StopOnError
)

$ErrorActionPreference = "Stop"

$manualRunsRoot = "outputs/optuna_manual_labels_studies/best_model_evaluation_redo"
$gptRunsRoot = "outputs/optuna_gpt_labels_studies/best_model_evaluation"

$logDir = "outputs/phase1_threshold_sweep_logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$failureLog = Join-Path $logDir "failures_$timestamp.log"

$Seeds = $Seeds | Where-Object { $_ -ne $null } | ForEach-Object { [int]$_ } | Sort-Object -Unique

if ($Seeds.Count -eq 0) {
    throw "No seeds provided. Use -Seeds <int,int,...>."
}

if (-not (Test-Path $TestManifestDir)) {
    throw "Test manifest directory not found: $TestManifestDir"
}

if ($TestManifests.Count -eq 0) {
    throw "No test manifests provided. Use -TestManifests <name1,name2,...>."
}

foreach ($testManifest in $TestManifests) {
    $resolvedTestManifest = Join-Path $TestManifestDir $testManifest
    if (-not (Test-Path $resolvedTestManifest)) {
        throw "Test manifest not found: $resolvedTestManifest"
    }
}

$joinedTestManifests = ($TestManifests | ForEach-Object { $_.Trim() }) -join ","

Write-Host "Running Phase 1 threshold sweep"
Write-Host "Source=$Source"
Write-Host "Seeds=$($Seeds -join ', ')"
Write-Host "TestManifestDir=$TestManifestDir"
Write-Host "TestManifests=$($TestManifests -join ', ')"

function Resolve-ManualRunDir {
    param([Parameter(Mandatory = $true)][int]$Seed)

    $runDir = Join-Path $manualRunsRoot "s$Seed"
    $cfgPath = Join-Path $runDir ".hydra/config.yaml"
    $ckptPath = Join-Path $runDir "checkpoints/final_model.pt"

    if (-not (Test-Path $cfgPath)) {
        throw "Manual run config not found for seed=$Seed at $cfgPath"
    }
    if (-not (Test-Path $ckptPath)) {
        throw "Manual checkpoint not found for seed=$Seed at $ckptPath"
    }

    return $runDir
}

function Resolve-GptRunDir {
    param([Parameter(Mandatory = $true)][int]$Seed)

    $seedDir = Join-Path $gptRunsRoot "best_model_seed_$Seed"
    if (-not (Test-Path $seedDir)) {
        throw "GPT seed directory not found for seed=$Seed at $seedDir"
    }

    # Pick latest timestamped run containing both config and checkpoint.
    $candidates = Get-ChildItem -Path $seedDir -Directory |
        Sort-Object LastWriteTime -Descending |
        Where-Object {
            (Test-Path (Join-Path $_.FullName ".hydra/config.yaml")) -and
            (Test-Path (Join-Path $_.FullName "checkpoints/final_model.pt"))
        }

    if (-not $candidates -or $candidates.Count -eq 0) {
        throw "No valid timestamped GPT run found for seed=$Seed under $seedDir"
    }

    return $candidates[0].FullName
}

function Invoke-OptimizeThresholds {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$RunDir,
        [Parameter(Mandatory = $true)][int]$Seed
    )

    $reportPath = Join-Path $RunDir "evaluation_optimized/optimized_thresholds_report.json"
    if ((-not $Force) -and (Test-Path $reportPath)) {
        Write-Host "[SKIP] $Label seed=$Seed already has optimized report: $reportPath"
        return
    }

    Write-Host "[RUN ] $Label seed=$Seed -> $RunDir"

    $args = @(
        ".\\scripts\\optimize_thresholds.py",
        "--run-dir", $RunDir,
        "--test-manifest-dir", $TestManifestDir,
        "--test-manifests", $joinedTestManifests
    )

    & python @args
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $message = "FAILED source=$Label seed=$Seed run_dir=$RunDir exit_code=$exitCode"
        Add-Content -Path $failureLog -Value $message
        Write-Host "[FAIL] $message"

        if ($StopOnError) {
            throw "Stopping on first error because -StopOnError was set."
        }
    }
}

try {
    foreach ($seed in $Seeds) {
        if ($Source -in @("both", "manual")) {
            $manualRunDir = Resolve-ManualRunDir -Seed $seed
            Invoke-OptimizeThresholds -Label "manual" -RunDir $manualRunDir -Seed $seed
        }

        if ($Source -in @("both", "gpt")) {
            $gptRunDir = Resolve-GptRunDir -Seed $seed
            Invoke-OptimizeThresholds -Label "gpt" -RunDir $gptRunDir -Seed $seed
        }
    }

    if (Test-Path $failureLog) {
        Write-Host "Threshold sweep finished with failures. See: $failureLog"
        exit 1
    }

    Write-Host "Threshold sweep finished successfully."
    exit 0
}
catch {
    $err = $_.Exception.Message
    Add-Content -Path $failureLog -Value "FATAL: $err"
    Write-Host "[FATAL] $err"
    Write-Host "Failure log: $failureLog"
    exit 1
}
