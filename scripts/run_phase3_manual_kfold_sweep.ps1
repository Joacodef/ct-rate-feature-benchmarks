param(
    [string]$ConfigName = "best_manual_labels_config.yaml",
    [string]$ManifestRoot = "data/manifests/manual/manual_kfold_budget_splits",
    [string]$ManifestIndexPath = "data/manifests/manual/manual_kfold_budget_splits/manifest_index.csv",
    [string]$RunsRoot = "outputs/manual_kfold_budget",
    [string]$HydraJobName = "manual_kfold_budget",
    [int]$Seed = 52,
    [switch]$Force,
    [switch]$StopOnError,
    [switch]$SkipAggregate
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ManifestIndexPath)) {
    throw "Manifest index not found: $ManifestIndexPath"
}

if (-not (Test-Path $ManifestRoot)) {
    throw "Manifest root not found: $ManifestRoot"
}

New-Item -ItemType Directory -Path $RunsRoot -Force | Out-Null
$logDir = "outputs/phase3_sweep_logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$failureLog = Join-Path $logDir "kfold_failures_$timestamp.log"

$rows = Import-Csv $ManifestIndexPath | Sort-Object @{Expression = {[int]$_.fold}}, @{Expression = {[int]$_.requested_budget}}

function Invoke-Phase3Run {
    param(
        [Parameter(Mandatory = $true)][pscustomobject]$Row
    )

    $fold = [int]$Row.fold
    $budget = [int]$Row.requested_budget
    $trainManifest = [string]$Row.train_manifest
    $valManifest = [string]$Row.val_manifest
    $testManifest = [string]$Row.test_manifest

    $resolvedTrain = Join-Path $ManifestRoot $trainManifest
    $resolvedVal = Join-Path $ManifestRoot $valManifest
    $resolvedTest = Join-Path $ManifestRoot $testManifest

    if (-not (Test-Path $resolvedTrain)) { throw "Train manifest not found: $resolvedTrain" }
    if (-not (Test-Path $resolvedVal)) { throw "Validation manifest not found: $resolvedVal" }
    if (-not (Test-Path $resolvedTest)) { throw "Test manifest not found: $resolvedTest" }

    $runDir = Join-Path $RunsRoot ("f{0}_n{1}_s{2}" -f $fold, $budget, $Seed)
    $finalCheckpoint = Join-Path $runDir "checkpoints/final_model.pt"

    if ((-not $Force) -and (Test-Path $finalCheckpoint)) {
        Write-Host "[SKIP] fold=$fold budget=$budget already completed: $finalCheckpoint"
        return
    }

    Write-Host "[RUN ] fold=$fold budget=$budget seed=$Seed -> $runDir"

    $manifestRootHydra = $ManifestRoot.Replace('\\', '/')
    $runDirHydra = $runDir.Replace('\\', '/')

    $args = @(
        "-m", "src.classification.train",
        "--config-name", $ConfigName,
        "data.auto_split.enabled=false",
        "paths.manifest_dir=$manifestRootHydra",
        "data.train_manifest=$trainManifest",
        "data.val_manifest=$valManifest",
        "data.test_manifests=[$testManifest]",
        "hydra.job.name=$HydraJobName",
        "hydra.run.dir=$runDirHydra",
        "utils.seed=$Seed"
    )

    & python @args
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        $message = "FAILED fold=$fold budget=$budget seed=$Seed train_manifest=$trainManifest val_manifest=$valManifest test_manifest=$testManifest exit_code=$exitCode"
        Add-Content -Path $failureLog -Value $message
        Write-Host "[FAIL] $message"

        if ($StopOnError) {
            throw "Stopping on first error because -StopOnError was set."
        }
    }
}

try {
    foreach ($row in $rows) {
        Invoke-Phase3Run -Row $row
    }

    if (Test-Path $failureLog) {
        Write-Host "Sweep finished with failures. See: $failureLog"
        exit 1
    }

    if (-not $SkipAggregate) {
        Write-Host "Running aggregate evaluation for all K-fold runs..."
        $aggregateArgs = @(
            ".\scripts\evaluate_and_aggregate_runs.py",
            "--runs-root", $RunsRoot,
            "--test-manifest-dir", $ManifestRoot,
            "--source", "manual_kfold",
            "--output-prefix", "manual_kfold_budget"
        )

        & python @aggregateArgs
        $aggregateExit = $LASTEXITCODE
        if ($aggregateExit -ne 0) {
            throw "Aggregation failed with exit code $aggregateExit."
        }
    }

    Write-Host "Phase 3 K-fold sweep finished successfully."
    exit 0
}
catch {
    $err = $_.Exception.Message
    Add-Content -Path $failureLog -Value "FATAL: $err"
    Write-Host "[FATAL] $err"
    Write-Host "Failure log: $failureLog"
    exit 1
}
