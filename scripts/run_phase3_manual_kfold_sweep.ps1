param(
    [ValidateSet("manual", "gpt")]
    [string]$LabelSource = "manual",
    [string]$ConfigName,
    [string]$ManifestRoot,
    [string]$TestManifestRoot,
    [string]$ManifestIndexPath,
    [string]$RunsRoot,
    [string]$HydraJobName,
    [string]$AggregateSource,
    [string]$AggregateOutputPrefix,
    [string]$PythonExe,
    [int]$Seed,
    [int[]]$Seeds,
    [switch]$Force,
    [switch]$StopOnError,
    [switch]$SkipAggregate
)

$ErrorActionPreference = "Stop"

$sourceDefaults = @{
    manual = @{
        ConfigName = "best_manual_labels_config.yaml"
        ManifestRoot = "data/manifests/manual/manual_kfold_budget_splits"
        TestManifestRoot = "data/manifests/manual/manual_kfold_budget_splits"
        ManifestIndexPath = "data/manifests/manual/manual_kfold_budget_splits/manifest_index.csv"
        RunsRoot = "outputs/manual_kfold_budget"
        HydraJobName = "manual_kfold_budget"
        AggregateSource = "manual_kfold"
        AggregateOutputPrefix = "manual_kfold_budget"
    }
    gpt = @{
        ConfigName = "best_gpt_labels_config.yaml"
        ManifestRoot = "data/manifests/gpt/gpt_kfold_budget_splits"
        TestManifestRoot = "data/manifests/manual/manual_kfold_budget_splits"
        ManifestIndexPath = "data/manifests/gpt/gpt_kfold_budget_splits/manifest_index.csv"
        RunsRoot = "outputs/gpt_kfold_budget"
        HydraJobName = "gpt_kfold_budget"
        AggregateSource = "gpt_kfold"
        AggregateOutputPrefix = "gpt_kfold_budget"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path

if (-not $PSBoundParameters.ContainsKey("PythonExe")) {
    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $PythonExe = $venvPython
    } else {
        $PythonExe = "python"
    }
}

if ($PythonExe -eq "python") {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        throw "Python executable not found. Pass -PythonExe <path-to-python.exe>."
    }
} elseif (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$srcPath = Join-Path $projectRoot "src"
if (-not (Test-Path $srcPath)) {
    throw "Source path not found: $srcPath"
}

if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $env:PYTHONPATH = $srcPath
} elseif (-not ($env:PYTHONPATH.Split(';') -contains $srcPath)) {
    $env:PYTHONPATH = "$srcPath;$($env:PYTHONPATH)"
}

$defaults = $sourceDefaults[$LabelSource]
if (-not $PSBoundParameters.ContainsKey("ConfigName")) { $ConfigName = $defaults.ConfigName }
if (-not $PSBoundParameters.ContainsKey("ManifestRoot")) { $ManifestRoot = $defaults.ManifestRoot }
if (-not $PSBoundParameters.ContainsKey("TestManifestRoot")) { $TestManifestRoot = $defaults.TestManifestRoot }
if (-not $PSBoundParameters.ContainsKey("ManifestIndexPath")) { $ManifestIndexPath = $defaults.ManifestIndexPath }
if (-not $PSBoundParameters.ContainsKey("RunsRoot")) { $RunsRoot = $defaults.RunsRoot }
if (-not $PSBoundParameters.ContainsKey("HydraJobName")) { $HydraJobName = $defaults.HydraJobName }
if (-not $PSBoundParameters.ContainsKey("AggregateSource")) { $AggregateSource = $defaults.AggregateSource }
if (-not $PSBoundParameters.ContainsKey("AggregateOutputPrefix")) { $AggregateOutputPrefix = $defaults.AggregateOutputPrefix }

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
$defaultSeeds = @(52, 123, 456, 789, 999)

$seedList = @()
if ($PSBoundParameters.ContainsKey("Seeds")) {
    $seedList = $Seeds | Where-Object { $_ -ne $null } | ForEach-Object { [int]$_ }
} elseif ($PSBoundParameters.ContainsKey("Seed")) {
    $seedList = @([int]$Seed)
} else {
    $seedList = $defaultSeeds
}

if ($seedList.Count -eq 0) {
    throw "No seeds provided. Use -Seed <int> or -Seeds <int,int,...>."
}

$seedList = $seedList | Sort-Object -Unique
Write-Host "Running Phase 3 K-fold sweep for source=$LabelSource with seeds: $($seedList -join ', ')"
Write-Host "Config=$ConfigName"
Write-Host "ManifestRoot=$ManifestRoot"
Write-Host "TestManifestRoot=$TestManifestRoot"
Write-Host "ManifestIndexPath=$ManifestIndexPath"
Write-Host "RunsRoot=$RunsRoot"
Write-Host "PythonExe=$PythonExe"
Write-Host "PYTHONPATH=$($env:PYTHONPATH)"

$rows = Import-Csv $ManifestIndexPath | Sort-Object @{Expression = {[int]$_.fold}}, @{Expression = {[int]$_.requested_budget}}

function Invoke-Phase3Run {
    param(
        [Parameter(Mandatory = $true)][pscustomobject]$Row,
        [Parameter(Mandatory = $true)][int]$RunSeed
    )

    $fold = [int]$Row.fold
    $budget = [int]$Row.requested_budget
    $trainManifest = [string]$Row.train_manifest
    $valManifest = [string]$Row.val_manifest
    $testManifest = [string]$Row.test_manifest

    $resolvedTrain = Join-Path $ManifestRoot $trainManifest
    $resolvedVal = Join-Path $ManifestRoot $valManifest
    $resolvedTest = Join-Path $TestManifestRoot $testManifest

    if (-not (Test-Path $resolvedTrain)) { throw "Train manifest not found: $resolvedTrain" }
    if (-not (Test-Path $resolvedVal)) { throw "Validation manifest not found: $resolvedVal" }
    if (-not (Test-Path $resolvedTest)) { throw "Test manifest not found: $resolvedTest" }

    $runDir = Join-Path $RunsRoot ("f{0}_n{1}_s{2}" -f $fold, $budget, $RunSeed)
    $finalCheckpoint = Join-Path $runDir "checkpoints/final_model.pt"

    if ((-not $Force) -and (Test-Path $finalCheckpoint)) {
        Write-Host "[SKIP] fold=$fold budget=$budget already completed: $finalCheckpoint"
        return
    }

    Write-Host "[RUN ] fold=$fold budget=$budget seed=$RunSeed -> $runDir"

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
        "utils.seed=$RunSeed"
    )

    & $PythonExe @args
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        $message = "FAILED fold=$fold budget=$budget seed=$RunSeed train_manifest=$trainManifest val_manifest=$valManifest test_manifest=$testManifest exit_code=$exitCode"
        Add-Content -Path $failureLog -Value $message
        Write-Host "[FAIL] $message"

        if ($StopOnError) {
            throw "Stopping on first error because -StopOnError was set."
        }
    }
}

try {
    foreach ($currentSeed in $seedList) {
        Write-Host "Starting sweep for seed=$currentSeed"
        foreach ($row in $rows) {
            Invoke-Phase3Run -Row $row -RunSeed $currentSeed
        }
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
            "--test-manifest-dir", $TestManifestRoot,
            "--fold-map-csv", $ManifestIndexPath,
            "--source", $AggregateSource,
            "--output-prefix", $AggregateOutputPrefix
        )

        & $PythonExe @aggregateArgs
        $aggregateExit = $LASTEXITCODE
        if ($aggregateExit -ne 0) {
            throw "Aggregation failed with exit code $aggregateExit."
        }
    }

    Write-Host "Phase 3 K-fold sweep finished successfully for source=$LabelSource."
    exit 0
}
catch {
    $err = $_.Exception.Message
    Add-Content -Path $failureLog -Value "FATAL: $err"
    Write-Host "[FATAL] $err"
    Write-Host "Failure log: $failureLog"
    exit 1
}
