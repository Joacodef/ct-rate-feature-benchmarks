param(
    [ValidateSet("both", "manual", "gpt")]
    [string]$Source = "both",
    [int[]]$Budgets = @(100, 500, 1191),
    [int[]]$Seeds = @(42, 123, 456, 789, 999),
    [string]$TestManifestDir = "data/manifests/manual",
    [string[]]$TestManifests = @("FINAL_TEST.csv"),
    [switch]$Force,
    [switch]$StopOnError,
    [switch]$SkipAggregate
)

$ErrorActionPreference = "Stop"

$manualConfig = "best_manual_labels_config.yaml"
$gptConfig = "best_gpt_labels_config.yaml"

$manualHydraJobName = "manual_budget_linear_probe"
$gptHydraJobName = "gpt_budget_linear_probe"

$manualRunsRoot = "outputs/manual_budget_linear_probe"
$gptRunsRoot = "outputs/gpt_budget_linear_probe"

$manualManifestPrefix = "manual_budget_splits/train"
$gptManifestPrefix = "gpt_budget_splits/all"
$manualManifestRoot = "data/manifests/manual"
$gptManifestRoot = "data/manifests/gpt"

$logDir = "outputs/phase1_linear_probe_sweep_logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$failureLog = Join-Path $logDir "failures_$timestamp.log"

$Seeds = $Seeds | Where-Object { $_ -ne $null } | ForEach-Object { [int]$_ } | Sort-Object -Unique
$Budgets = $Budgets | Where-Object { $_ -ne $null } | ForEach-Object { [int]$_ } | Sort-Object -Unique

if ($Seeds.Count -eq 0) {
    throw "No seeds provided. Use -Seeds <int,int,...>."
}

if ($Budgets.Count -eq 0) {
    throw "No budgets provided. Use -Budgets <int,int,...>."
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

Write-Host "Running Phase 1 linear-probe sweep"
Write-Host "Source=$Source"
Write-Host "Budgets=$($Budgets -join ', ')"
Write-Host "Seeds=$($Seeds -join ', ')"
Write-Host "TestManifestDir=$TestManifestDir"
Write-Host "TestManifests=$($TestManifests -join ', ')"

function Invoke-TrainingRun {
    param(
        [Parameter(Mandatory = $true)][string]$ConfigName,
        [Parameter(Mandatory = $true)][string]$TrainManifestPath,
        [Parameter(Mandatory = $true)][string]$ValManifestPath,
        [Parameter(Mandatory = $true)][string]$ManifestRoot,
        [Parameter(Mandatory = $true)][string]$HydraJobName,
        [Parameter(Mandatory = $true)][string]$RunDir,
        [Parameter(Mandatory = $true)][int]$Seed,
        [Parameter(Mandatory = $true)][int]$Budget
    )

    $finalCheckpoint = Join-Path $RunDir "checkpoints/final_model.pt"
    $resolvedTrainManifest = Join-Path $ManifestRoot $TrainManifestPath
    $resolvedValManifest = Join-Path $ManifestRoot $ValManifestPath

    if (-not (Test-Path $resolvedTrainManifest)) {
        throw "Train manifest not found: $resolvedTrainManifest"
    }

    if (-not (Test-Path $resolvedValManifest)) {
        throw "Validation manifest not found: $resolvedValManifest"
    }

    if ((-not $Force) -and (Test-Path $finalCheckpoint)) {
        Write-Host "[SKIP] budget=$Budget seed=$Seed already completed: $finalCheckpoint"
        return
    }

    Write-Host "[RUN ] budget=$Budget seed=$Seed -> $RunDir"

    $runDirHydra = $RunDir.Replace('\\', '/')
    $args = @(
        "-m", "src.classification.train",
        "--config-name", $ConfigName,
        "model=linear_probe",
        "data.auto_split.enabled=false",
        "data.train_manifest=$TrainManifestPath",
        "data.val_manifest=$ValManifestPath",
        "hydra.job.name=$HydraJobName",
        "hydra.run.dir=$runDirHydra",
        "utils.seed=$Seed"
    )

    & python @args
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        $message = "FAILED budget=$Budget seed=$Seed config=$ConfigName train_manifest=$TrainManifestPath val_manifest=$ValManifestPath exit_code=$exitCode"
        Add-Content -Path $failureLog -Value $message
        Write-Host "[FAIL] $message"

        if ($StopOnError) {
            throw "Stopping on first error because -StopOnError was set."
        }
    }
}

function Invoke-Sweep {
    param(
        [Parameter(Mandatory = $true)][int[]]$SourceBudgets,
        [Parameter(Mandatory = $true)][int[]]$SourceSeeds,
        [Parameter(Mandatory = $true)][string]$ConfigName,
        [Parameter(Mandatory = $true)][string]$HydraJobName,
        [Parameter(Mandatory = $true)][string]$RunsRoot,
        [Parameter(Mandatory = $true)][string]$ManifestRoot,
        [Parameter(Mandatory = $true)][string]$ManifestPrefix
    )

    foreach ($budget in $SourceBudgets) {
        foreach ($seed in $SourceSeeds) {
            $trainManifest = "${ManifestPrefix}_n${budget}_s${seed}.csv"
            $valManifest = "${ManifestPrefix}_n${budget}_s${seed}_val.csv"
            $runDir = Join-Path $RunsRoot "train_n${budget}_s${seed}"

            Invoke-TrainingRun `
                -ConfigName $ConfigName `
                -TrainManifestPath $trainManifest `
                -ValManifestPath $valManifest `
                -ManifestRoot $ManifestRoot `
                -HydraJobName $HydraJobName `
                -RunDir $runDir `
                -Seed $seed `
                -Budget $budget
        }
    }
}

function Invoke-Aggregate {
    param(
        [Parameter(Mandatory = $true)][string]$RunsRoot,
        [Parameter(Mandatory = $true)][string]$SourceName,
        [Parameter(Mandatory = $true)][string]$OutputPrefix,
        [Parameter(Mandatory = $true)][string]$TestManifestDirectory,
        [Parameter(Mandatory = $true)][string[]]$SelectedTestManifests
    )

    $joinedTestManifests = $SelectedTestManifests -join ","

    $aggregateArgs = @(
        ".\\scripts\\evaluate_and_aggregate_runs.py",
        "--runs-root", $RunsRoot,
        "--test-manifest-dir", $TestManifestDirectory,
        "--test-manifests", $joinedTestManifests,
        "--source", $SourceName,
        "--output-prefix", $OutputPrefix
    )

    & python @aggregateArgs
    $aggregateExit = $LASTEXITCODE
    if ($aggregateExit -ne 0) {
        throw "Aggregation failed for source=$SourceName with exit code $aggregateExit."
    }
}

try {
    if ($Source -in @("both", "manual")) {
        Write-Host "=== Manual linear-probe sweep ==="
        Invoke-Sweep `
            -SourceBudgets $Budgets `
            -SourceSeeds $Seeds `
            -ConfigName $manualConfig `
            -HydraJobName $manualHydraJobName `
            -RunsRoot $manualRunsRoot `
            -ManifestRoot $manualManifestRoot `
            -ManifestPrefix $manualManifestPrefix
    }

    if ($Source -in @("both", "gpt")) {
        Write-Host "=== GPT linear-probe sweep ==="
        Invoke-Sweep `
            -SourceBudgets $Budgets `
            -SourceSeeds $Seeds `
            -ConfigName $gptConfig `
            -HydraJobName $gptHydraJobName `
            -RunsRoot $gptRunsRoot `
            -ManifestRoot $gptManifestRoot `
            -ManifestPrefix $gptManifestPrefix
    }

    if (Test-Path $failureLog) {
        Write-Host "Sweep finished with failures. See: $failureLog"
        exit 1
    }

    if (-not $SkipAggregate) {
        if ($Source -in @("both", "manual")) {
            Write-Host "Running aggregate evaluation for manual linear-probe runs..."
            Invoke-Aggregate `
                -RunsRoot $manualRunsRoot `
                -SourceName "manual_linear_probe" `
                -OutputPrefix "manual_linear_probe" `
                -TestManifestDirectory $TestManifestDir `
                -SelectedTestManifests $TestManifests
        }

        if ($Source -in @("both", "gpt")) {
            Write-Host "Running aggregate evaluation for GPT linear-probe runs..."
            Invoke-Aggregate `
                -RunsRoot $gptRunsRoot `
                -SourceName "gpt_linear_probe" `
                -OutputPrefix "gpt_linear_probe" `
                -TestManifestDirectory $TestManifestDir `
                -SelectedTestManifests $TestManifests
        }
    }

    Write-Host "Phase 1 linear-probe sweep finished successfully."
    exit 0
}
catch {
    $err = $_.Exception.Message
    Add-Content -Path $failureLog -Value "FATAL: $err"
    Write-Host "[FATAL] $err"
    Write-Host "Failure log: $failureLog"
    exit 1
}
