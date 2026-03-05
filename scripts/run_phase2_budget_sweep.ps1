param(
    [ValidateSet("both", "manual", "gpt")]
    [string]$Source = "both",
    [switch]$Force,
    [switch]$StopOnError
)

$ErrorActionPreference = "Stop"

$seeds = @(42, 123, 456, 789, 999, 2344, 5678, 9012, 3456, 6789, 23423, 54321, 98765, 43210, 11111)
$manualBudgets = @(20,50,100,250,500,800,1191)
$gptBudgets = @(20, 50, 100, 250, 500, 800, 1191, 2000, 5000, 10000, 20000, 46438)

$manualConfig = "best_manual_labels_config.yaml"
$gptConfig = "best_gpt_labels_config.yaml"

$manualHydraJobName = "manual_budget"
$gptHydraJobName = "gpt_budget"

$manualRunsRoot = "outputs/manual_budget"
$gptRunsRoot = "outputs/gpt_budget"

$manualManifestPrefix = "manual_budget_splits/train"
$gptManifestPrefix = "gpt_budget_splits/all"
$manualManifestRoot = "data/manifests/manual"
$gptManifestRoot = "data/manifests/gpt"

$logDir = "outputs/phase2_sweep_logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$failureLog = Join-Path $logDir "failures_$timestamp.log"

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

    $args = @(
        "-m", "src.classification.train",
        "--config-name", $ConfigName,
        "data.auto_split.enabled=false",
        "data.train_manifest=$TrainManifestPath",
        "data.val_manifest=$ValManifestPath",
        "hydra.job.name=$HydraJobName",
        "hydra.run.dir=$RunDir",
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
        [Parameter(Mandatory = $true)][int[]]$Budgets,
        [Parameter(Mandatory = $true)][string]$ConfigName,
        [Parameter(Mandatory = $true)][string]$HydraJobName,
        [Parameter(Mandatory = $true)][string]$RunsRoot,
        [Parameter(Mandatory = $true)][string]$ManifestRoot,
        [Parameter(Mandatory = $true)][string]$ManifestPrefix
    )

    foreach ($budget in $Budgets) {
        foreach ($seed in $seeds) {
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

try {
    if ($Source -in @("both", "manual")) {
        Write-Host "=== Manual budget sweep ==="
        Invoke-Sweep `
            -Budgets $manualBudgets `
            -ConfigName $manualConfig `
            -HydraJobName $manualHydraJobName `
            -RunsRoot $manualRunsRoot `
            -ManifestRoot $manualManifestRoot `
            -ManifestPrefix $manualManifestPrefix
    }

    if ($Source -in @("both", "gpt")) {
        Write-Host "=== GPT budget sweep ==="
        Invoke-Sweep `
            -Budgets $gptBudgets `
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

    Write-Host "Sweep finished successfully."
    exit 0
}
catch {
    $err = $_.Exception.Message
    Add-Content -Path $failureLog -Value "FATAL: $err"
    Write-Host "[FATAL] $err"
    Write-Host "Failure log: $failureLog"
    exit 1
}
