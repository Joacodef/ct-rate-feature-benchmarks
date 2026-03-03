param(
    [ValidateSet("both", "manual", "gpt")]
    [string]$Source = "both",
    [switch]$Force,
    [switch]$StopOnError
)

$ErrorActionPreference = "Stop"

$seeds = @(42, 123, 456, 789, 999)
$manualBudgets = @(1191) # original: (20, 50, 100, 250, 500, 800, 1071)
$gptBudgets = @(1071, 46438) # original: (20, 50, 100, 250, 500, 800, 1071, 2000, 5000, 10000, 20000, 40000)

$manualConfig = "best_manual_labels_config.yaml"
$gptConfig = "best_gpt_labels_config.yaml"

$manualHydraJobName = "manual_budget"
$gptHydraJobName = "gpt_budget"

$manualRunsRoot = "outputs/manual_budget"
$gptRunsRoot = "outputs/gpt_budget"

$manualManifestPrefix = "manual_budget_splits/test_manual_train"
$gptManifestPrefix = "gpt_budget_splits/all"

$logDir = "outputs/phase2_sweep_logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$failureLog = Join-Path $logDir "failures_$timestamp.log"

function Invoke-TrainingRun {
    param(
        [Parameter(Mandatory = $true)][string]$ConfigName,
        [Parameter(Mandatory = $true)][string]$TrainManifestPath,
        [Parameter(Mandatory = $true)][string]$ValManifestPath,
        [Parameter(Mandatory = $true)][string]$HydraJobName,
        [Parameter(Mandatory = $true)][string]$RunDir,
        [Parameter(Mandatory = $true)][int]$Seed,
        [Parameter(Mandatory = $true)][int]$Budget
    )

    $finalCheckpoint = Join-Path $RunDir "checkpoints/final_model.pt"

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
            -ManifestPrefix $manualManifestPrefix
    }

    if ($Source -in @("both", "gpt")) {
        Write-Host "=== GPT budget sweep ==="
        Invoke-Sweep `
            -Budgets $gptBudgets `
            -ConfigName $gptConfig `
            -HydraJobName $gptHydraJobName `
            -RunsRoot $gptRunsRoot `
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
