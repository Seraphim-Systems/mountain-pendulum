# Wait for the master training launcher to finish, then re-run only
# dqn_continuous (now using n_actions=5 instead of the original 3).
# This is needed because with 3-bin discretisation |a| == a^2, which made
# dqn_continuous and dqn_minsteps mathematically identical.

param(
    [int]$MasterPid = 46200,
    [int]$PollSeconds = 30
)

$logDir = "outputs\rerun_dqn_continuous"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdoutPath = Join-Path $logDir "stdout.log"
$stderrPath = Join-Path $logDir "stderr.log"
$pollLog    = Join-Path $logDir "poll.log"

"[$(Get-Date -Format s)] Watching PID $MasterPid; will rerun dqn_continuous after exit." |
    Out-File $pollLog -Encoding utf8

while ($true) {
    $proc = Get-Process -Id $MasterPid -ErrorAction SilentlyContinue
    if (-not $proc) {
        "[$(Get-Date -Format s)] Master PID $MasterPid no longer running. Launching rerun." |
            Add-Content $pollLog
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

# Remove the previous (n_actions=3) artefacts so the rerun is clean.
Get-ChildItem outputs\logs -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like 'dqn_continuous*' } |
    ForEach-Object { Remove-Item $_.FullName -Recurse -Force }
Get-ChildItem outputs\models -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like 'dqn_continuous*' } |
    ForEach-Object { Remove-Item $_.FullName -Recurse -Force }

"[$(Get-Date -Format s)] Cleaned previous dqn_continuous artefacts. Starting rerun." |
    Add-Content $pollLog

& .\.venv\Scripts\python.exe -u -m src.training.train `
    --config configs/dqn_continuous.yaml `
    --project-root . `
    1>$stdoutPath 2>$stderrPath

"[$(Get-Date -Format s)] Rerun finished (exit code $LASTEXITCODE)." |
    Add-Content $pollLog
