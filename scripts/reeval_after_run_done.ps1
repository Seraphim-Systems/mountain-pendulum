# Wait for the master training launcher to finish, then run the
# student-only re-evaluation pass for the teacher-pretrained REINFORCE
# configs. Existing per-seed _summary.json files are overwritten so the
# notebook reads honest student numbers instead of teacher numbers.

param(
    [int]$MasterPid = 9900,
    [int]$PollSeconds = 30
)

$logDir = "outputs\reeval_student"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$pollLog    = Join-Path $logDir "poll.log"
$stdoutPath = Join-Path $logDir "stdout.log"
$stderrPath = Join-Path $logDir "stderr.log"

"[$(Get-Date -Format s)] Watching PID $MasterPid; will run student re-eval after exit." |
    Out-File $pollLog -Encoding utf8

while ($true) {
    $proc = Get-Process -Id $MasterPid -ErrorAction SilentlyContinue
    if (-not $proc) {
        "[$(Get-Date -Format s)] Master PID $MasterPid no longer running. Launching re-eval." |
            Add-Content $pollLog
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

& .\.venv\Scripts\python.exe -u scripts\student_eval_after_teacher.py `
    --configs configs/reinforce_fuel.yaml `
    --episodes 30 `
    1>$stdoutPath 2>$stderrPath

"[$(Get-Date -Format s)] Re-eval finished (exit code $LASTEXITCODE)." |
    Add-Content $pollLog
