@echo off
rem Sets up the Auto Maple phone remote (Telegram: /status /start /restart /go /pause ...).
rem Run once: double-click, accept the UAC prompt. It registers a Windows task that
rem starts the remote at every log-on as Administrator (so it can start Auto Maple
rem without a UAC box), and starts it now. Run again any time to reinstall.
rem Remove: schtasks /Delete /TN "Auto Maple Phone Remote" /F

net session >nul 2>&1
if %errorlevel% neq 0 (
    powershell -NoProfile -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$root = 'C:\Users\sumee\projects\auto-maple';" ^
  "$action = New-ScheduledTaskAction -Execute ($root + '\venv\Scripts\pythonw.exe') -Argument ('\"' + $root + '\src\modules\phone_remote.py\"') -WorkingDirectory $root;" ^
  "$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME;" ^
  "$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest;" ^
  "$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew;" ^
  "Register-ScheduledTask -TaskName 'Auto Maple Phone Remote' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null;" ^
  "Start-ScheduledTask -TaskName 'Auto Maple Phone Remote';" ^
  "Write-Host 'Phone remote installed and started. Send /help to your Telegram bot.'"

echo.
echo Log: C:\Users\sumee\projects\auto-maple\logs\phone_remote.log
pause
