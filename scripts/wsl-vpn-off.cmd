@echo off
echo Disabling mirrored networking (restoring Docker port forwarding)...
powershell -Command "(Get-Content '%USERPROFILE%\.wslconfig') -replace '^networkingMode=mirrored', '# networkingMode=mirrored' | Set-Content '%USERPROFILE%\.wslconfig'"
echo Shutting down WSL...
wsl --shutdown
echo Done. Docker ports now accessible from Windows browser.
echo Start Docker Desktop and services as usual.
pause
