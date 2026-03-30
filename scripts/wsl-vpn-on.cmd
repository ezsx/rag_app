@echo off
echo Enabling mirrored networking for VPN in WSL...
powershell -Command "(Get-Content '%USERPROFILE%\.wslconfig') -replace '# networkingMode=mirrored', 'networkingMode=mirrored' | Set-Content '%USERPROFILE%\.wslconfig'"
echo Shutting down WSL...
wsl --shutdown
echo Done. VPN now works in WSL. Docker ports will NOT be accessible from Windows.
echo Don't forget to run wsl-vpn-off.cmd when done!
pause
