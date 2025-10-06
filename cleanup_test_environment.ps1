#Requires -Version 7.0

<#
.SYNOPSIS
    Очистка тестового окружения
.DESCRIPTION
    Останавливает контейнеры, очищает логи и временные файлы
#>

Write-Host "=== ОЧИСТКА ТЕСТОВОГО ОКРУЖЕНИЯ ===" -ForegroundColor Cyan

# Функции логирования
function Write-Success { param($Message) Write-Host "[OK] $Message" -ForegroundColor Green }
function Write-Error { param($Message) Write-Host "[FAIL] $Message" -ForegroundColor Red }
function Write-Info { param($Message) Write-Host "[INFO] $Message" -ForegroundColor Blue }

try {
    # Шаг 1: Остановка контейнеров
    Write-Info "Шаг 1: Остановка контейнеров..."

    try {
        docker-compose down
        Write-Success "Контейнеры остановлены"
    } catch {
        Write-Warning "Не удалось остановить контейнеры: $_"
    }

    # Шаг 2: Очистка логов
    Write-Info "Шаг 2: Очистка логов..."

    $logFiles = @(
        "test_agent_logic.log",
        "test_agent_dataset.log",
        "test_agent_minimal.log",
        "test_results.log",
        "minimal_test.log"
    )

    foreach ($logFile in $logFiles) {
        if (Test-Path $logFile) {
            Remove-Item $logFile -Force
            Write-Info "Удален лог: $logFile"
        }
    }

    # Шаг 3: Очистка временных файлов
    Write-Info "Шаг 3: Очистка временных файлов..."

    $tempFiles = @(
        "test_agent_basic.py",
        "test_agent_logic.py",
        "test_agent_dataset.py",
        "test_agent_minimal.py",
        "test_agent_full.ps1",
        "test_agent_logic_only.ps1",
        "test_agent_with_containers.ps1",
        "test_agent_real_data.ps1",
        "cleanup_test_environment.ps1"
    )

    foreach ($tempFile in $tempFiles) {
        if (Test-Path $tempFile) {
            Remove-Item $tempFile -Force
            Write-Info "Удален временный файл: $tempFile"
        }
    }

    Write-Host "`n=== ОЧИСТКА ЗАВЕРШЕНА ===" -ForegroundColor Cyan
    Write-Host "Окружение очищено" -ForegroundColor Green

} catch {
    Write-Error "Ошибка очистки: $_"
    exit 1
}
