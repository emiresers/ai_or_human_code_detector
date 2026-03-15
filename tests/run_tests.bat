@echo off
echo ========================================
echo AI or Human Code Detector - Test Suite
echo ========================================
echo.

echo [1] Test bağımlılıkları kontrol ediliyor...
pip install -q pytest pytest-cov httpx

echo.
echo [2] Testler çalıştırılıyor...
echo.

cd /d %~dp0
pytest test_backend.py -v --tb=short

echo.
echo ========================================
echo Test tamamlandı!
echo ========================================
pause


