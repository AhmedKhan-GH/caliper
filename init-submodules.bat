@echo off
REM Script to properly initialize git submodules on Windows
REM This removes build artifacts and existing submodule directories before reinitializing

echo Starting submodule initialization process...

REM Remove build directories
echo Removing build directories...
if exist cmake-build-debug rmdir /s /q cmake-build-debug
if exist cmake-build-release rmdir /s /q cmake-build-release
if exist build rmdir /s /q build

REM Deinitialize and clean submodules first
echo Cleaning git submodules...
git submodule deinit -f --all 2>nul

REM Remove submodule entries from .git
echo Removing .git/modules...
if exist .git\modules rmdir /s /q .git\modules

REM Remove all third_party directories forcefully
echo Removing all third_party directories...
if exist third_party rmdir /s /q third_party
mkdir third_party

REM Remove pytorch from .gitmodules permanently (it's not a real submodule)
echo Removing pytorch from .gitmodules...
git config -f .gitmodules --get-all submodule.third_party/pytorch.path >nul 2>&1
if %errorlevel% equ 0 (
    git config -f .gitmodules --remove-section submodule.third_party/pytorch
    git add .gitmodules
    git commit -m "Remove pytorch submodule (using libtorch binaries instead)" --no-verify 2>nul
    echo   [OK] PyTorch submodule entry removed
)

REM Remove pytorch from .git/config as well
git config --remove-section submodule.third_party/pytorch 2>nul

REM Sync submodule URLs
echo Syncing submodule URLs...
git submodule sync --recursive

REM Initialize and update submodules (excluding pytorch)
echo Initializing submodules...
git submodule update --init --recursive --force

REM Skip GLEW source generation on Windows (using pre-built binaries)
echo Skipping GLEW source generation on Windows (using pre-built binaries)

echo.
echo Submodule initialization complete!
echo Note: PyTorch libtorch binaries will be downloaded by CMake during build.
