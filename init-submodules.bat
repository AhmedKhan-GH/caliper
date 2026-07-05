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
REM
REM Mirror of init-submodules.sh: each submodule updates individually because
REM a single `git submodule update --init --recursive --force` aborts at the
REM first submodule whose pinned commit is missing from its remote, leaving
REM everything after it un-initialized. Some pins in this repo reference
REM commits that were never pushed upstream (locally-generated GLEW sources,
REM a llama.cpp fork commit) — tolerate those and fall back to the remote's
REM default branch rather than failing the whole setup.
echo Initializing submodules...
for /f "tokens=2" %%P in ('git config -f .gitmodules --get-regexp "path$"') do call :init_one "%%P"

REM Skip GLEW source generation on Windows (using pre-built binaries)
echo Skipping GLEW source generation on Windows (using pre-built binaries)

echo.
echo Submodule initialization complete!
echo Note: PyTorch libtorch binaries will be downloaded by CMake during build.
goto :eof

:init_one
set "SM_PATH=%~1"
git submodule update --init --recursive --force "%SM_PATH%" >nul 2>&1
if not errorlevel 1 (
    echo   [OK] %SM_PATH%
    exit /b 0
)
for /f "delims=" %%U in ('git config -f .gitmodules --get "submodule.%SM_PATH%.url"') do set "SM_URL=%%U"
echo   [!] %SM_PATH%: pinned commit unavailable from %SM_URL%
git -C "%SM_PATH%" rev-parse HEAD >nul 2>&1
if not errorlevel 1 (
    REM update already cloned the default branch before the checkout failed
    echo       using default branch instead
    exit /b 0
)
echo       cloning default branch from %SM_URL%
if exist "%SM_PATH%" rmdir /s /q "%SM_PATH%"
git clone --recursive "%SM_URL%" "%SM_PATH%" >nul 2>&1
if not errorlevel 1 (
    echo       [OK] cloned %SM_PATH%
) else (
    echo       [X] failed to clone %SM_PATH% - build may not work
)
exit /b 0
