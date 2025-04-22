@echo off
echo Starting Weather Station Application...
echo.

:: Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found!
    echo Please run setup.bat first to set up the application.
    pause
    exit /b 1
)

:: Activate virtual environment and run the application
call venv\Scripts\activate
python app.py

:: Keep the window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred while running the application.
    pause
)