@echo off
setlocal enabledelayedexpansion

:: Set output directory
set OUTPUT_DIR=visualizations
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

:: Load environment variables from .env file
echo Loading environment variables from .env file...
if not exist .env (
    echo Error: .env file not found
    echo Please create a .env file with GROQ_API_KEY=your-api-key-here
    pause
    exit /b 1
)

for /f "tokens=*" %%a in (.env) do (
    set %%a
)

:: Check if GROQ_API_KEY is set
if "%GROQ_API_KEY%"=="" (
    echo Error: GROQ_API_KEY not found in .env file
    echo Please add GROQ_API_KEY=your-api-key-here to your .env file
    pause
    exit /b 1
)

:: Check if MODEL_NAME is set
if "%MODEL_NAME%"=="" (
    echo Warning: MODEL_NAME not found in .env file
    echo Using default model: llama3-8b-8192
    set MODEL_NAME=llama3-8b-8192
)

:: Check if EMBEDDING_MODEL is set
if "%EMBEDDING_MODEL%"=="" (
    echo Warning: EMBEDDING_MODEL not found in .env file
    echo Using default embedding model: all-MiniLM-L6-v2
    set EMBEDDING_MODEL=all-MiniLM-L6-v2
)

:: Check Neo4j connection parameters
if "%NEO4J_URI%"=="" (
    echo Warning: NEO4J_URI not found in .env file
    echo Using default Neo4j URI: bolt://localhost:7687
    set NEO4J_URI=bolt://localhost:7687
)

if "%NEO4J_USER%"=="" (
    echo Warning: NEO4J_USER not found in .env file
    echo Using default Neo4j user: neo4j
    set NEO4J_USER=neo4j
)

if "%NEO4J_PASSWORD%"=="" (
    echo Warning: NEO4J_PASSWORD not found in .env file
    echo Using default Neo4j password: password
    set NEO4J_PASSWORD=password
)

:: Parse command line arguments
set SIMPLE_FLAG=
set HELP_FLAG=

:parse_args
if "%~1"=="" goto :end_parse_args
if /i "%~1"=="--simple" set SIMPLE_FLAG=--simple
if /i "%~1"=="-s" set SIMPLE_FLAG=--simple
if /i "%~1"=="--help" set HELP_FLAG=1
if /i "%~1"=="-h" set HELP_FLAG=1
shift
goto :parse_args
:end_parse_args

:: Display help if requested
if defined HELP_FLAG (
    echo.
    echo Usage: run_benchmark.bat [options]
    echo.
    echo Options:
    echo   --simple, -s    Run with simplified test scenarios for quick testing
    echo   --help, -h      Display this help message
    echo.
    exit /b 0
)

:: Run benchmarks for a single model
echo.
echo ===================================================
echo Starting benchmark for %MODEL_NAME%
echo ===================================================
echo Using embedding model: %EMBEDDING_MODEL%
echo Using Neo4j at: %NEO4J_URI%
echo Output directory: %OUTPUT_DIR%
if defined SIMPLE_FLAG echo Using simplified test scenarios for quick testing
echo.

:: Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not available in the PATH
    echo Please make sure Python is installed and added to your PATH
    pause
    exit /b 1
)

:: Check if required packages are installed
echo Checking required packages...
python -c "import pandas, plotly, numpy" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Required packages are missing
    echo Please install the required packages with:
    echo pip install pandas plotly numpy
    pause
    exit /b 1
)

:: Run the benchmark
python run_benchmarks.py --model %MODEL_NAME% --output-dir %OUTPUT_DIR% %SIMPLE_FLAG%
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: Benchmark failed with error code %ERRORLEVEL%
    echo Please check the error messages above
    pause
    exit /b %ERRORLEVEL%
)

:: Create index.html for easy navigation
echo.
echo Creating index.html...
python create_index.py --output-dir %OUTPUT_DIR%
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to create index.html
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===================================================
echo Benchmarking complete!
echo ===================================================
echo.
echo Results are available at: %OUTPUT_DIR%\index.html
echo Opening results in your default browser...
echo.

start "" "%OUTPUT_DIR%\index.html"

endlocal