@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

:: ======================================================================
::   Mamformer -- One-Click Training Script
::   Auto: Data Processing -> Tokenization -> Pretraining -> GRPO
::   Encoding: ASCII-safe (works on GBK/UTF-8/all codepages)
:: ======================================================================

title Mamformer Training Pipeline

:: -- Paths -----------------------------------------------------------
set "PROJECT_DIR=%~dp0"
set "DATA_RAW=%PROJECT_DIR%data\raw"
set "DATA_PROCESSED=%PROJECT_DIR%data\processed"
set "DATA_TOKENIZED=%PROJECT_DIR%data\tokenized"
set "CHECKPOINT_DIR=%PROJECT_DIR%checkpoints"
set "GRPO_CHECKPOINT_DIR=%PROJECT_DIR%grpo_checkpoints"

:: -- Create directories ---------------------------------------------
if not exist "%DATA_RAW%" mkdir "%DATA_RAW%"
if not exist "%DATA_PROCESSED%" mkdir "%DATA_PROCESSED%"
if not exist "%DATA_TOKENIZED%" mkdir "%DATA_TOKENIZED%"
if not exist "%CHECKPOINT_DIR%" mkdir "%CHECKPOINT_DIR%"
if not exist "%GRPO_CHECKPOINT_DIR%" mkdir "%GRPO_CHECKPOINT_DIR%"

:: -- Detect GPU ------------------------------------------------------
set GPU_COUNT=0
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    for /f "skip=1 tokens=1" %%a in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul ^| find /c /v ""') do set GPU_COUNT=%%a
)
if %GPU_COUNT% gtr 0 (
    echo [OK] Detected %GPU_COUNT% GPU(s)
) else (
    echo [OK] CPU mode
    set GPU_COUNT=0
)

:: BF16 only when GPU available
set BF16_FLAG=
if %GPU_COUNT% geq 1 set BF16_FLAG=--bf16

:: =====================================================================
::  MAIN MENU
:: =====================================================================

:MAIN_MENU
cls
echo(
echo  +============================================================+
echo  ^|        Mamformer One-Click Training v0.4                  ^|
echo  +============================================================+
echo  ^|  GPU Count: %GPU_COUNT%
echo  +------------------------------------------------------------+
echo  ^|  Data Processing:                                          ^|
echo  ^|   [1] Auto clean + classify (scan data\raw\)               ^|
echo  ^|   [2] Tokenize -^> .bin (need step 1 first)                ^|
echo  ^|   [3] Full data pipeline (step 1 + 2, auto)                ^|
echo  +------------------------------------------------------------+
echo  ^|  Verify / Debug:                                           ^|
echo  ^|   [0] Arch Verify (0.1B + all features) -- CPU/1GPU, ~30m  ^|
echo  ^|   [4] Debug test (0.01B, basic) -- quick pipeline check     ^|
echo  +------------------------------------------------------------+
echo  ^|  Pretraining (SFT):                                        ^|
echo  ^|   [5] 7B Dense -- 1~8 GPU                                  ^|
echo  ^|   [6] Ultra 7B (39B total / 7.5B active) -- 8 GPU          ^|
echo  ^|   [7] Ultra 37B (200B / 37B) -- 32 GPU                     ^|
echo  ^|   [8] Ultra 371B (371B / 28B) -- 64 GPU                    ^|
echo  ^|   [9] Ultra 671B MAX (671B / 37B) -- 64~128 GPU            ^|
echo  +------------------------------------------------------------+
echo  ^|  Post-training:                                            ^|
echo  ^|   [G] GRPO Reasoning RL (with S-GRPO option)               ^|
echo  +------------------------------------------------------------+
echo  ^|  Inference / Demo:                                         ^|
echo  ^|   [D] Generate with thinking mode                          ^|
echo  +------------------------------------------------------------+
echo  ^|  Other:                                                    ^|
echo  ^|   [T] Run all tests                                        ^|
echo  ^|   [Q] Quit                                                 ^|
echo  +------------------------------------------------------------+
echo(
set /p CHOICE="  Select [0-9/G/D/T/Q]: "

if "%CHOICE%"=="0" goto TRAIN_ARCH_CHECK
if /i "%CHOICE%"=="1" goto DATA_CLEAN
if /i "%CHOICE%"=="2" goto DATA_TOKENIZE
if /i "%CHOICE%"=="3" goto DATA_FULL
if "%CHOICE%"=="4" goto TRAIN_DEBUG
if "%CHOICE%"=="5" goto TRAIN_7B
if "%CHOICE%"=="6" goto TRAIN_ULTRA7B
if "%CHOICE%"=="7" goto TRAIN_ULTRA37B
if "%CHOICE%"=="8" goto TRAIN_ULTRA371B
if "%CHOICE%"=="9" goto TRAIN_ULTRA671B
if /i "%CHOICE%"=="G" goto GRPO
if /i "%CHOICE%"=="D" goto DEMO
if /i "%CHOICE%"=="T" goto RUN_TESTS
if /i "%CHOICE%"=="Q" goto END
goto MAIN_MENU

:: =====================================================================
::  DATA PROCESSING
:: =====================================================================

:DATA_CLEAN
echo(
echo  +------------------------------------------------------------+
echo  ^|  Step 1: Auto Clean + Classify                             ^|
echo  ^|  Scan all files in data\raw\                               ^|
echo  ^|  Supports: .txt .jsonl .csv .md .html .py .js .tex .pdf   ^|
echo  +------------------------------------------------------------+
echo(
echo  Processing...
python "%PROJECT_DIR%scripts\data_pipeline.py" ^
    --input "%DATA_RAW%" ^
    --output "%DATA_PROCESSED%"
if %errorlevel% neq 0 (
    echo [ERROR] Data cleaning failed!
    pause
    goto MAIN_MENU
)
echo [DONE] Cleaned data in data\processed\
pause
goto MAIN_MENU

:DATA_TOKENIZE
echo(
echo  +------------------------------------------------------------+
echo  ^|  Step 2: Tokenize -^> .bin                                  ^|
echo  +------------------------------------------------------------+
echo(
call :SELECT_MODEL_SIZE
echo(
echo  Suggested seq_len: 7B=8192, Ultra=8192~1048576
set /p SEQ_LEN="  Sequence length [8192]: "
if "%SEQ_LEN%"=="" set SEQ_LEN=8192
set /p NUM_SHARDS="  Shard count [64]: "
if "%NUM_SHARDS%"=="" set NUM_SHARDS=64
echo(
echo  Tokenizing...
echo  Make sure data\processed\ has cleaned files.
echo  If not, run step 1 first.
echo(
pause

:: Find processed files
set "FOUND_INPUT="
if exist "%DATA_PROCESSED%\cleaned.jsonl" set "FOUND_INPUT=%DATA_PROCESSED%\cleaned.jsonl"
if "%FOUND_INPUT%"=="" (
    for %%f in ("%DATA_PROCESSED%\*.jsonl") do set "FOUND_INPUT=%%f"
)
if "%FOUND_INPUT%"=="" (
    for %%f in ("%DATA_PROCESSED%\*.txt") do set "FOUND_INPUT=%%f"
)
if "%FOUND_INPUT%"=="" (
    echo [WARN] No processed files found in data\processed\
    echo        Run step 1 first, or manually add .jsonl/.txt files
    set /p FOUND_INPUT="  Enter file path: "
)

:: Detect format
echo !FOUND_INPUT! | findstr /i "\.jsonl$" >nul
if !errorlevel! equ 0 (
    set INPUT_TYPE=jsonl
) else (
    set INPUT_TYPE=txt
)

python "%PROJECT_DIR%scripts\prepare_data.py" ^
    --input "!FOUND_INPUT!" ^
    --input_type !INPUT_TYPE! ^
    --output "%DATA_TOKENIZED%" ^
    --tokenizer huggyllama/llama-7b ^
    --seq_len %SEQ_LEN% ^
    --num_shards %NUM_SHARDS%

if %errorlevel% neq 0 (
    echo [ERROR] Tokenization failed!
    pause
    goto MAIN_MENU
)
echo [DONE] Binary files in data\tokenized\
pause
goto MAIN_MENU

:DATA_FULL
echo(
echo  +------------------------------------------------------------+
echo  ^|  Full Pipeline (Clean + Tokenize)                          ^|
echo  +------------------------------------------------------------+
echo(

:: Step 1: Clean
echo [1/2] Cleaning data...
python "%PROJECT_DIR%scripts\data_pipeline.py" ^
    --input "%DATA_RAW%" ^
    --output "%DATA_PROCESSED%"
if %errorlevel% neq 0 (
    echo [ERROR] Cleaning failed!
    pause
    goto MAIN_MENU
)

:: Step 2: Tokenize
echo(
echo [2/2] Tokenizing...
call :SELECT_MODEL_SIZE

:: Find processed files
set "FOUND_INPUT="
for %%f in ("%DATA_PROCESSED%\*.jsonl") do set "FOUND_INPUT=%%f"
if "%FOUND_INPUT%"=="" (
    for %%f in ("%DATA_PROCESSED%\*.txt") do set "FOUND_INPUT=%%f"
)
if "!FOUND_INPUT!"=="" (
    echo [ERROR] No output files found after cleaning!
    pause
    goto MAIN_MENU
)

echo !FOUND_INPUT! | findstr /i "\.jsonl$" >nul
if !errorlevel! equ 0 (set INPUT_TYPE=jsonl) else (set INPUT_TYPE=txt)

python "%PROJECT_DIR%scripts\prepare_data.py" ^
    --input "!FOUND_INPUT!" ^
    --input_type !INPUT_TYPE! ^
    --output "%DATA_TOKENIZED%" ^
    --tokenizer huggyllama/llama-7b ^
    --seq_len %SEQ_LEN% ^
    --num_shards 64

if %errorlevel% neq 0 (
    echo [ERROR] Tokenization failed!
    pause
    goto MAIN_MENU
)
echo(
echo [DONE] Pipeline complete! Ready for training.
pause
goto MAIN_MENU

:: =====================================================================
::  TRAINING -- Architecture Verify
:: =====================================================================

:TRAIN_ARCH_CHECK
echo(
echo  +------------------------------------------------------------+
echo  ^|  Architecture Verify (0.1B, all Ultra features ON)         ^|
echo  ^|  MoE + KDA-Diff + MTP + Interleave -- full stack check     ^|
echo  +------------------------------------------------------------+
echo(
echo  This verifies the COMPLETE architecture at small scale:
echo    - DeepSeekMoE (2 shared + 16 routed, top-4)
echo    - KDA-Diff (kernelized differential attention)
echo    - MTP (multi-token prediction, depth=2)
echo    - Interleave cross_layer (attn every 4th layer)
echo    - Mamba-2 SSM with expand=1
echo(
echo  ^^! This runs ~300 steps on CPU (~30 min) or GPU (~2 min)
echo(

:: Generate arch-verify config
python "%PROJECT_DIR%scripts\gen_arch_config.py"
if %errorlevel% neq 0 (
    echo [ERROR] Config generation failed!
    pause
    goto MAIN_MENU
)

set "CONFIG=%PROJECT_DIR%configs\_arch_verify.yaml"
set BATCH_SIZE=1
set GRAD_ACCUM=4
set MAX_STEPS=300
set LR=8e-4
set MAX_SEQ_LEN=512
set SAVE_EVERY=100
set LOG_EVERY=10
set WARMUP=30
set WANDB_FLAG=
goto TRAIN_START

:: =====================================================================
::  TRAINING -- Debug
:: =====================================================================

:TRAIN_DEBUG
echo(
echo  +------------------------------------------------------------+
echo  ^|  Debug Model (0.01B, 4-layer, d_model=256)                ^|
echo  ^|  Quick pipeline verification                               ^|
echo  +------------------------------------------------------------+
echo(
echo  Config: built-in debug preset (no YAML needed)
set "CONFIG=debug"
set BATCH_SIZE=2
set GRAD_ACCUM=4
set MAX_STEPS=200
set LR=1e-3
set MAX_SEQ_LEN=128
set SAVE_EVERY=50
set LOG_EVERY=5
set WARMUP=20
set WANDB_FLAG=
goto TRAIN_START

:: =====================================================================
::  TRAINING -- 7B Dense
:: =====================================================================

:TRAIN_7B
echo(
echo  +------------------------------------------------------------+
echo  ^|  Mamformer 7B Dense                                        ^|
echo  ^|  ~7B params, 8K context, 32 layers                         ^|
echo  +------------------------------------------------------------+
set "CONFIG=%PROJECT_DIR%configs\7b.yaml"
set BATCH_SIZE=2
set GRAD_ACCUM=8
set MAX_STEPS=100000
set LR=3e-4
set MAX_SEQ_LEN=8192
set SAVE_EVERY=5000
set LOG_EVERY=10
set WARMUP=2000
set WANDB_FLAG=
goto TRAIN_START

:: =====================================================================
::  TRAINING -- Ultra 7B
:: =====================================================================

:TRAIN_ULTRA7B
echo(
echo  +------------------------------------------------------------+
echo  ^|  Mamformer Ultra 7B                                        ^|
echo  ^|  ~39B total / ~7.5B active, 8K context, MoE + DSA + MTP   ^|
echo  +------------------------------------------------------------+
set "CONFIG=%PROJECT_DIR%configs\ultra-7b.yaml"
set BATCH_SIZE=1
set GRAD_ACCUM=8
set MAX_STEPS=100000
set LR=3e-4
set MAX_SEQ_LEN=8192
set SAVE_EVERY=5000
set LOG_EVERY=10
set WARMUP=2000
set WANDB_FLAG=
goto TRAIN_START

:: =====================================================================
::  TRAINING -- Ultra 37B
:: =====================================================================

:TRAIN_ULTRA37B
echo(
echo  +------------------------------------------------------------+
echo  ^|  Mamformer Ultra 37B                                       ^|
echo  ^|  ~200B total / ~37B active, 128K context                   ^|
echo  ^|  Recommended: 32 GPU (4 nodes x 8 GPU)                     ^|
echo  +------------------------------------------------------------+
set "CONFIG=%PROJECT_DIR%configs\ultra-37b.yaml"
set BATCH_SIZE=1
set GRAD_ACCUM=16
set MAX_STEPS=200000
set LR=1.5e-4
set MAX_SEQ_LEN=8192
set SAVE_EVERY=5000
set LOG_EVERY=10
set WARMUP=4000
set WANDB_FLAG=--use_wandb
set TP_SIZE=2
set PP_SIZE=2
set EP_SIZE=2
set DP_SIZE=4
goto TRAIN_DIST_START

:: =====================================================================
::  TRAINING -- Ultra 371B
:: =====================================================================

:TRAIN_ULTRA371B
echo(
echo  +------------------------------------------------------------+
echo  ^|  Mamformer Ultra 371B                                      ^|
echo  ^|  371B total / 28B active, 256K context                     ^|
echo  ^|  Recommended: 64 GPU (8 nodes x 8 GPU)                     ^|
echo  +------------------------------------------------------------+
set "CONFIG=%PROJECT_DIR%configs\ultra-371b.yaml"
set BATCH_SIZE=1
set GRAD_ACCUM=16
set MAX_STEPS=200000
set LR=1.5e-4
set MAX_SEQ_LEN=8192
set SAVE_EVERY=5000
set LOG_EVERY=10
set WARMUP=4000
set WANDB_FLAG=--use_wandb
set TP_SIZE=4
set PP_SIZE=4
set EP_SIZE=2
set DP_SIZE=2
goto TRAIN_DIST_START

:: =====================================================================
::  TRAINING -- Ultra 671B MAX
:: =====================================================================

:TRAIN_ULTRA671B
echo(
echo  +------------------------------------------------------------+
echo  ^|  Mamformer Ultra 671B MAX                                  ^|
echo  ^|  671B total / 37B active, 1M context, 52 layers            ^|
echo  ^|  Recommended: 64~128 GPU (8~16 nodes x 8 GPU)             ^|
echo  ^|  Requires: CPU offload + FSDP + gradient checkpointing     ^|
echo  +------------------------------------------------------------+
set "CONFIG=%PROJECT_DIR%configs\ultra-671b-max.yaml"
set BATCH_SIZE=1
set GRAD_ACCUM=16
set MAX_STEPS=200000
set LR=1.5e-4
set MAX_SEQ_LEN=8192
set SAVE_EVERY=5000
set LOG_EVERY=10
set WARMUP=4000
set WANDB_FLAG=--use_wandb
set TP_SIZE=4
set PP_SIZE=4
set EP_SIZE=2
set DP_SIZE=2
goto TRAIN_DIST_START

:: =====================================================================
::  TRAINING LAUNCH (Single Node / FSDP)
:: =====================================================================

:TRAIN_START
echo(
echo  -- Training Parameters --------------------------------------
echo    Config:           %CONFIG%
echo    Batch size:       %BATCH_SIZE%
echo    Gradient accum:   %GRAD_ACCUM%
echo    Max steps:        %MAX_STEPS%
echo    Learning rate:    %LR%
echo    Max seq len:      %MAX_SEQ_LEN%
echo    Warmup steps:     %WARMUP%
echo    Save every:       %SAVE_EVERY%
echo  -------------------------------------------------------------
echo(

:: CommunicativeMoE option
set /p USE_COMM="  Enable CommunicativeMoE? [y/N]: "
set COMM_FLAG=
if /i "!USE_COMM!"=="y" (
    set COMM_FLAG=--comm_moe
    echo   [OK] CommunicativeMoE enabled
)

:: Thinking format option (SFT with reasoning tokens)
set /p USE_THINK="  Train with thinking tokens? [y/N]: "
set THINK_FLAG=
if /i "!USE_THINK!"=="y" (
    set THINK_FLAG=--thinking_format
    echo   [OK] Thinking format enabled
)

:: WandB option
if "%WANDB_FLAG%"=="" (
    set /p USE_WANDB="  Use WandB logging? [y/N]: "
    if /i "!USE_WANDB!"=="y" set WANDB_FLAG=--use_wandb
)

:: Resume option
set RESUME_FLAG=
set /p DO_RESUME="  Resume from checkpoint? [y/N]: "
if /i "!DO_RESUME!"=="y" (
    echo   Available checkpoints:
    if exist "%CHECKPOINT_DIR%\*.pt" (
        dir /b "%CHECKPOINT_DIR%\*.pt" 2>nul
    ) else (
        echo     (none)
    )
    set /p RESUME_PATH="  Enter checkpoint path: "
    if not "!RESUME_PATH!"=="" set RESUME_FLAG=--resume "!RESUME_PATH!"
)

echo(
echo  Launching training...

if %GPU_COUNT% leq 1 (
    :: Single GPU
    echo   Mode: CPU
    python "%PROJECT_DIR%scripts\train.py" ^
        --config "%CONFIG%" ^
        --data "%DATA_TOKENIZED%" ^
        --batch_size %BATCH_SIZE% ^
        --gradient_accumulation_steps %GRAD_ACCUM% ^
        --max_steps %MAX_STEPS% ^
        --learning_rate %LR% ^
        --max_seq_len %MAX_SEQ_LEN% ^
        --warmup_steps %WARMUP% ^
        --save_every %SAVE_EVERY% ^
        --log_every %LOG_EVERY% ^
        --output_dir "%CHECKPOINT_DIR%" ^
        !BF16_FLAG! ^
        !WANDB_FLAG! ^
        !COMM_FLAG! ^
        !THINK_FLAG! ^
        !RESUME_FLAG!
) else (
    :: Multi-GPU FSDP
    echo   Mode: %GPU_COUNT% GPU (FSDP)
    torchrun --nproc_per_node=%GPU_COUNT% "%PROJECT_DIR%scripts\train.py" ^
        --config "%CONFIG%" ^
        --data "%DATA_TOKENIZED%" ^
        --batch_size %BATCH_SIZE% ^
        --gradient_accumulation_steps %GRAD_ACCUM% ^
        --max_steps %MAX_STEPS% ^
        --learning_rate %LR% ^
        --max_seq_len %MAX_SEQ_LEN% ^
        --warmup_steps %WARMUP% ^
        --save_every %SAVE_EVERY% ^
        --log_every %LOG_EVERY% ^
        --output_dir "%CHECKPOINT_DIR%" ^
        !BF16_FLAG! ^
        !WANDB_FLAG! ^
        !COMM_FLAG! ^
        !THINK_FLAG! ^
        !RESUME_FLAG!
)

if %errorlevel% neq 0 (
    echo [ERROR] Training terminated abnormally
    pause
    goto MAIN_MENU
)
echo [DONE] Training complete!
pause
goto MAIN_MENU

:: =====================================================================
::  TRAINING LAUNCH (Multi-Node 4D Parallel)
:: =====================================================================

:TRAIN_DIST_START
set /a TOTAL_GPU = %TP_SIZE% * %PP_SIZE% * %EP_SIZE% * %DP_SIZE%
echo(
echo  -- 4D Parallel Training Parameters --------------------------
echo    Config:           %CONFIG%
echo    TP=%TP_SIZE%  PP=%PP_SIZE%  EP=%EP_SIZE%  DP=%DP_SIZE%
echo    Total GPUs:       %TOTAL_GPU%
echo    Batch size:       %BATCH_SIZE%
echo    Gradient accum:   %GRAD_ACCUM%
echo    Max steps:        %MAX_STEPS%
echo    Learning rate:    %LR%
echo  -------------------------------------------------------------
echo(

if %GPU_COUNT% geq %TOTAL_GPU% (
    set NNODES=1
    set NPROC=%TOTAL_GPU%
) else (
    echo   Available GPU: %GPU_COUNT%, Required: %TOTAL_GPU%
    echo   Using multi-node mode
    set /p NNODES="  Enter node count: "
    set NPROC=%GPU_COUNT%
)

echo(
echo  Launching 4D parallel training...

torchrun --nnodes=!NNODES! --nproc_per_node=!NPROC! "%PROJECT_DIR%scripts\train_distributed.py" ^
    --config "%CONFIG%" ^
    --data "%DATA_TOKENIZED%" ^
    --tp %TP_SIZE% --pp %PP_SIZE% --ep %EP_SIZE% --dp %DP_SIZE% ^
    --batch_size %BATCH_SIZE% ^
    --gradient_accumulation_steps %GRAD_ACCUM% ^
    --max_steps %MAX_STEPS% ^
    --learning_rate %LR% ^
    --max_seq_len %MAX_SEQ_LEN% ^
    --warmup_steps %WARMUP% ^
    --save_every %SAVE_EVERY% ^
    --log_every %LOG_EVERY% ^
    --output_dir "%CHECKPOINT_DIR%" ^
    !BF16_FLAG! ^
    %WANDB_FLAG%

if %errorlevel% neq 0 (
    echo [ERROR] Training terminated abnormally
    pause
    goto MAIN_MENU
)
echo [DONE] Training complete!
pause
goto MAIN_MENU

:: =====================================================================
::  GRPO Training
:: =====================================================================

:GRPO
echo(
echo  +------------------------------------------------------------+
echo  ^|  GRPO Reasoning RL (+ S-GRPO option)                       ^|
echo  ^|  Requires SFT checkpoint first                             ^|
echo  +------------------------------------------------------------+
echo(
echo  Available checkpoints:
if exist "%CHECKPOINT_DIR%\*.pt" (
    dir /b "%CHECKPOINT_DIR%\*.pt" 2>nul
) else (
    echo    (none - complete pretraining first)
)
echo(
set /p GRPO_CKPT="  SFT checkpoint path: "
set /p GRPO_DATA="  GRPO prompt data [data\grpo_prompts.jsonl]: "
if "%GRPO_DATA%"=="" set "GRPO_DATA=%PROJECT_DIR%data\grpo_prompts.jsonl"

call :SELECT_MODEL_SIZE
set /p GRPO_REWARD="  Reward type [math/format/code/combined] [math]: "
if "%GRPO_REWARD%"=="" set GRPO_REWARD=math

set /p GRPO_G="  Group size G [8]: "
if "%GRPO_G%"=="" set GRPO_G=8

set /p GRPO_BETA="  KL penalty beta [0.04]: "
if "%GRPO_BETA%"=="" set GRPO_BETA=0.04

set /p GRPO_MAX_STEPS="  Max steps [10000]: "
if "%GRPO_MAX_STEPS%"=="" set GRPO_MAX_STEPS=10000

:: S-GRPO options
set /p USE_SGRPO="  Enable S-GRPO (sparse token sampling)? [y/N]: "
set SGRPO_FLAG=
set SGRPO_P=0.4
if /i "!USE_SGRPO!"=="y" (
    set SGRPO_FLAG=--sgrpo
    set /p SGRPO_P="    Token sampling probability [0.4]: "
    if "!SGRPO_P!"=="" set SGRPO_P=0.4
    set /p SGRPO_ALPHA="    First N tokens always kept [0]: "
    if "!SGRPO_ALPHA!"=="" set SGRPO_ALPHA=0
    set /p SGRPO_K="    Max tokens cap (0=disabled) [0]: "
    if "!SGRPO_K!"=="" set SGRPO_K=0
    echo   [OK] S-GRPO enabled (p=!SGRPO_P!, alpha=!SGRPO_ALPHA!, k=!SGRPO_K!)
)

:: Thinking reward option
set /p USE_THINK_R="  Include thinking quality reward? [y/N]: "
set THINK_R_FLAG=
if /i "!USE_THINK_R!"=="y" (
    set THINK_R_FLAG=--reward_type combined
    echo   [OK] Thinking quality reward added
)

echo(
echo  Launching GRPO training...

if %GPU_COUNT% leq 1 (
    python "%PROJECT_DIR%scripts\train_grpo.py" ^
        --config "!CONFIG!" ^
        --checkpoint "%GRPO_CKPT%" ^
        --data "%GRPO_DATA%" ^
        --reward_type %GRPO_REWARD% ^
        --group_size %GRPO_G% ^
        --kl_beta %GRPO_BETA% ^
        --max_steps %GRPO_MAX_STEPS% ^
        --batch_size 4 ^
        --gradient_accumulation_steps 2 ^
        --learning_rate 1e-6 ^
        !BF16_FLAG! ^
        --max_prompt_len 2048 ^
        --gen_max_tokens 1024 ^
        --output_dir "%GRPO_CHECKPOINT_DIR%" ^
        !SGRPO_FLAG! --sgrpo_p !SGRPO_P! --sgrpo_alpha !SGRPO_ALPHA! --sgrpo_k !SGRPO_K!
) else (
    torchrun --nproc_per_node=%GPU_COUNT% "%PROJECT_DIR%scripts\train_grpo.py" ^
        --config "!CONFIG!" ^
        --checkpoint "%GRPO_CKPT%" ^
        --data "%GRPO_DATA%" ^
        --reward_type %GRPO_REWARD% ^
        --group_size %GRPO_G% ^
        --kl_beta %GRPO_BETA% ^
        --max_steps %GRPO_MAX_STEPS% ^
        --batch_size 4 ^
        --gradient_accumulation_steps 2 ^
        --learning_rate 1e-6 ^
        !BF16_FLAG! ^
        --max_prompt_len 2048 ^
        --gen_max_tokens 1024 ^
        --output_dir "%GRPO_CHECKPOINT_DIR%" ^
        !SGRPO_FLAG! --sgrpo_p !SGRPO_P! --sgrpo_alpha !SGRPO_ALPHA! --sgrpo_k !SGRPO_K!
)

if %errorlevel% neq 0 (
    echo [ERROR] GRPO training terminated abnormally
    pause
    goto MAIN_MENU
)
echo [DONE] GRPO training complete!
pause
goto MAIN_MENU

:: =====================================================================
::  DEMO: Generate with Thinking Mode
:: =====================================================================

:DEMO
echo(
echo  +------------------------------------------------------------+
echo  ^|  Generate with Thinking Mode                               ^|
echo  +------------------------------------------------------------+
echo(
echo  Available checkpoints:
if exist "%CHECKPOINT_DIR%\*.pt" (
    dir /b "%CHECKPOINT_DIR%\*.pt" 2>nul
) else (
    echo    (none - using random init)
)
echo(
set /p DEMO_CKPT="  Checkpoint path (or blank for random): "
set /p DEMO_PROMPT="  Prompt: "

call :SELECT_MODEL_SIZE
call :SELECT_THINKING_MODE

set /p DEMO_TOKENS="  Max new tokens [256]: "
if "%DEMO_TOKENS%"=="" set DEMO_TOKENS=256

:: Build thinking args
set THINK_ARGS=
if not "%THINK_MODE%"=="NoThink" (
    set THINK_ARGS=--thinking %THINK_MODE% --think_budget %THINK_BUDGET% --num_paths %THINK_PATHS% --summary_budget %THINK_SUMMARY%
    set /p SHOW_THINK="  Show thinking tokens? [y/N]: "
    if /i "!SHOW_THINK!"=="y" set THINK_ARGS=!THINK_ARGS! --show_thinking
)

set CKPT_ARG=
if not "%DEMO_CKPT%"=="" set CKPT_ARG=--checkpoint "%DEMO_CKPT%"

echo(
echo  Generating...
python "%PROJECT_DIR%scripts\generate.py" ^
    --config "!CONFIG!" ^
    --prompt "!DEMO_PROMPT!" ^
    !CKPT_ARG! ^
    --max_new_tokens %DEMO_TOKENS% ^
    --temperature 0.7 ^
    !THINK_ARGS!

echo(
pause
goto MAIN_MENU

:: =====================================================================
::  RUN TESTS
:: =====================================================================

:RUN_TESTS
echo(
echo  Running all tests...
python -m pytest "%PROJECT_DIR%tests" -v --tb=short
echo(
pause
goto MAIN_MENU

:: =====================================================================
::  HELPER: Select Model Size
:: =====================================================================

:SELECT_MODEL_SIZE
echo(
echo   Select model size:
echo     [1] 7B Dense       (7b.yaml)
echo     [2] Ultra 7B       (ultra-7b.yaml)
echo     [3] Ultra 37B      (ultra-37b.yaml)
echo     [4] Ultra 371B     (ultra-371b.yaml)
echo     [5] Ultra 671B     (ultra-671b-max.yaml)
echo     [6] Debug Test
set /p MODEL_CHOICE="   Select [1-6]: "

if "%MODEL_CHOICE%"=="1" set "CONFIG=%PROJECT_DIR%configs\7b.yaml"
if "%MODEL_CHOICE%"=="2" set "CONFIG=%PROJECT_DIR%configs\ultra-7b.yaml"
if "%MODEL_CHOICE%"=="3" set "CONFIG=%PROJECT_DIR%configs\ultra-37b.yaml"
if "%MODEL_CHOICE%"=="4" set "CONFIG=%PROJECT_DIR%configs\ultra-371b.yaml"
if "%MODEL_CHOICE%"=="5" set "CONFIG=%PROJECT_DIR%configs\ultra-671b-max.yaml"
if "%MODEL_CHOICE%"=="6" set "CONFIG=debug"

if "%CONFIG%"=="debug" (
    set SEQ_LEN=128
) else if "%MODEL_CHOICE%"=="1" (
    set SEQ_LEN=8192
) else if "%MODEL_CHOICE%"=="2" (
    set SEQ_LEN=8192
) else if "%MODEL_CHOICE%"=="3" (
    set SEQ_LEN=8192
) else if "%MODEL_CHOICE%"=="4" (
    set SEQ_LEN=8192
) else if "%MODEL_CHOICE%"=="5" (
    set SEQ_LEN=8192
)
goto :EOF

:: =====================================================================
::  HELPER: Select Thinking Mode
:: =====================================================================

:SELECT_THINKING_MODE
echo(
echo   Thinking mode:
echo     [0] NoThink   - standard generation (no thinking)
echo     [1] FastThink - 2 paths, 128 tokens each, 64 summary
echo     [2] CoreThink - 3 paths, 341 tokens each, 128 summary
echo     [3] DeepThink - 5 paths, 819 tokens each, 256 summary
echo     [4] Custom
set /p THINK_CHOICE="   Select [0-4]: "

if "%THINK_CHOICE%"=="0" (
    set THINK_MODE=NoThink
    set THINK_BUDGET=0
    set THINK_PATHS=0
    set THINK_SUMMARY=0
) else if "%THINK_CHOICE%"=="1" (
    set THINK_MODE=FastThink
    set THINK_BUDGET=128
    set THINK_PATHS=2
    set THINK_SUMMARY=64
) else if "%THINK_CHOICE%"=="2" (
    set THINK_MODE=CoreThink
    set THINK_BUDGET=341
    set THINK_PATHS=3
    set THINK_SUMMARY=128
) else if "%THINK_CHOICE%"=="3" (
    set THINK_MODE=DeepThink
    set THINK_BUDGET=819
    set THINK_PATHS=5
    set THINK_SUMMARY=256
) else if "%THINK_CHOICE%"=="4" (
    set /p THINK_MODE="   Mode name [CoreThink]: "
    if "!THINK_MODE!"=="" set THINK_MODE=CoreThink
    set /p THINK_BUDGET="   Per-path budget [341]: "
    if "!THINK_BUDGET!"=="" set THINK_BUDGET=341
    set /p THINK_PATHS="   Number of paths [3]: "
    if "!THINK_PATHS!"=="" set THINK_PATHS=3
    set /p THINK_SUMMARY="   Summary budget [128]: "
    if "!THINK_SUMMARY!"=="" set THINK_SUMMARY=128
) else (
    set THINK_MODE=NoThink
    set THINK_BUDGET=0
    set THINK_PATHS=0
    set THINK_SUMMARY=0
)
goto :EOF

:: =====================================================================

:END
echo(
echo  Goodbye!
endlocal
exit /b 0
