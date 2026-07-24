@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ======================================================================
::   Mamformer — 一鍵訓練腳本
::   自動化：資料處理 → 分詞 → 預訓練 → GRPO
:: ======================================================================

title Mamformer Training Pipeline

:: ── 路徑設定 ────────────────────────────────────────────────────────
set "PROJECT_DIR=%~dp0"
set "DATA_RAW=%PROJECT_DIR%data\raw"
set "DATA_PROCESSED=%PROJECT_DIR%data\processed"
set "DATA_TOKENIZED=%PROJECT_DIR%data\tokenized"
set "CHECKPOINT_DIR=%PROJECT_DIR%checkpoints"
set "GRPO_CHECKPOINT_DIR=%PROJECT_DIR%grpo_checkpoints"

:: ── 建立必要目錄 ───────────────────────────────────────────────────
if not exist "%DATA_RAW%" mkdir "%DATA_RAW%"
if not exist "%DATA_PROCESSED%" mkdir "%DATA_PROCESSED%"
if not exist "%DATA_TOKENIZED%" mkdir "%DATA_TOKENIZED%"
if not exist "%CHECKPOINT_DIR%" mkdir "%CHECKPOINT_DIR%"
if not exist "%GRPO_CHECKPOINT_DIR%" mkdir "%GRPO_CHECKPOINT_DIR%"

:: ── 檢測 GPU ───────────────────────────────────────────────────────
set GPU_COUNT=0
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    for /f "skip=1 tokens=1" %%a in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul ^| find /c /v ""') do set GPU_COUNT=%%a
)
if %GPU_COUNT% gtr 0 (
    echo [OK] 檢測到 %GPU_COUNT% 張 GPU
) else (
    echo [OK] 使用 CPU 模式
    set GPU_COUNT=0
)

:: ══════════════════════════════════════════════════════════════════════
::  主選單
:: ══════════════════════════════════════════════════════════════════════

:MAIN_MENU
cls
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║        Mamformer 一鍵訓練腳本 v0.2                     ║
echo  ╠══════════════════════════════════════════════════════════╣
echo  ║  GPU 數量: %GPU_COUNT% 張                                  ║
echo  ╠══════════════════════════════════════════════════════════╣
echo  ║  資料處理:                                              ║
echo  ║   [1] 自動清洗 + 分類（掃描 data\raw\ 所有檔案）       ║
echo  ║   [2] 分詞 → .bin（需先完成步驟 1）                    ║
echo  ║   [3] 完整資料管線（步驟 1 + 2，全部自動）             ║
echo  ╠══════════════════════════════════════════════════════════╣
echo  ║  預訓練（從頭開始）:                                     ║
echo  ║   [4] Debug 測試 (0.01B) — 1 GPU, 快速驗證            ║
echo  ║   [5] 7B 密集模型 — 1~8 GPU                            ║
echo  ║   [6] Ultra 7B (39B total / 7.5B active) — 8 GPU      ║
echo  ║   [7] Ultra 37B (200B / 37B) — 32 GPU                 ║
echo  ║   [8] Ultra 371B (371B / 28B) — 64 GPU                ║
echo  ║   [9] Ultra 671B MAX (671B / 37B) — 64~128 GPU        ║
echo  ╠══════════════════════════════════════════════════════════╣
echo  ║  後訓練:                                                ║
echo  ║   [G] GRPO 推理增強訓練                                 ║
echo  ╠══════════════════════════════════════════════════════════╣
echo  ║  其他:                                                  ║
echo  ║   [T] 執行全部測試                                      ║
echo  ║   [Q] 離開                                              ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
set /p CHOICE="  請選擇 [1-9/G/T/Q]: "

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
if /i "%CHOICE%"=="T" goto RUN_TESTS
if /i "%CHOICE%"=="Q" goto END
goto MAIN_MENU

:: ══════════════════════════════════════════════════════════════════════
::  資料處理
:: ══════════════════════════════════════════════════════════════════════

:DATA_CLEAN
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║  步驟 1: 自動清洗 + 分類                                ║
echo  ║  掃描 data\raw\ 目錄下所有檔案                          ║
echo  ║  支援: .txt .jsonl .csv .md .html .py .js .tex .pdf     ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
echo  開始處理...
python "%PROJECT_DIR%scripts\data_pipeline.py" ^
    --input "%DATA_RAW%" ^
    --output "%DATA_PROCESSED%"
if %errorlevel% neq 0 (
    echo [錯誤] 資料清洗失敗！
    pause
    goto MAIN_MENU
)
echo [完成] 資料清洗完成，結果在 data\processed\
pause
goto MAIN_MENU

:DATA_TOKENIZE
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║  步驟 2: 分詞 → .bin                                    ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
call :SELECT_MODEL_SIZE
echo.
echo  序列長度建議: 7B=8192, Ultra=8192~1048576
set /p SEQ_LEN="  序列長度 [8192]: "
if "%SEQ_LEN%"=="" set SEQ_LEN=8192
set /p NUM_SHARDS="  分片數量 [64]: "
if "%NUM_SHARDS%"=="" set NUM_SHARDS=64
echo.
echo  開始分詞...
echo  請確保 data\processed\ 目錄下有清洗後的檔案
echo  若無，請先執行步驟 1
echo.
pause

:: 尋找處理後的檔案
set "FOUND_INPUT="
if exist "%DATA_PROCESSED%\cleaned.jsonl" set "FOUND_INPUT=%DATA_PROCESSED%\cleaned.jsonl"
if exist "%DATA_PROCESSED%\*.jsonl" (
    for %%f in ("%DATA_PROCESSED%\*.jsonl") do set "FOUND_INPUT=%%f"
)
if exist "%DATA_PROCESSED%\*.txt" (
    for %%f in ("%DATA_PROCESSED%\*.txt") do set "FOUND_INPUT=%%f"
)

if "%FOUND_INPUT%"=="" (
    echo [警告] data\processed\ 目錄中沒有找到處理後的檔案
    echo         請先執行步驟 1，或手動放入 .jsonl/.txt 檔案
    set /p FOUND_INPUT="  請輸入檔案路徑: "
)

:: 判斷格式
echo !FOUND_INPUT! | findstr /i "\.jsonl$" >nul
if %errorlevel% equ 0 (
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
    echo [錯誤] 分詞失敗！
    pause
    goto MAIN_MENU
)
echo [完成] 分詞完成，二進制檔案在 data\tokenized\
pause
goto MAIN_MENU

:DATA_FULL
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║  完整資料管線（清洗 + 分詞）                            ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

:: Step 1: Clean
echo [1/2] 正在清洗資料...
python "%PROJECT_DIR%scripts\data_pipeline.py" ^
    --input "%DATA_RAW%" ^
    --output "%DATA_PROCESSED%"
if %errorlevel% neq 0 (
    echo [錯誤] 清洗失敗！
    pause
    goto MAIN_MENU
)

:: Step 2: Tokenize
echo.
echo [2/2] 正在分詞...
call :SELECT_MODEL_SIZE

:: Find processed files
set "FOUND_INPUT="
if exist "%DATA_PROCESSED%\*.jsonl" (
    for %%f in ("%DATA_PROCESSED%\*.jsonl") do set "FOUND_INPUT=%%f"
)
if exist "%DATA_PROCESSED%\*.txt" (
    for %%f in ("%DATA_PROCESSED%\*.txt") do set "FOUND_INPUT=%%f"
)
if "!FOUND_INPUT!"=="" (
    echo [錯誤] 處理後沒有找到輸出檔案
    pause
    goto MAIN_MENU
)

echo !FOUND_INPUT! | findstr /i "\.jsonl$" >nul
if %errorlevel% equ 0 (set INPUT_TYPE=jsonl) else (set INPUT_TYPE=txt)

python "%PROJECT_DIR%scripts\prepare_data.py" ^
    --input "!FOUND_INPUT!" ^
    --input_type !INPUT_TYPE! ^
    --output "%DATA_TOKENIZED%" ^
    --tokenizer huggyllama/llama-7b ^
    --seq_len %SEQ_LEN% ^
    --num_shards 64

if %errorlevel% neq 0 (
    echo [錯誤] 分詞失敗！
    pause
    goto MAIN_MENU
)
echo.
echo [完成] 資料管線完成！可開始訓練
pause
goto MAIN_MENU

:: ══════════════════════════════════════════════════════════════════════
::  訓練 — Debug
:: ══════════════════════════════════════════════════════════════════════

:TRAIN_DEBUG
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║  Debug 測試模型 (0.01B, 4-layer, d_model=256)          ║
echo  ║  快速驗證流程是否正常                                  ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
set "CONFIG=%PROJECT_DIR%configs\debug.yaml   ← 自動產生"
echo   Config: 內建 debug preset (不需要 YAML 檔案)
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

:: ══════════════════════════════════════════════════════════════════════
::  訓練 — 7B 密集
:: ══════════════════════════════════════════════════════════════════════

:TRAIN_7B
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║  Mamformer 7B 密集模型                                  ║
echo  ║  ~7B params, 8K context, 32 layers                      ║
echo  ╚══════════════════════════════════════════════════════════╝
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

:: ══════════════════════════════════════════════════════════════════════
::  訓練 — Ultra 7B
:: ══════════════════════════════════════════════════════════════════════

:TRAIN_ULTRA7B
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║  Mamformer Ultra 7B                                     ║
echo  ║  ~39B total / ~7.5B active, 8K context, MoE + DSA + MTP ║
echo  ╚══════════════════════════════════════════════════════════╝
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

:: ══════════════════════════════════════════════════════════════════════
::  訓練 — Ultra 37B
:: ══════════════════════════════════════════════════════════════════════

:TRAIN_ULTRA37B
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║  Mamformer Ultra 37B                                    ║
echo  ║  ~200B total / ~37B active, 128K context                ║
echo  ║  建議: 32 GPU (4 nodes x 8 GPU)                         ║
echo  ╚══════════════════════════════════════════════════════════╝
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
:: 4D 並行參數
set TP_SIZE=2
set PP_SIZE=2
set EP_SIZE=2
set DP_SIZE=4
goto TRAIN_DIST_START

:: ══════════════════════════════════════════════════════════════════════
::  訓練 — Ultra 371B
:: ══════════════════════════════════════════════════════════════════════

:TRAIN_ULTRA371B
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║  Mamformer Ultra 371B                                   ║
echo  ║  371B total / 28B active, 256K context                  ║
echo  ║  建議: 64 GPU (8 nodes x 8 GPU)                         ║
echo  ╚══════════════════════════════════════════════════════════╝
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

:: ══════════════════════════════════════════════════════════════════════
::  訓練 — Ultra 671B MAX
:: ══════════════════════════════════════════════════════════════════════

:TRAIN_ULTRA671B
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║  Mamformer Ultra 671B MAX                                ║
echo  ║  671B total / 37B active, 1M context, 52 layers         ║
echo  ║  建議: 64~128 GPU (8~16 nodes x 8 GPU)                  ║
echo  ║  需要 CPU offload + FSDP + gradient checkpointing        ║
echo  ╚══════════════════════════════════════════════════════════╝
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

:: ══════════════════════════════════════════════════════════════════════
::  訓練啟動（單機 / FSDP）
:: ══════════════════════════════════════════════════════════════════════

:TRAIN_START
echo.
echo  ── 訓練參數摘要 ─────────────────────────────────────────
echo    Config:           %CONFIG%
echo    Batch size:       %BATCH_SIZE%
echo    Gradient accum:   %GRAD_ACCUM%
echo    Max steps:        %MAX_STEPS%
echo    Learning rate:    %LR%
echo    Max seq len:      %MAX_SEQ_LEN%
echo    Warmup steps:     %WARMUP%
echo    Save every:       %SAVE_EVERY%
echo  ────────────────────────────────────────────────────────
echo.

:: CommunicativeMoE 選項
set /p USE_COMM="  啟用 CommunicativeMoE 跨專家通信? [y/N]: "
set COMM_FLAG=
if /i "!USE_COMM!"=="y" (
    set COMM_FLAG=--comm_moe
    echo   [OK] CommunicativeMoE 已啟用 (4-head cross-attention, depth=1)
)

:: WandB 選項
if "%WANDB_FLAG%"=="" (
    set /p USE_WANDB="  使用 WandB 記錄? [y/N]: "
    if /i "!USE_WANDB!"=="y" set WANDB_FLAG=--use_wandb
)

:: Resume 選項
set RESUME_FLAG=
set /p DO_RESUME="  從 checkpoint 繼續訓練? [y/N]: "
if /i "!DO_RESUME!"=="y" (
    echo   可用 checkpoints:
    if exist "%CHECKPOINT_DIR%\*.pt" (
        dir /b "%CHECKPOINT_DIR%\*.pt" 2>nul
    ) else (
        echo     (無)
    )
    set /p RESUME_PATH="  輸入 checkpoint 路徑: "
    if not "!RESUME_PATH!"=="" set RESUME_FLAG=--resume "!RESUME_PATH!"
)

echo.
echo  啟動訓練...

if %GPU_COUNT% leq 1 (
    :: 單卡
    echo   模式: 單 GPU
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
        --bf16 ^
        !WANDB_FLAG! ^
        !COMM_FLAG! ^
        !RESUME_FLAG!
) else (
    :: 多卡 FSDP
    echo   模式: %GPU_COUNT% GPU (FSDP)
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
        --bf16 ^
        !WANDB_FLAG! ^
        !COMM_FLAG! ^
        !RESUME_FLAG!
)

if %errorlevel% neq 0 (
    echo [錯誤] 訓練異常終止
    pause
    goto MAIN_MENU
)
echo [完成] 訓練結束！
pause
goto MAIN_MENU

:: ══════════════════════════════════════════════════════════════════════
::  訓練啟動（多機 4D 並行）
:: ══════════════════════════════════════════════════════════════════════

:TRAIN_DIST_START
echo.
echo  ── 4D 並行訓練參數摘要 ──────────────────────────────────
echo    Config:           %CONFIG%
echo    TP=%TP_SIZE%  PP=%PP_SIZE%  EP=%EP_SIZE%  DP=%DP_SIZE%
echo    Total GPUs:      %TP_SIZE% x %PP_SIZE% x %EP_SIZE% x %DP_SIZE% = !TOTAL_GPU!
set /a TOTAL_GPU = %TP_SIZE% * %PP_SIZE% * %EP_SIZE% * %DP_SIZE%
echo    Batch size:       %BATCH_SIZE%
echo    Gradient accum:   %GRAD_ACCUM%
echo    Max steps:        %MAX_STEPS%
echo    Learning rate:    %LR%
echo  ────────────────────────────────────────────────────────
echo.

if %GPU_COUNT% geq %TOTAL_GPU% (
    set NNODES=1
    set NPROC=%TOTAL_GPU%
) else (
    echo   可用 GPU: %GPU_COUNT%, 需要: %TOTAL_GPU%
    echo   將使用多節點模式
    set /p NNODES="  輸入節點數量: "
    set NPROC=%GPU_COUNT%
)

echo.
echo  啟動 4D 並行訓練...

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
    --bf16 ^
    %WANDB_FLAG%

if %errorlevel% neq 0 (
    echo [錯誤] 訓練異常終止
    pause
    goto MAIN_MENU
)
echo [完成] 訓練結束！
pause
goto MAIN_MENU

:: ══════════════════════════════════════════════════════════════════════
::  GRPO 推理增強訓練
:: ══════════════════════════════════════════════════════════════════════

:GRPO
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║  GRPO 推理增強訓練 (DeepSeek-R1 風格)                  ║
echo  ║  需要先完成 SFT checkpoint                             ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
echo  可用 checkpoints:
if exist "%CHECKPOINT_DIR%\*.pt" (
    dir /b "%CHECKPOINT_DIR%\*.pt" 2>nul
) else (
    echo    (無 — 請先完成預訓練)
)
echo.
set /p GRPO_CKPT="  SFT checkpoint 路徑: "
set /p GRPO_DATA="  GRPO prompt 資料檔 [data/grpo_prompts.jsonl]: "
if "%GRPO_DATA%"=="" set "GRPO_DATA=%PROJECT_DIR%data\grpo_prompts.jsonl"

call :SELECT_MODEL_SIZE
set /p GRPO_REWARD="  Reward 類型 [math/format/code/combined] [math]: "
if "%GRPO_REWARD%"=="" set GRPO_REWARD=math

set /p GRPO_G="  Group size G [8]: "
if "%GRPO_G%"=="" set GRPO_G=8

set /p GRPO_BETA="  KL penalty β [0.04]: "
if "%GRPO_BETA%"=="" set GRPO_BETA=0.04

set /p GRPO_MAX_STEPS="  Max steps [10000]: "
if "%GRPO_MAX_STEPS%"=="" set GRPO_MAX_STEPS=10000

echo.
echo  啟動 GRPO 訓練...

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
        --bf16 ^
        --max_prompt_len 2048 ^
        --gen_max_tokens 1024 ^
        --output_dir "%GRPO_CHECKPOINT_DIR%"
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
        --bf16 ^
        --max_prompt_len 2048 ^
        --gen_max_tokens 1024 ^
        --output_dir "%GRPO_CHECKPOINT_DIR%"
)

if %errorlevel% neq 0 (
    echo [錯誤] GRPO 訓練異常終止
    pause
    goto MAIN_MENU
)
echo [完成] GRPO 訓練結束！
pause
goto MAIN_MENU

:: ══════════════════════════════════════════════════════════════════════
::  執行測試
:: ══════════════════════════════════════════════════════════════════════

:RUN_TESTS
echo.
echo  執行全部測試...
python -m pytest "%PROJECT_DIR%tests\" -v --tb=short
echo.
pause
goto MAIN_MENU

:: ══════════════════════════════════════════════════════════════════════
::  輔助：選擇模型大小 → 設定 CONFIG 變數
:: ══════════════════════════════════════════════════════════════════════

:SELECT_MODEL_SIZE
echo.
echo   選擇模型大小:
echo     [1] 7B 密集      (7b.yaml)
echo     [2] Ultra 7B      (ultra-7b.yaml)
echo     [3] Ultra 37B     (ultra-37b.yaml)
echo     [4] Ultra 371B    (ultra-371b.yaml)
echo     [5] Ultra 671B    (ultra-671b-max.yaml)
echo     [6] Debug 測試
set /p MODEL_CHOICE="   請選擇 [1-6]: "

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

:: ══════════════════════════════════════════════════════════════════════

:END
echo.
echo  再見！
endlocal
exit /b 0
