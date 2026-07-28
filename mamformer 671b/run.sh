#!/bin/bash
# ======================================================================
#   Mamformer -- One-Click Training Script v0.5 (Linux)
# ======================================================================

set -e

# -- Colors -----------------------------------------------------------
GREEN='\033[92m'
RED='\033[91m'
YELLOW='\033[93m'
CYAN='\033[96m'
RESET='\033[0m'

# -- Paths -----------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_RAW="$PROJECT_DIR/data/raw"
DATA_PROCESSED="$PROJECT_DIR/data/processed"
DATA_TOKENIZED="$PROJECT_DIR/data/tokenized"
CHECKPOINT_DIR="/workspace/checkpoints"
GRPO_CHECKPOINT_DIR="$PROJECT_DIR/grpo_checkpoints"

# -- Create directories ---------------------------------------------
mkdir -p "$DATA_RAW" "$DATA_PROCESSED" "$DATA_TOKENIZED" "$CHECKPOINT_DIR" "$GRPO_CHECKPOINT_DIR"

# -- Detect GPU ------------------------------------------------------
GPU_COUNT=0
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
fi
if [ "$GPU_COUNT" -gt 0 ]; then
    echo -e "[OK] Detected $GPU_COUNT GPU(s)"
    BF16_FLAG="--bf16"
else
    echo -e "[OK] CPU mode"
    BF16_FLAG=""
fi

# =====================================================================
#  HELPER FUNCTIONS
# =====================================================================

select_model_size() {
    echo
    echo "  Select model size:"
    echo "    [1] 7B Dense       (7b.yaml)"
    echo "    [2] Ultra 7B       (ultra-7b.yaml)"
    echo "    [3] Ultra 37B      (ultra-37b.yaml)"
    echo "    [4] Ultra 371B     (ultra-371b.yaml)"
    echo "    [5] Ultra 671B     (ultra-671b-max.yaml)"
    echo "    [6] Debug Test"
    read -p "  Select [1-6]: " MODEL_CHOICE

    case "$MODEL_CHOICE" in
        1) CONFIG="$PROJECT_DIR/configs/7b.yaml"; SEQ_LEN=8192 ;;
        2) CONFIG="$PROJECT_DIR/configs/ultra-7b.yaml"; SEQ_LEN=8192 ;;
        3) CONFIG="$PROJECT_DIR/configs/ultra-37b.yaml"; SEQ_LEN=8192 ;;
        4) CONFIG="$PROJECT_DIR/configs/ultra-371b.yaml"; SEQ_LEN=8192 ;;
        5) CONFIG="$PROJECT_DIR/configs/ultra-671b-max.yaml"; SEQ_LEN=8192 ;;
        6) CONFIG="debug"; SEQ_LEN=128 ;;
        *) CONFIG="debug"; SEQ_LEN=128 ;;
    esac
}

select_thinking_mode() {
    echo
    echo "  Thinking mode:"
    echo "    [0] NoThink   - standard generation (no thinking)"
    echo "    [1] FastThink - 2 paths, 128 tokens each, 64 summary"
    echo "    [2] CoreThink - 3 paths, 341 tokens each, 128 summary"
    echo "    [3] DeepThink - 5 paths, 819 tokens each, 256 summary"
    echo "    [4] Custom"
    read -p "  Select [0-4]: " THINK_CHOICE

    case "$THINK_CHOICE" in
        0) THINK_MODE="NoThink"; THINK_BUDGET=0; THINK_PATHS=0; THINK_SUMMARY=0 ;;
        1) THINK_MODE="FastThink"; THINK_BUDGET=128; THINK_PATHS=2; THINK_SUMMARY=64 ;;
        2) THINK_MODE="CoreThink"; THINK_BUDGET=341; THINK_PATHS=3; THINK_SUMMARY=128 ;;
        3) THINK_MODE="DeepThink"; THINK_BUDGET=819; THINK_PATHS=5; THINK_SUMMARY=256 ;;
        4)
            read -p "  Mode name [CoreThink]: " THINK_MODE
            THINK_MODE="${THINK_MODE:-CoreThink}"
            read -p "  Per-path budget [341]: " THINK_BUDGET
            THINK_BUDGET="${THINK_BUDGET:-341}"
            read -p "  Number of paths [3]: " THINK_PATHS
            THINK_PATHS="${THINK_PATHS:-3}"
            read -p "  Summary budget [128]: " THINK_SUMMARY
            THINK_SUMMARY="${THINK_SUMMARY:-128}"
            ;;
        *) THINK_MODE="NoThink"; THINK_BUDGET=0; THINK_PATHS=0; THINK_SUMMARY=0 ;;
    esac
}

# =====================================================================
#  DATA DOWNLOAD
# =====================================================================

data_download() {
    local PRESET="$1"
    local DESC="$2"
    echo
    echo "  +------------------------------------------------------------+"
    echo "  |  Download: $DESC"
    echo "  +------------------------------------------------------------+"
    echo
    echo "  Tokenizer: huggyllama/llama-7b"
    echo "  Seq len: 8192"
    echo "  Output: data/"
    echo
    echo "  Requires: pip install datasets transformers tqdm"
    echo
    read -p "  Start download? [Y/n]: " CONFIRM
    if [ "$CONFIRM" = "n" ] || [ "$CONFIRM" = "N" ]; then
        return
    fi

    echo
    echo "  Downloading + Tokenizing... (this may take hours for large presets)"
    echo
    python "$PROJECT_DIR/scripts/download_data.py" --preset "$PRESET" --output "$PROJECT_DIR/data" --tokenizer huggyllama/llama-7b --seq_len 8192
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR]${RESET} Download failed! Check internet and disk space."
    else
        echo
        echo -e "${GREEN}[DONE]${RESET} Data downloaded and tokenized to data/tokenized/"
        echo "  Ready for training."
    fi
    read -p "Press Enter to continue..."
}

# =====================================================================
#  TRAINING LAUNCH
# =====================================================================

train_launch() {
    echo
    echo "  -- Training Parameters --------------------------------------"
    echo "    Config:           $CONFIG"
    echo "    Batch size:       $BATCH_SIZE"
    echo "    Gradient accum:   $GRAD_ACCUM"
    echo "    Max steps:        $MAX_STEPS"
    echo "    Learning rate:    $LR"
    echo "    Max seq len:      $MAX_SEQ_LEN"
    echo "    Warmup steps:     $WARMUP"
    echo "    Save every:       $SAVE_EVERY"
    echo "  -------------------------------------------------------------"
    echo

    read -p "  Enable CommunicativeMoE? [y/N]: " USE_COMM
    COMM_FLAG=""
    if [ "$USE_COMM" = "y" ] || [ "$USE_COMM" = "Y" ]; then
        COMM_FLAG="--comm_moe"
        echo "    [OK] CommunicativeMoE enabled"
    fi

    read -p "  Train with thinking tokens? [y/N]: " USE_THINK
    THINK_FLAG=""
    if [ "$USE_THINK" = "y" ] || [ "$USE_THINK" = "Y" ]; then
        THINK_FLAG="--thinking_format"
        echo "    [OK] Thinking format enabled"
    fi

    if [ -z "$WANDB_FLAG" ]; then
        read -p "  Use WandB logging? [y/N]: " USE_WANDB
        if [ "$USE_WANDB" = "y" ] || [ "$USE_WANDB" = "Y" ]; then
            WANDB_FLAG="--use_wandb"
        fi
    fi

    read -p "  Resume from checkpoint? [y/N]: " DO_RESUME
    RESUME_FLAG=""
    if [ "$DO_RESUME" = "y" ] || [ "$DO_RESUME" = "Y" ]; then
        echo "  Available checkpoints:"
        if ls "$CHECKPOINT_DIR"/*.pt 2>/dev/null; then
            :
        else
            echo "    (none)"
        fi
        read -p "  Enter checkpoint path: " RESUME_PATH
        if [ -n "$RESUME_PATH" ]; then
            RESUME_FLAG="--resume $RESUME_PATH"
        fi
    fi

    echo
    echo "  Launching training..."

    if [ "$GPU_COUNT" -le 1 ]; then
        if [ "$GPU_COUNT" -eq 0 ]; then
            echo "  Mode: CPU"
        else
            echo "  Mode: Single GPU"
        fi
        python "$PROJECT_DIR/scripts/train.py" \
            --config "$CONFIG" \
            --data "$DATA_TOKENIZED" \
            --batch_size "$BATCH_SIZE" \
            --gradient_accumulation_steps "$GRAD_ACCUM" \
            --max_steps "$MAX_STEPS" \
            --learning_rate "$LR" \
            --max_seq_len "$MAX_SEQ_LEN" \
            --warmup_steps "$WARMUP" \
            --save_every "$SAVE_EVERY" \
            --log_every "$LOG_EVERY" \
            --output_dir "$CHECKPOINT_DIR" \
            $BF16_FLAG $WANDB_FLAG $COMM_FLAG $THINK_FLAG $RESUME_FLAG
    else
        echo "  Mode: $GPU_COUNT GPU (FSDP)"
        torchrun --nproc_per_node="$GPU_COUNT" "$PROJECT_DIR/scripts/train.py" \
            --config "$CONFIG" \
            --data "$DATA_TOKENIZED" \
            --batch_size "$BATCH_SIZE" \
            --gradient_accumulation_steps "$GRAD_ACCUM" \
            --max_steps "$MAX_STEPS" \
            --learning_rate "$LR" \
            --max_seq_len "$MAX_SEQ_LEN" \
            --warmup_steps "$WARMUP" \
            --save_every "$SAVE_EVERY" \
            --log_every "$LOG_EVERY" \
            --output_dir "$CHECKPOINT_DIR" \
            $BF16_FLAG $WANDB_FLAG $COMM_FLAG $THINK_FLAG $RESUME_FLAG
    fi

    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR]${RESET} Training terminated abnormally"
    else
        echo -e "${GREEN}[DONE]${RESET} Training complete!"
    fi
    read -p "Press Enter to continue..."
}

# =====================================================================
#  MAIN MENU
# =====================================================================

while true; do
    clear
    echo
    echo "  +============================================================+"
    echo "  |        Mamformer One-Click Training v0.5 (Linux)          |"
    echo "  +============================================================+"
    echo "  |  GPU Count: $GPU_COUNT"
    echo "  +------------------------------------------------------------+"
    echo "  |  Download Data (cloud GPU, auto tokenize):                |"
    echo "  |   [A] Debug (~1B tok) -> debug + arch-verify              |"
    echo "  |   [B] 1B (~20B tok) -> 1b preset                          |"
    echo "  |   [C] 7B (~140B tok) -> 7b.yaml                             |"
    echo "  |   [D] Ultra 7B (~140B tok) -> ultra-7b.yaml               |"
    echo "  |   [E] Ultra 37B (~740B tok) -> ultra-37b.yaml             |"
    echo "  |   [F] Ultra 371B (~560B tok) -> ultra-371b.yaml           |"
    echo "  |   [H] Ultra 671B (~1T tok) -> ultra-671b-max.yaml         |"
    echo "  +------------------------------------------------------------+"
    echo "  |  Data Processing (local files):                           |"
    echo "  |   [1] Auto clean + classify (scan data/raw/)              |"
    echo "  |   [2] Tokenize -> .bin (need step 1 first)                |"
    echo "  |   [3] Full data pipeline (step 1 + 2, auto)               |"
    echo "  +------------------------------------------------------------+"
    echo "  |  Verify / Debug:                                          |"
    echo "  |   [0] Arch Verify (0.1B + all features) -- CPU/1GPU, ~30m |"
    echo "  |   [4] Debug test (0.01B, basic) -- quick pipeline check   |"
    echo "  +------------------------------------------------------------+"
    echo "  |  Pretraining (SFT):                                       |"
    echo "  |   [5] 7B Dense -- 1~8 GPU                                 |"
    echo "  |   [6] Ultra 7B (39B total / 7.5B active) -- 8 GPU         |"
    echo "  |   [7] Ultra 37B (200B / 37B) -- 32 GPU                    |"
    echo "  |   [8] Ultra 371B (371B / 28B) -- 64 GPU                   |"
    echo "  |   [9] Ultra 671B MAX (671B / 37B) -- 64~128 GPU           |"
    echo "  +------------------------------------------------------------+"
    echo "  |  Quick Train (budget-friendly):                           |"
    echo "  |   [U] 1B Ultra (~40M active, all features) -- 1 GPU, ~8hr |"
    echo "  +------------------------------------------------------------+"
    echo "  |  Post-training:                                           |"
    echo "  |   [G] GRPO Reasoning RL (with S-GRPO option)              |"
    echo "  +------------------------------------------------------------+"
    echo "  |  Inference / Demo:                                        |"
    echo "  |   [R] Generate with thinking mode                         |"
    echo "  +------------------------------------------------------------+"
    echo "  |  Other:                                                   |"
    echo "  |   [T] Run all tests                                       |"
    echo "  |   [Q] Quit                                                |"
    echo "  +------------------------------------------------------------+"
    echo
    read -p "  Select [0-9/A-H/G/R/T/U/Q]: " CHOICE

    case "$CHOICE" in
        # -- Download Data --
        [Aa]) data_download "debug"      "Debug (~1B tokens)" ;;
        [Bb]) data_download "1b"         "1B (~20B tokens)" ;;
        [Cc]) data_download "7b"         "7B (~140B tokens)" ;;
        [Dd]) data_download "ultra-7b"   "Ultra 7B (~140B tokens)" ;;
        [Ee]) data_download "ultra-37b"   "Ultra 37B (~740B tokens)" ;;
        [Ff]) data_download "ultra-371b"  "Ultra 371B (~560B tokens)" ;;
        [Hh]) data_download "ultra-671b"  "Ultra 671B (~1T tokens)" ;;

        # -- Data Processing --
        1)
            echo
            echo "  +------------------------------------------------------------+"
            echo "  |  Step 1: Auto Clean + Classify                             |"
            echo "  +------------------------------------------------------------+"
            echo
            python "$PROJECT_DIR/scripts/data_pipeline.py" --input "$DATA_RAW" --output "$DATA_PROCESSED"
            echo -e "${GREEN}[DONE]${RESET} Cleaned data in data/processed/"
            read -p "Press Enter to continue..."
            ;;
        2)
            echo
            echo "  +------------------------------------------------------------+"
            echo "  |  Step 2: Tokenize -> .bin                                  |"
            echo "  +------------------------------------------------------------+"
            select_model_size
            read -p "  Sequence length [8192]: " SEQ_LEN_IN
            SEQ_LEN="${SEQ_LEN_IN:-8192}"
            read -p "  Shard count [64]: " NUM_SHARDS
            NUM_SHARDS="${NUM_SHARDS:-64}"
            echo
            echo "  Tokenizing..."

            # Find processed files
            FOUND_INPUT=""
            if [ -f "$DATA_PROCESSED/cleaned.jsonl" ]; then
                FOUND_INPUT="$DATA_PROCESSED/cleaned.jsonl"
            else
                FOUND_INPUT=$(ls "$DATA_PROCESSED"/*.jsonl 2>/dev/null | head -1)
            fi
            if [ -z "$FOUND_INPUT" ]; then
                FOUND_INPUT=$(ls "$DATA_PROCESSED"/*.txt 2>/dev/null | head -1)
            fi
            if [ -z "$FOUND_INPUT" ]; then
                echo -e "${YELLOW}[WARN]${RESET} No processed files found in data/processed/"
                read -p "  Enter file path: " FOUND_INPUT
            fi

            # Detect format
            INPUT_TYPE="txt"
            if echo "$FOUND_INPUT" | grep -qi '\.jsonl$'; then
                INPUT_TYPE="jsonl"
            fi

            python "$PROJECT_DIR/scripts/prepare_data.py" \
                --input "$FOUND_INPUT" \
                --input_type "$INPUT_TYPE" \
                --output "$DATA_TOKENIZED" \
                --tokenizer huggyllama/llama-7b \
                --seq_len "$SEQ_LEN" \
                --num_shards "$NUM_SHARDS"
            echo -e "${GREEN}[DONE]${RESET} Binary files in data/tokenized/"
            read -p "Press Enter to continue..."
            ;;
        3)
            echo
            echo "  +------------------------------------------------------------+"
            echo "  |  Full Pipeline (Clean + Tokenize)                          |"
            echo "  +------------------------------------------------------------+"
            echo
            echo "  [1/2] Cleaning data..."
            python "$PROJECT_DIR/scripts/data_pipeline.py" --input "$DATA_RAW" --output "$DATA_PROCESSED"
            echo
            echo "  [2/2] Tokenizing..."
            select_model_size

            FOUND_INPUT=$(ls "$DATA_PROCESSED"/*.jsonl 2>/dev/null | head -1)
            if [ -z "$FOUND_INPUT" ]; then
                FOUND_INPUT=$(ls "$DATA_PROCESSED"/*.txt 2>/dev/null | head -1)
            fi
            if [ -z "$FOUND_INPUT" ]; then
                echo -e "${RED}[ERROR]${RESET} No output files found after cleaning!"
                read -p "Press Enter to continue..."
                continue
            fi

            INPUT_TYPE="txt"
            if echo "$FOUND_INPUT" | grep -qi '\.jsonl$'; then
                INPUT_TYPE="jsonl"
            fi

            python "$PROJECT_DIR/scripts/prepare_data.py" \
                --input "$FOUND_INPUT" \
                --input_type "$INPUT_TYPE" \
                --output "$DATA_TOKENIZED" \
                --tokenizer huggyllama/llama-7b \
                --seq_len "$SEQ_LEN" \
                --num_shards 64
            echo
            echo -e "${GREEN}[DONE]${RESET} Pipeline complete! Ready for training."
            read -p "Press Enter to continue..."
            ;;

        # -- Verify / Debug --
        0)
            echo
            echo "  +------------------------------------------------------------+"
            echo "  |  Architecture Verify (0.1B, all Ultra features ON)         |"
            echo "  +------------------------------------------------------------+"
            echo
            python "$PROJECT_DIR/scripts/gen_arch_config.py"
            CONFIG="$PROJECT_DIR/configs/_arch_verify.yaml"
            BATCH_SIZE=1; GRAD_ACCUM=4; MAX_STEPS=300; LR=8e-4
            MAX_SEQ_LEN=512; SAVE_EVERY=100; LOG_EVERY=10; WARMUP=30
            WANDB_FLAG=""
            train_launch
            ;;
        4)
            echo
            echo "  +------------------------------------------------------------+"
            echo "  |  Debug Model (0.01B, 4-layer, d_model=256)                |"
            echo "  +------------------------------------------------------------+"
            CONFIG="debug"
            BATCH_SIZE=2; GRAD_ACCUM=4; MAX_STEPS=200; LR=1e-3
            MAX_SEQ_LEN=128; SAVE_EVERY=50; LOG_EVERY=5; WARMUP=20
            WANDB_FLAG=""
            train_launch
            ;;

        # -- Pretraining --
        5)
            CONFIG="$PROJECT_DIR/configs/7b.yaml"
            BATCH_SIZE=2; GRAD_ACCUM=8; MAX_STEPS=100000; LR=3e-4
            MAX_SEQ_LEN=8192; SAVE_EVERY=5000; LOG_EVERY=10; WARMUP=2000
            WANDB_FLAG=""
            train_launch
            ;;
        6)
            CONFIG="$PROJECT_DIR/configs/ultra-7b.yaml"
            BATCH_SIZE=1; GRAD_ACCUM=8; MAX_STEPS=100000; LR=3e-4
            MAX_SEQ_LEN=8192; SAVE_EVERY=5000; LOG_EVERY=10; WARMUP=2000
            WANDB_FLAG=""
            train_launch
            ;;
        7)
            CONFIG="$PROJECT_DIR/configs/ultra-37b.yaml"
            BATCH_SIZE=1; GRAD_ACCUM=16; MAX_STEPS=200000; LR=1.5e-4
            MAX_SEQ_LEN=8192; SAVE_EVERY=5000; LOG_EVERY=10; WARMUP=4000
            WANDB_FLAG="--use_wandb"
            train_launch
            ;;
        8)
            CONFIG="$PROJECT_DIR/configs/ultra-371b.yaml"
            BATCH_SIZE=1; GRAD_ACCUM=16; MAX_STEPS=200000; LR=1.5e-4
            MAX_SEQ_LEN=8192; SAVE_EVERY=5000; LOG_EVERY=10; WARMUP=4000
            WANDB_FLAG="--use_wandb"
            train_launch
            ;;
        9)
            CONFIG="$PROJECT_DIR/configs/ultra-671b-max.yaml"
            BATCH_SIZE=1; GRAD_ACCUM=16; MAX_STEPS=200000; LR=1.5e-4
            MAX_SEQ_LEN=8192; SAVE_EVERY=5000; LOG_EVERY=10; WARMUP=4000
            WANDB_FLAG="--use_wandb"
            train_launch
            ;;

        # -- 1B Ultra Train (budget-friendly full architecture) --
        [Uu])
            echo
            echo "  +------------------------------------------------------------+"
            echo "  |  1B Ultra Train (~40M active, all features ON)             |"
            echo "  |  MoE + KDA-Diff + MTP + Interleave -- full architecture    |"
            echo "  +------------------------------------------------------------+"
            echo
            echo "  Budget: ~8 hours on H100 (~HK$184)"
            echo "  Good for: architecture validation at meaningful scale"
            echo
            python "$PROJECT_DIR/scripts/gen_1b_ultra_config.py"
            if [ $? -ne 0 ]; then
                echo -e "${RED}[ERROR]${RESET} Config generation failed!"
                read -p "Press Enter to continue..."
                continue
            fi
            CONFIG="$PROJECT_DIR/configs/_1b_ultra.yaml"
            BATCH_SIZE=2; GRAD_ACCUM=4; MAX_STEPS=14000; LR=3e-4
            MAX_SEQ_LEN=2048; SAVE_EVERY=2000; LOG_EVERY=50; WARMUP=500
            WANDB_FLAG=""
            train_launch
            ;;

        # -- GRPO --
        [Gg])
            echo
            echo "  +------------------------------------------------------------+"
            echo "  |  GRPO Reasoning RL (+ S-GRPO option)                       |"
            echo "  +------------------------------------------------------------+"
            echo
            echo "  Available checkpoints:"
            if ls "$CHECKPOINT_DIR"/*.pt 2>/dev/null; then :; else echo "    (none)"; fi
            echo
            read -p "  SFT checkpoint path: " GRPO_CKPT
            read -p "  GRPO prompt data [data/grpo_prompts.jsonl]: " GRPO_DATA
            GRPO_DATA="${GRPO_DATA:-$PROJECT_DIR/data/grpo_prompts.jsonl}"
            select_model_size
            read -p "  Reward type [math/format/code/combined] [math]: " GRPO_REWARD
            GRPO_REWARD="${GRPO_REWARD:-math}"
            read -p "  Group size G [8]: " GRPO_G
            GRPO_G="${GRPO_G:-8}"
            read -p "  KL penalty beta [0.04]: " GRPO_BETA
            GRPO_BETA="${GRPO_BETA:-0.04}"
            read -p "  Max steps [10000]: " GRPO_MAX_STEPS
            GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-10000}"

            read -p "  Enable S-GRPO (sparse token sampling)? [y/N]: " USE_SGRPO
            SGRPO_FLAG=""
            SGRPO_P=0.4; SGRPO_ALPHA=0; SGRPO_K=0
            if [ "$USE_SGRPO" = "y" ] || [ "$USE_SGRPO" = "Y" ]; then
                SGRPO_FLAG="--sgrpo"
                read -p "    Token sampling probability [0.4]: " SGRPO_P
                SGRPO_P="${SGRPO_P:-0.4}"
                read -p "    First N tokens always kept [0]: " SGRPO_ALPHA
                SGRPO_ALPHA="${SGRPO_ALPHA:-0}"
                read -p "    Max tokens cap (0=disabled) [0]: " SGRPO_K
                SGRPO_K="${SGRPO_K:-0}"
                echo "    [OK] S-GRPO enabled (p=$SGRPO_P)"
            fi

            echo
            echo "  Launching GRPO training..."

            if [ "$GPU_COUNT" -le 1 ]; then
                python "$PROJECT_DIR/scripts/train_grpo.py" \
                    --config "$CONFIG" \
                    --checkpoint "$GRPO_CKPT" \
                    --data "$GRPO_DATA" \
                    --reward_type "$GRPO_REWARD" \
                    --group_size "$GRPO_G" \
                    --kl_beta "$GRPO_BETA" \
                    --max_steps "$GRPO_MAX_STEPS" \
                    --batch_size 4 \
                    --gradient_accumulation_steps 2 \
                    --learning_rate 1e-6 \
                    $BF16_FLAG \
                    --max_prompt_len 2048 \
                    --gen_max_tokens 1024 \
                    --output_dir "$GRPO_CHECKPOINT_DIR" \
                    $SGRPO_FLAG --sgrpo_p "$SGRPO_P" --sgrpo_alpha "$SGRPO_ALPHA" --sgrpo_k "$SGRPO_K"
            else
                torchrun --nproc_per_node="$GPU_COUNT" "$PROJECT_DIR/scripts/train_grpo.py" \
                    --config "$CONFIG" \
                    --checkpoint "$GRPO_CKPT" \
                    --data "$GRPO_DATA" \
                    --reward_type "$GRPO_REWARD" \
                    --group_size "$GRPO_G" \
                    --kl_beta "$GRPO_BETA" \
                    --max_steps "$GRPO_MAX_STEPS" \
                    --batch_size 4 \
                    --gradient_accumulation_steps 2 \
                    --learning_rate 1e-6 \
                    $BF16_FLAG \
                    --max_prompt_len 2048 \
                    --gen_max_tokens 1024 \
                    --output_dir "$GRPO_CHECKPOINT_DIR" \
                    $SGRPO_FLAG --sgrpo_p "$SGRPO_P" --sgrpo_alpha "$SGRPO_ALPHA" --sgrpo_k "$SGRPO_K"
            fi
            echo -e "${GREEN}[DONE]${RESET} GRPO training complete!"
            read -p "Press Enter to continue..."
            ;;

        # -- Demo --
        [Rr])
            echo
            echo "  +------------------------------------------------------------+"
            echo "  |  Generate with Thinking Mode                               |"
            echo "  +------------------------------------------------------------+"
            echo
            echo "  Available checkpoints:"
            if ls "$CHECKPOINT_DIR"/*.pt 2>/dev/null; then :; else echo "    (none - using random init)"; fi
            echo
            read -p "  Checkpoint path (or blank for random): " DEMO_CKPT
            read -p "  Prompt: " DEMO_PROMPT
            select_model_size
            select_thinking_mode
            read -p "  Max new tokens [256]: " DEMO_TOKENS
            DEMO_TOKENS="${DEMO_TOKENS:-256}"

            THINK_ARGS=""
            if [ "$THINK_MODE" != "NoThink" ]; then
                THINK_ARGS="--thinking $THINK_MODE --think_budget $THINK_BUDGET --num_paths $THINK_PATHS --summary_budget $THINK_SUMMARY"
                read -p "  Show thinking tokens? [y/N]: " SHOW_THINK
                if [ "$SHOW_THINK" = "y" ] || [ "$SHOW_THINK" = "Y" ]; then
                    THINK_ARGS="$THINK_ARGS --show_thinking"
                fi
            fi

            CKPT_ARG=""
            if [ -n "$DEMO_CKPT" ]; then
                CKPT_ARG="--checkpoint $DEMO_CKPT"
            fi

            echo
            echo "  Generating..."
            python "$PROJECT_DIR/scripts/generate.py" \
                --config "$CONFIG" \
                --prompt "$DEMO_PROMPT" \
                $CKPT_ARG \
                --max_new_tokens "$DEMO_TOKENS" \
                --temperature 0.7 \
                $THINK_ARGS
            echo
            read -p "Press Enter to continue..."
            ;;

        # -- Tests --
        [Tt])
            echo
            echo "  Running all tests..."
            python -m pytest "$PROJECT_DIR/tests" -v --tb=short
            echo
            read -p "Press Enter to continue..."
            ;;

        # -- Quit --
        [Qq])
            echo
            echo "  Goodbye!"
            exit 0
            ;;

        *)
            ;;
    esac
done
