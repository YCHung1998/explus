#!/bin/bash

FOLDER="phash_0112_bdry"
# not yet
# FOLDER="Block_0112_ema_phash"
# FOLDER="Block_0112_yolo"

FOLDER="phash_0112_bdry_v2"
SESSION="eval_session"
PORT="5005"

# 1. 判斷要使用的腳本名稱
if [[ "$FOLDER" == *phash* ]]; then
    EVAL_SCRIPT="scripts/eval_phash.sh"
    PY='inference_something_phash.py'
    echo "檢測到 phash 關鍵字，選用: $EVAL_SCRIPT"
else
    EVAL_SCRIPT="scripts/eval.sh"
    PY='inference_something.py'
    echo "未檢測到 phash，選用預設: $EVAL_SCRIPT"
fi

tmux kill-session -t $SESSION 2>/dev/null

# --- 開始 tmux 任務 ---

# Pane 1: Inference
# 在指令前加上 echo，讓你進入 tmux 後能看到剛剛下了什麼指令
CMD1="python ${PY}"
tmux new-session -d -s $SESSION "echo -e '\033[1;34m執行指令: $CMD1\033[0m'; $CMD1; tmux wait-for -S task1_done; bash"

# Pane 2: Evaluation
CMD2="bash ${EVAL_SCRIPT} -f ${FOLDER}"
tmux split-window -h -t $SESSION "tmux wait-for task1_done; echo -e '\033[1;32m執行指令: $CMD2\033[0m'; $CMD2; tmux wait-for -S task2_done; bash"

# Pane 3: Visualization
# 這裡使用變數包裝長指令，讓 echo 出來的畫面比較整潔
GT_PATH="/Users/eason.hung/Documents/Projects/explus/output/${FOLDER}/ground_truth/data.json"
PD_PATH="/Users/eason.hung/Documents/Projects/explus/output/${FOLDER}/predictions/merge_data.json"
CMD3="python visual_results.py --gt $GT_PATH --pd $PD_PATH --port ${PORT}"

tmux split-window -v -t $SESSION "tmux wait-for task2_done; echo -e '\033[1;33m執行指令: $CMD3\033[0m'; $CMD3; bash"

# 進入該 session
tmux attach-session -t $SESSION