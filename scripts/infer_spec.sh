#!/bin/bash

# ==============================================================================
# 環境路徑設定
# ==============================================================================
# 輸出根目錄
# BASE_OUT_DIR="/vol/08822801/AutoTrigger/TMP_output"
BASE_OUT_DIR="/Users/eason.hung/Documents/Projects/explus/output"

# ==============================================================================
# Task 名稱定義 (在這邊管理你的任務組合)
# ==============================================================================
# TASK_NECK="Block_m0_ema_phash"
TASK_NECK="Block_m0_Neck_P4_yolo_v2"
# TASK_NECK="Block_m0_Backbone_fusion_yolo_v3"
# TASK_BACKBONE="Block_m0_Backbone_fusion_yolo_v2"

# ==============================================================================
# 1. 執行推論 (Inference)
# ==============================================================================

echo ">>> [1/2] Running Inference: $TASK_NECK"
# python inference_something.py -m ema_phash -v P4 -feat Neck -mr 0.0 -o "${BASE_OUT_DIR}/${TASK_NECK}"
python inference_something.py -m yolo -v P4 -feat Neck -mr 0.0 -o "${BASE_OUT_DIR}/${TASK_NECK}"
# python inference_something.py -m yolo -v P4 -feat Backbone -mr 0.0 -o "${BASE_OUT_DIR}/${TASK_NECK}" -sim 0.4
# python inference_something.py -m yolo -v P4 -feat Backbone -mr 0.0 -o "${BASE_OUT_DIR}/${TASK_NECK}"

# echo ">>> [2/2] Running Inference: $TASK_BACKBONE"
# python inference_something.py -m yolo -v fusion -feat Backbone -mr 0.0 -o "${BASE_OUT_DIR}/${TASK_BACKBONE}"

# ==============================================================================
# 2. 執行驗證 (Evaluation)
# ==============================================================================

# 使用 Array 進行循環驗證，避免重複寫 eval 指令
for TASK in "$TASK_NECK" # "$TASK_BACKBONE"
do
    echo "--------------------------------------------------"
    echo "Evaluating Task: ${TASK}"
    
    GT_PATH="${BASE_OUT_DIR}/${TASK}/ground_truth/data.json"
    PD_PATH="${BASE_OUT_DIR}/${TASK}/predictions/merge_data.json"
    
    # 檢查檔案是否存在再執行，避免噴錯
    if [ -f "$GT_PATH" ] && [ -f "$PD_PATH" ]; then
        python -m mmaction2.evaluation.eval_custom --gt "$GT_PATH" --pd "$PD_PATH"
    else
        echo "[Error] File not found for task ${TASK}:"
        echo "        GT: $GT_PATH"
        echo "        PD: $PD_PATH"
    fi
done

echo "=================================================="
echo "All process finished."