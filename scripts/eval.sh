#!/bin/bash

# 設定基礎路徑 (Base Path)
BASE_PATH="/Users/eason.hung/Documents/Projects/explus/output"
TARGET_FOLDER="Block_0112_ema_phash"
# TARGET_FOLDER="Block_0112_yolo"

# 預設執行指令（當沒有參數時）
if [ $# -eq 0 ]; then
    echo "執行預設評估程序..."
    python -m mmaction2.evaluation.eval_custom -eg
    exit 0
fi

# -f -m
while getopts "f:m" opt; do
  case $opt in
    f)
      TARGET_FOLDER=$OPTARG
      ;;
    \?)
      echo "無效的參數: -$OPTARG"
      exit 1
      ;;
    :)
      echo "參數 -$OPTARG 需要輸入路徑"
      exit 1
      ;;
  esac
done

# 檢查是否有抓到參數
if [ -z "$TARGET_FOLDER" ]; then
    echo "錯誤: 必須使用 -f 指定資料夾路徑"
    echo "用法: bash eval_phash.sh -f /path/to/folder"
    exit 1
fi


# 組合路徑並執行
GT_PATH="${BASE_PATH}/${TARGET_FOLDER}/ground_truth/data.json"
PD_PATH="${BASE_PATH}/${TARGET_FOLDER}/predictions/merge_data.json"

echo "正在評估模式: $MODE"
echo "GT 路徑: $GT_PATH"
echo "PD 路徑: $PD_PATH"

python -m mmaction2.evaluation.eval_custom \
    --gt "$GT_PATH" \
    --pd "$PD_PATH"