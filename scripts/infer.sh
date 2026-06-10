# --- Neck 組 ---
# python3 inference_something.py -m yolo -v P4 -feat Neck --stable_hold_time 0.1 --unstable_hold_time 0.5  -o output/Block_m0_Neck_P4_yolo_2_8
# python inference_something.py -m yolo -v P4 -feat Neck -mr 0.0 -o output/Block_m0_Neck_P4_yolo
python inference_something.py -m yolo -v P4 -feat Neck -mr 0.0 -o /vol/08822801/AutoTrigger/TMP_output/Block_m0_Neck_P4_yolo
# python3 inference_something.py -m yolo -v P4 -feat Neck -mr 0.0 --stable_hold_time 0.3 --unstable_hold_time 0.3  -o output/Block_m0_Neck_P4_yolo_3_3


# python inference_something.py -m yolo -v fusion -feat Neck -mr 0.0 -o output/Block_m0_Neck_fusion_yolo
# python inference_something.py -m yolo -v fusion -feat Neck -mr 0.4 -o output/Block_m0.4_Neck_fusion_yolo
# python inference_something.py -m yolo -v fusion -feat Neck -mr 1.0 -o output/Block_m1_Neck_fusion_yolo

# # --- Backbone 組 ---
# (Backbone, P4) -> 結果很差 不使用
# python inference_something.py -m yolo -v P4 -feat Backbone -mr 0.0 -o output/Block_m0_Backbone_P4_yolo
python inference_something.py -m yolo -v fusion -feat Backbone -mr 0.0 -o /vol/08822801/AutoTrigger/TMP_output/Block_m0_Backbone_fusion_yolo_v2


# --- Neck 組 驗證 ---
# python -m mmaction2.evaluation.eval_custom --gt /Users/eason.hung/Documents/Projects/explus/output/Block_m0_Neck_P4_yolo/ground_truth/data.json --pd /Users/eason.hung/Documents/Projects/explus/output/Block_m0_Neck_P4_yolo/predictions/merge_data.json
python -m mmaction2.evaluation.eval_custom --gt /vol/08822801/AutoTrigger/TMP_output/Block_m0_Neck_P4_yolo/ground_truth/data.json --pd /vol/08822801/AutoTrigger/TMP_output/Block_m0_Neck_P4_yolo/predictions/merge_data.json
echo "Block_m0_Neck_P4_yolo"

# python -m mmaction2.evaluation.eval_custom --gt /Users/eason.hung/Documents/Projects/explus/output/Block_m0_Neck_P4_yolo_3_3/ground_truth/data.json --pd /Users/eason.hung/Documents/Projects/explus/output/Block_m0_Neck_P4_yolo_3_3/predictions/merge_data.json
# echo "Block_m0_Neck_P4_yolo_3_3"

# python -m mmaction2.evaluation.eval_custom --gt /Users/eason.hung/Documents/Projects/explus/output/Block_m0.4_Neck_P4_yolo/ground_truth/data.json --pd /Users/eason.hung/Documents/Projects/explus/output/Block_m0.4_Neck_P4_yolo/predictions/merge_data.json
# echo "Block_m0.4_Neck_P4_yolo"

# python -m mmaction2.evaluation.eval_custom --gt /Users/eason.hung/Documents/Projects/explus/output/Block_m1_Neck_P4_yolo/ground_truth/data.json --pd /Users/eason.hung/Documents/Projects/explus/output/Block_m1_Neck_P4_yolo/predictions/merge_data.json
# echo "Block_m1_Neck_P4_yolo"

# --- Backbone 組 驗證 ---

# python -m mmaction2.evaluation.eval_custom --gt /Users/eason.hung/Documents/Projects/explus/output/Block_m0_Backbone_fusion_yolo/ground_truth/data.json --pd /Users/eason.hung/Documents/Projects/explus/output/Block_m0_Backbone_fusion_yolo/predictions/merge_data.json
python -m mmaction2.evaluation.eval_custom --gt /vol/08822801/AutoTrigger/TMP_output/Block_m0_Backbone_fusion_yolo/ground_truth/data.json --pd /vol/08822801/AutoTrigger/TMP_output/Block_m0_Backbone_fusion_yolo/predictions/merge_data.json
echo "Block_m0_Backbone_fusion_yolo"

# python -m mmaction2.evaluation.eval_custom --gt /Users/eason.hung/Documents/Projects/explus/output/Block_m0.4_Backbone_fusion_yolo/ground_truth/data.json --pd /Users/eason.hung/Documents/Projects/explus/output/Block_m0.4_Backbone_fusion_yolo/predictions/merge_data.json
# echo "Block_m0.4_Backbone_fusion_yolo"


# python -m mmaction2.evaluation.eval_custom --gt /Users/eason.hung/Documents/Projects/explus/output/Block_m1_Backbone_fusion_yolo/ground_truth/data.json --pd /Users/eason.hung/Documents/Projects/explus/output/Block_m1_Backbone_fusion_yolo/predictions/merge_data.json
# echo "Block_m1_Backbone_fusion_yolo"



# python -m mmaction2.evaluation.eval_custom --gt /Users/eason.hung/Documents/Projects/explus/output/Block_m1_Neck_P4_yolo_repeat/ground_truth/data.json --pd /Users/eason.hung/Documents/Projects/explus/output/Block_m1_Neck_P4_yolo_repeat/predictions/merge_data.json
# echo "Block_m1_Neck_P4_yolo_repeat"

# python -m mmaction2.evaluation.eval_custom --gt /Users/eason.hung/Documents/Projects/explus/output/Block_m0_Neck_P4_yolo_repeat/ground_truth/data.json --pd /Users/eason.hung/Documents/Projects/explus/output/Block_m0_Neck_P4_yolo_repeat/predictions/merge_data.json
# echo "output/Block_m0_Neck_P4_yolo_repeat"


# python visual_results.py \
#     --gt /Users/eason.hung/Documents/Projects/explus/output/Block_m1_Neck_P4_yolo/ground_truth/data.json \
#     --pd /Users/eason.hung/Documents/Projects/explus/output/Block_m1_Neck_P4_yolo/predictions/merge_data.json \
#     --port 5002