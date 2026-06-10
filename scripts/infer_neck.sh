#!/bin/bash
set -e

# --- Neck 組 ---
python inference_something.py -m yolo -v P4 -feat Neck -mr 0.0 -o output/Block_m0_Neck_P4_yolo
python inference_something.py -m yolo -v P4 -feat Neck -mr 0.4 -o output/Block_m0.4_Neck_P4_yolo
python inference_something.py -m yolo -v P4 -feat Neck -mr 1.0 -o output/Block_m1_Neck_P4_yolo

python inference_something.py -m yolo -v fusion -feat Neck -mr 0.0 -o output/Block_m0_Neck_fusion_yolo
python inference_something.py -m yolo -v fusion -feat Neck -mr 0.4 -o output/Block_m0.4_Neck_fusion_yolo
python inference_something.py -m yolo -v fusion -feat Neck -mr 1.0 -o output/Block_m1_Neck_fusion_yolo

# # --- Backbone 組 ---
# python inference_something.py -m yolo -v P4 -feat Backbone -mr 0.0 -o output/Block_m0_Backbone_P4_yolo
# python inference_something.py -m yolo -v P4 -feat Backbone -mr 0.4 -o output/Block_m0.4_Backbone_P4_yolo
# python inference_something.py -m yolo -v P4 -feat Backbone -mr 1.0 -o output/Block_m1_Backbone_P4_yolo

# python inference_something.py -m yolo -v fusion -feat Backbone -mr 0.0 -o output/Block_m0_Backbone_fusion_yolo
# python inference_something.py -m yolo -v fusion -feat Backbone -mr 0.4 -o output/Block_m0.4_Backbone_fusion_yolo
# python inference_something.py -m yolo -v fusion -feat Backbone -mr 1.0 -o output/Block_m1_Backbone_fusion_yolo