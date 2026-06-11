### Slides
[AutoTrigger v2](https://docs.google.com/presentation/d/1aJF0RL69w7tJLthgDJLLyZP8ysn_I2pG-hzzKn7oXL4/edit?slide=id.p#slide=id.p)
[AutoTrigger Benchmark slide(2026-02-02)](https://docs.google.com/presentation/d/1_9LxiAdict5pj3IXNbtHb8rx88f29eX3T8jnwxlgSmE/edit?slide=id.g3bb66e257f6_1_0#slide=id.g3bb66e257f6_1_0)

# AutoTrigger — 即時觸發偵測系統
針對固定攝影機場景(bread checkout / demo room)的**觸發事件偵測**:偵測畫面從「穩定」轉為「不穩定」並完成一次有效操作(Trigger),最終目標是部署到 **real-time edge device**。


```
影像輸入 ─► 前處理 ─► 動態偵測 ─► 訊號狀態機 ─► 觸發驗證 ─► 輸出 segment
           (grayscale,  (block-based +   (stable /     (YOLO ONNX 或
            縮放,分塊)   texture/LBSP +   unstable      ema_phash 比對)
                         background EMA)   狀態切換)
```

從模型到 demo 的整體脈絡:**YOLOv12 fine-tune → export ONNX → [add_feature_map_output.py](add_feature_map_output.py) 拆出 feature map → 收影片用 VIDAT2 標註 → [convert_vidat.py](convert_vidat.py) 轉 ActivityNet GT json → 由單一 `run.yaml` 驅動 inference / 評估 / 視覺化 / demo。**

核心模組都在 [src/](src/):`preprocessor` → `background` → `block_engine`/`texture_engine` → `motion_decider` → `signal_processor` → `trigger_verifier`,由 [src/detection_pipeline.py](src/detection_pipeline.py) 的 `DetectionPipeline` 串接,演算法參數見 [src/pipeline_config.py](src/pipeline_config.py)。

---

## 1. 快速開始(操作流程)

### 1.1 環境安裝(Python 3.12)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements_test.txt                # 跑測試才需要
```

### 1.2 資料與權重就位(未進版控,需自行放置)

`external_camera/`、`models/`、`weight/`、`datasets/` 都在 `.gitignore` 內,**全新 clone 不會有**。執行任何推論前需先就位 —— 共用影片與模型都在實驗室 server `/vol/08822801/AutoTrigger/` 之下:

| 內容 | 放置位置 | 來源 |
|---|---|---|
| 影片 + GT 標註(`*.mp4` + `*_annotations_ActivityNet.json`) | `external_camera/<子資料夾>/` | server `/vol/08822801/AutoTrigger/dataset/external_camera/`,或向專案擁有者索取 |
| ONNX 模型 | `models/`,或在 config 直接指向絕對路徑 | server `/vol/08822801/AutoTrigger/model/` |

要跑哪些資料由 `run.yaml` 的 `source.dataset`(dataset CSV)決定;要用哪顆模型由 `pipeline.trigger.model_path` 決定 —— 設了就用該路徑,沒設(`null`)則依 `feature_position` 推導預設(Neck→`models/best_vis_with_8400_3.onnx`、Backbone→`models/best_vis_with_8400_b3.onnx`,見 [src/pipeline_config.py](src/pipeline_config.py) `TriggerConfig.resolved_model_path`)。換到其他機器只需改 config 與 CSV 內的影片路徑。

### 1.3 執行

```bash
# 單影片 / 攝影機即時偵測(-f 覆蓋 config 的 source 為單一影片;不帶則依 source.type 走)
python main.py --config configs/run.yaml -f <影片路徑.mp4>
python main.py --config configs/run.yaml          # source.type: dataset/video/camera
#   選項:-sign_record(edge 狀態記錄,append-only,預設關)  --no-visual  --save-video

# 批次推論(資料集/輸出/演算法全由 config 提供;同時輸出 predictions 與 ground_truth)
python batch_infer.py --config configs/run.yaml
python batch_infer_phash.py --config configs/run.yaml   # 舊 phash 批次線

# 評估與視覺化(--config 自動推導 gt/pd,或沿用 --gt/--pd)
python -m mmaction2.evaluation.eval_custom --config configs/run.yaml
python eval_dashboard.py --config configs/run.yaml --port 5005   # Flask 評估儀表板
python visual_results.py --config configs/run.yaml --port 5000   # 時間軸視覺化
streamlit run streamlit_app.py -- --config configs/run.yaml       # 互動調參,見 README_STREAMLIT.md
streamlit run streamlit_app.py                                    # 互動調參,見 README_STREAMLIT.md
```

批次輸出 layout(評估與視覺化都依賴):

```
output/<run>/
├── predictions/merge_data.json   # {"results": {video_id: [{segment,label,score}]}}
├── predictions/<video_id>.json
└── ground_truth/data.json        # {video_id: {annotations:[{segment,label,score}], ...}}
```

無法開啟的影片會記錄到 `output/<run>/skipped.json`(不再被靜默跳過)。

---

## 2. 衡量基準(Benchmark)

評估採**雙階段**(`eval_custom.py`):**Stage 1 低階定位**(畫面是否在對的時間判定為不穩定)與 **Stage 2 觸發決策**(是否 1:1 對上每一次有效 Trigger)。Stable(label 1)在評估時忽略。

參考成績 —— Neck / P4 / yolo 設定([configs/neck_p4.yaml](configs/neck_p4.yaml)),`output/run_exp_neck_p4`:

| 指標 | Stage 1(定位) | Stage 2(觸發) |
|---|---|---|
| Precision | 75.00% | 90.24% |
| Recall | 81.08% | 90.24% |
| F1 | 77.92% | — |
| Perfect Capture Rate | — | 90.24% |
| mAP (ActivityNet) | 0.239 | 0.200 |

重現:`python -m mmaction2.evaluation.eval_custom --gt output/run_exp_neck_p4/ground_truth/data.json --pd output/run_exp_neck_p4/predictions/merge_data.json`。

---

## 3. 標籤與資料格式

標籤定義(見 [constants.py](constants.py) / [label_map.txt](label_map.txt)):

| ID | 標籤 |
|---|---|
| 1 | Stable(背景,評估時忽略) |
| 2 | Positive Unstable |
| 3 | Negative Unstable External Disturbances |
| 4 | Negative Unstable Overexposure |
| 5 | Trigger |

標註流程(**source of truth 是 json**):VIDAT2 平台(https://vidat2.davidz.cn)匯出原始 json → [convert_vidat.py](convert_vidat.py) → `<video>_annotations_ActivityNet.json`(GT 標準格式,每影片一個,含 `segment/label/score`)。

- **GT 標準格式** = ActivityNet 風格 `*_annotations_ActivityNet.json`。
- **標籤 schema** = `external_camera/config.json`(`actionLabelData`)。
- 一支影片切多段 = ActivityNet `annotations` 陣列裡的多個 segment。

---

## 4. 從 GT json 產生 dataset CSV

[scripts/build_dataset_csv.py](scripts/build_dataset_csv.py) 把 GT json **單向展平**成扁平 CSV 索引(一列一片段),方便挑選 / 切 split / 開關子集。json 永遠是來源,CSV 可隨時重生。

```bash
python scripts/build_dataset_csv.py \
    --src /vol/08822801/AutoTrigger/dataset/external_camera/ \
    --out datasets/bread_demo.csv
```

> ⚠️ `--src` 請用**絕對路徑**:CSV 的 `video_path` 欄會原樣沿用 `--src` 的形式,後續開影片 / 找 boundary json / 儀表板播影片都直接吃它。資料搬到哪、就對著那個絕對路徑重 build 一次。`.mp4` 與其 `_annotations_ActivityNet.json` 必須放在一起。

CSV 欄位:`video_path, start, end, label, split, enabled, boundary`。

---

## 5. Config 驅動

所有 runner 由**單一 config 驅動**:

- [src/config/run_config.py](src/config/run_config.py) — `RunConfig`(實驗層:source / output / record),內嵌既有 `PipelineConfig`(演算法層)。
- [configs/run.example.yaml](configs/run.example.yaml) — 參考範本(另有 [configs/neck_p4.yaml](configs/neck_p4.yaml) 等具體實驗 config)。
- [src/io/](src/io/) — `dataset_csv`(CSV ↔ GT)、`sources`(dataset/video/camera)、`state_recorder`(append-only 狀態記錄,預設關)。
- [src/runtime/clip_runner.py](src/runtime/clip_runner.py) — `run_clip()` 單-clip 擷取迴圈,`main.py` 與 `batch_infer.py` 共用。

```python
from src.config.run_config import RunConfig
cfg = RunConfig.from_yaml("configs/run.example.yaml")
```

完整設計與計畫見 [docs/superpowers/](docs/superpowers/)。

---

## 6. 測試

```bash
.venv/bin/python -m pytest -q
```

涵蓋 config 與 dataset-IO 模組(`tests/`);pytest 設定只收 `tests/`(見 `pyproject.toml`)。

---

## 7. 專案結構

```
.
├── main.py / batch_infer.py / batch_infer_phash.py     # 執行入口(皆吃 --config)
├── eval_dashboard.py / visual_results.py / streamlit_app.py   # 評估 / 視覺化
├── src/                  # 偵測 pipeline 核心
│   ├── config/           # RunConfig(實驗層 config)
│   ├── io/               # dataset CSV / sources / state recorder
│   └── runtime/          # run_clip()(main/batch 共用)
├── mmaction2/evaluation/ # 評估指標計算
├── scripts/              # build_dataset_csv.py
├── configs/              # run.example.yaml / neck_p4.yaml
├── external_camera/ , models/ , weight/   # 影片 / GT / ONNX 權重(未進版控)
└── docs/superpowers/     # 設計 spec 與實作計畫
```

---

## 8. 已知限制 (Known issues)

驗證與 code review 找到的真實行為。#1–#4 已於 Phase B 修正;#5 可接受;#6 待清理:

| # | 位置 | 現象 | 狀態 |
|---|---|---|---|
| 1 | [src/runtime/clip_runner.py](src/runtime/clip_runner.py) | cv2 開檔失敗曾靜默 `return None`,批次層會靜默跳過 | ✅ Phase B:`run_clip()` 改丟 `ClipOpenError`,批次記入 `skipped.json` |
| 2 | [batch_infer_phash.py](batch_infer_phash.py) | 舊 phash 線曾在 module top-level 執行,無 `__main__` 守衛 | ✅ Phase B:包進 `main()` + `__main__` 守衛 |
| 3 | [batch_infer.py](batch_infer.py) | 資料集曾硬編碼 8 個 `external_camera` 子資料夾 | ✅ Phase B:由 `run.yaml` 的 `source.dataset` 決定 |
| 4 | [main.py](main.py) `-f` | 舊版 `-f` 失效,實讀 `VIDEO_FILES_TO_TEST` | ✅ Phase B:`-f` 覆蓋 config source |
| 5 | [src/io/state_recorder.py](src/io/state_recorder.py) | 記錄器的 `DatasetRow` 無 `score` 欄,`score` 不持久化 | 可接受:CSV schema 本就不含 score,回灌時視為 1.0 |
| 6 | [eval_dashboard.py](eval_dashboard.py) / [visual_results.py](visual_results.py) | 沒給 `--config` 也沒給 `--gt/--pd` 時會回退寫死預設 | 待清理:請一律帶 `--config`(§1.3) |
