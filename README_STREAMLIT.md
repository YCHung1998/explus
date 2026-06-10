# Streamlit 互動式偵測系統使用說明

## 概述

這個 Streamlit 應用提供了完整的互動式動態偵測系統，支援即時參數調整和視覺化。

## 功能特點

### 1. 多種輸入源支援
- **Camera**: 即時攝影機輸入
- **Selected Video**: 從指定目錄選擇影片檔案
- **Selected Image**: 從 `images/` 目錄選擇單張圖片

### 2. 完整的參數調整面板

#### Adaptive Decision 參數
- **Base Sensitivity**: 基礎敏感度閾值 (0-200)
- **Max Sensitivity**: 最大敏感度閾值 (0-200)
- **Score Threshold**: Block-based 分數閾值 (0-100)
- **Active Ratio Threshold**: 活躍區域比例閾值 (0-1)
- **Boundary**: X/Y 軸邊界區域選擇

#### Texture 參數
- **Relative Threshold**: LBSP 相對強度閾值 (0.01-1.0)
- **Hamming Distance Threshold**: 紋理比較的 Hamming 距離閾值 (1-16)

#### Background 參數
- **Background EMA Alpha**: 背景更新的指數移動平均係數 (0.01-1.0)

#### Signal 參數
- **Stable Hold Time**: 維持穩定狀態的時間 (秒)
- **Unstable Hold Time**: 維持不穩定狀態的時間 (秒)

#### Trigger 參數
- **Trigger Mode**: 選擇驗證模式 (yolo / ema_phash)
- **NVCC Threshold**: ZNCC 相似度閾值 (YOLO 模式)
- **EMA Threshold**: EMA phash 距離閾值 (ema_phash 模式)

### 3. 即時視覺化

#### 主要顯示區域
- **Live Frame**: 即時影像顯示，可選擇不同視覺化模式
- **Status**: 當前狀態顯示 (Stable/Unstable/Trigger)
- **Metrics**: 即時數值指標

#### 圖表顯示
- **Event Chart**: 觸發事件和不穩定事件的時間序列
- **Distance Chart**: 原始距離和 EMA 距離的變化
- **Score Chart**: 最大分數的變化趨勢

#### 熱力圖
- **Score Map Heatmap**: Block-based 分數分佈
- **Active Mask Heatmap**: 活躍遮罩區域

### 4. 視覺化模式
- **Original Frame**: 原始影像
- **Score Map Overlay**: 分數地圖疊加顯示
- **Active Mask Overlay**: 活躍遮罩疊加顯示

## 使用步驟

1. **啟動應用**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **調整參數**
   - 在左側邊欄調整各項參數
   - 點擊 "Save & Start" 開始處理

3. **觀察結果**
   - 主畫面會即時顯示處理結果
   - 圖表和熱力圖會自動更新
   - 當觸發事件發生時會顯示通知

## 參數調整建議

### 初始設定
- 建議從預設值開始
- 根據實際場景逐步調整

### 敏感度調整
- **Base Sensitivity**: 降低可提高敏感度（更容易偵測到動態）
- **Score Threshold**: 提高可減少誤報

### 邊界設定
- 使用 X/Y 軸邊界選擇器來限制偵測區域
- 可有效過濾邊緣區域的干擾

### 紋理參數
- **Relative Threshold**: 較低值對光照變化更敏感
- **Hamming Distance**: 較高值允許更大的紋理差異

## 技術架構

### 模組化設計
- 保持 `main_v2.py` 的模組化架構
- 使用 `DetectionPipeline` 作為 Facade
- 所有模組職責清晰分離

### 參數管理
- 使用 `PipelineConfig` 統一管理參數
- 支援即時參數調整（透過 `runtime_params`）
- 參數分層組織，避免混亂

### 歷史記錄
- 自動維護時間序列資料
- 支援可配置的歷史大小
- 自動截斷過長歷史

## 注意事項

1. **模型檔案**: 確保 `models/best_vis_with_8400_3.onnx` 存在
2. **輸入源**: 選擇的輸入源必須可訪問
3. **效能**: 較高的 FPS 和歷史大小會影響效能
4. **參數調整**: 即時調整參數會立即生效，無需重啟

## 故障排除

### 無法開啟攝影機
- 檢查攝影機 ID 是否正確
- 確認攝影機未被其他程式使用

### 影片無法載入
- 檢查影片路徑是否正確
- 確認影片格式是否支援

### 效能問題
- 降低 FPS 設定
- 減少歷史大小
- 縮小處理影像尺寸

## 未來改進

- [ ] 支援配置保存/載入
- [ ] 批次處理多個影片
- [ ] 匯出處理結果
- [ ] 更詳細的統計報告
- [ ] 參數自動優化建議
