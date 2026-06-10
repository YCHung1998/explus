import json
import os
import sys
from collections import defaultdict
from flask import Flask, render_template_string, send_from_directory
import argparse


def get_data_from_json(raw_data, is_gt=True):
    """
    從 ActivityNet 格式的 JSON 結構中提取區段數據，並安全處理 None 值。
    """
    videos = raw_data.get("database", raw_data.get("results", raw_data))
    all_segments = defaultdict(list)

    # 用來記錄每個 video_id 的路徑（假設 GT 數據中包含路徑資訊）
    video_paths = {}

    for video_id, video_info in videos.items():
        if video_id == 'version':
            continue
        # GT: annotations; PD: results/annotations
        if isinstance(video_info, dict) and "annotations" in video_info:
            segments = video_info.get("annotations", video_info)
        else:
            segments = video_info

        # 如果是 GT 數據，嘗試在區段層面獲取路徑，並更新到 video_paths 字典
        if is_gt:
            # 優先使用區段內的 video_local_path
            # print("video_info", video_info)
            path_in_segment = video_info.get("video_local_path", '')
            if path_in_segment:
                current_video_path = path_in_segment
                video_paths[video_id] = current_video_path

        for seg in segments:
            segment_times = seg.get("segment")
            # interval
            if segment_times and len(segment_times) == 2:
                # 確保 label 不是 None
                label_val = seg.get("label", "Unknown")
                if label_val is None:
                    label_val = "Unknown"

                data = {
                    "segment": tuple(segment_times),
                    "label": label_val,
                    "video_id": video_id,
                }

                # XXX DEBUG
                # if is_gt and label_val==2 and video_id=='Viscovery_Bread_DemoRoom_20251107_110731':
                #     print("[[[DEBUG]]]")
                #     print("[[[DEBUG]]]")
                #     print("[[[DEBUG]]]")
                #     print(data)
                #     print("[[[DEBUG]]]")
                #     print("[[[DEBUG]]]")
                #     print("[[[DEBUG]]]")

                if is_gt and seg.get('video', None):
                    data['duration'] = seg['video']['duration']
                if not is_gt:  # prediction
                    # 確保 score 不是 None
                    score_val = seg.get("score", 0.0)
                    if score_val is None:
                        score_val = 0.0
                    data["score"] = score_val
                all_segments[video_id].append(data)
    # 返回所有區間數據和影片路徑
    return all_segments, video_paths


def structure_data_for_javascript(gt_data, pd_data, video_paths):
    """
    將 Python 字典轉換為 JavaScript MOCK_DATA 所需的結構。
    並計算每個影片的最大時間。
    """
    structured_data = {}

    # 獲取所有影片 ID
    all_video_ids = set(gt_data.keys()) | set(pd_data.keys())

    for video_id in sorted(list(all_video_ids)):
        gt_list = []
        pd_list = []
        max_t = 0.0

        # 從 video_paths 字典中獲取路徑
        video_path = video_paths.get(video_id, 'N/A: Path not found in GT data')

        # 處理 GT 數據
        for seg in gt_data.get(video_id, []):
            start, end = seg['segment']
            gt_list.append([start, end, seg['label']])
            max_t = max(max_t, end)

        # 處理 PD 數據
        for seg in pd_data.get(video_id, []):
            start, end = seg['segment']
            # 注意 PD 列表比 GT 列表多了一個 Score
            pd_list.append([start, end, seg['score'], seg['label']])
            max_t = max(max_t, end)

        # 只有在有區間數據時才寫入
        if max_t > 0.0:
            structured_data[video_id] = {
                # 影片總長度增加 5% 作為緩衝，讓圖表更美觀
                'max_time': max_t * 1.05,
                'gt': gt_list,
                'pd': pd_list,
                'video_path': video_path
            }
        elif video_path != 'N/A: Path not found in GT data':
            # 即使沒有區間，如果有路徑也記錄下來
            structured_data[video_id] = {
                'max_time': 0.0,
                'gt': [],
                'pd': [],
                'video_path': video_path
            }

    return structured_data


# --- 影片路徑處理：找出根目錄 ---
def find_common_video_root(structured_data):
    """
    從所有影片的絕對路徑中，嘗試找到一個共同的根目錄。
    這個根目錄將用於 Flask 的靜態檔案服務。
    """
    paths = [data['video_path'] for data in structured_data.values() if
             data['video_path'] and not data['video_path'].startswith('N/A')]

    if not paths:
        return None, None

    # 取得第一個路徑的目錄作為起點
    base_dir = os.path.dirname(paths[0])

    # 找出所有路徑的共同前綴
    while not all(path.startswith(base_dir) for path in paths) and len(base_dir) > 1:
        base_dir = os.path.dirname(base_dir)

    # 確保路徑是絕對路徑
    base_dir = os.path.abspath(base_dir)

    # 根目錄下的路徑必須是影片的目錄 (多層判斷)
    if os.path.isdir(paths[0]): # 如果路徑本身是資料夾，則使用父目錄
         video_root = os.path.dirname(paths[0])
    else: # 否則使用路徑的目錄
         video_root = os.path.dirname(paths[0])

    # 簡化處理：為了演示，我們假設所有影片都在第一個影片路徑的父目錄中。
    # 您的實際應用可能需要更健壯的邏輯。
    video_root = os.path.dirname(paths[0])

    # 返回路徑的根目錄和所有影片檔案的名稱
    video_filenames = {os.path.basename(p): p for p in paths}

    return video_root, video_filenames


def get_timeline_html_template(data_json_string, video_dir_path='', name=''):
    """
    完整版視覺化模板：
    1. 支援多 Label 分行。
    2. 補回所有 Meta Data (Duration, Score, Status)。
    3. 修正 Python f-string 轉義 ({{ }})。
    """
    html_template = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>時序區間視覺化 - 完整版</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        #timeline-chart {{ width: 100%; min-height: 400px; }}
        #video-player-container {{ width: 100%; aspect-ratio: 16 / 9; background-color: #000; }}
        .plotly-graph-div {{ width: 100%; height: 100%; }}
    </style>
</head>
<body class="bg-gray-100 p-6">
    <div class="max-w-7xl mx-auto flex space-x-6">
        <div class="w-1/2 bg-white p-6 rounded-xl shadow-lg">
            <h2 class="text-xl font-bold mb-2">🎥 影片同步檢視</h2>
            <p class="text-xs text-gray-400 mb-4">File Path: <span id="current-video-path" class="break-all">{video_dir_path}</span></p>
            <div id="video-player-container">
                <video id="review-video" controls width="100%" height="100%" class="rounded-lg shadow-inner"></video>
            </div>
            <div id="video-status" class="mt-4 p-3 bg-gray-50 rounded border text-sm text-gray-600 text-center">
                點擊右側圖表區間開始分析
            </div>
        </div>

        <div class="w-1/2 bg-white p-6 rounded-xl shadow-lg">
            <h1 class="text-2xl font-bold mb-2">時序定位分析</h1>
            <p class="text-xs text-gray-400 mb-6">File: {name}</p>
            
            <select id="video-selector" class="w-full p-3 border rounded-lg mb-6 shadow-sm bg-white focus:ring-2 focus:ring-blue-500"></select>
            
            <div id="timeline-chart-container" class="border rounded-lg bg-white overflow-hidden">
                <div id="timeline-chart"></div>
            </div>
        </div>
    </div>

    <script>
        const ALL_VIDEO_DATA = {data_json_string};
        const videoElement = document.getElementById('review-video');

        function calculateTIoU(segment1, segment2) {{
            const [s1, e1] = segment1;
            const [s2, e2] = segment2;
            const intersection = Math.max(0, Math.min(e1, e2) - Math.max(s1, s2));
            const union = (e1 - s1) + (e2 - s2) - intersection;
            return union === 0 ? 0 : intersection / union;
        }}

        function drawTimeline(videoId) {{
            const data = ALL_VIDEO_DATA[videoId];
            if (!data) return;

            // 更新 UI 顯示的完整路徑
            document.getElementById('current-video-path').textContent = data.video_path;

            // 提取所有 PD 標籤用於分行
            const pdLabels = [...new Set(data.pd.map(seg => seg[3]))].sort();
            const yTicks = ['gt', ...pdLabels];
            const yTickTexts = ['GT (真實)', ...pdLabels.map(l => `PD: Label ${{l}}`)];

            const gtTraces = [];
            const pdTraces = [];

            // 定義顏色表 (為不同的 Label 分配固定顏色，用於 GT)
            const labelColors = {{
                "1": "rgba(54, 162, 235, 0.7)",  // 藍色
                "2": "rgba(153, 102, 255, 0.7)", // 紫色
                "3": "rgba(255, 50, 50, 0.7)",  // other色
                "4": "rgba(255, 159, 64, 0.7)",  // 橘色
                "5": "rgba(255, 159, 64, 0.7)",  // 橘色
                "6": "rgba(255, 159, 64, 0.7)",  // 橘色
                "default": "rgba(144, 238, 144, 0.7)" // 原始綠色
            }};

            // 2. 繪製 GT (同一行，不同色)
            data.gt.forEach(seg => {{
                const [start, end, label] = seg;
                const duration = (end - start).toFixed(2);
                // 根據 label 取得對應顏色
                const color = labelColors[label] || labelColors["unknown"];

                gtTraces.push({{
                    x: [start, end],
                    y: ['gt', 'gt'], // 固定在同一行
                    mode: 'lines',
                    line: {{ width: 30, color: color }},
                    customdata: [start, end, label, 'GT'],
                    hoverinfo: 'text',
                    text: `<b>GT (真實)</b><br>標籤: ${{label}}<br>時間: ${{start}}s - ${{end}}s<br>長度: ${{duration}}s`
                }});
            }});


            // 2. PD Traces (補回 Score, Status, tIoU)
            data.pd.forEach(pdSeg => {{
                const [start, end, score, label] = pdSeg;
                const duration = (end - start).toFixed(2);
                let bestTIoU = 0;
                data.gt.forEach(gtSeg => {{
                    const tiou = calculateTIoU([start, end], [gtSeg[0], gtSeg[1]]);
                    if (tiou > bestTIoU) bestTIoU = tiou;
                }});

                // 狀態與顏色邏輯
                let color, status;
                if (bestTIoU >= 0.5) {{
                    color = 'rgba(255, 100, 100, 0.8)'; status = 'TP (正確檢測)';
                }} else if (bestTIoU > 0.0) {{
                    color = 'rgba(255, 165, 0, 0.9)'; status = 'Low tIoU (定位錯誤)';
                }} else {{
                    color = 'rgba(150, 0, 0, 0.8)'; status = 'FP (誤報)';
                }}

                pdTraces.push({{
                    x: [start, end],
                    y: [label, label],
                    mode: 'lines',
                    line: {{ width: 25, color: color }},
                    customdata: [start, end, label, 'PD'],
                    hoverinfo: 'text',
                    text: `<b>PD (預測)</b><br>狀態: ${{status}}<br>標籤: ${{label}}<br>分數: ${{score.toFixed(3)}}<br>時間: ${{start.toFixed(2)}}s - ${{end.toFixed(2)}}s<br>長度: ${{duration}}s<br>tIoU: ${{bestTIoU.toFixed(4)}}`
                }});
            }});

            const layout = {{
                title: `影片分析: ${{videoId}}`,
                height: 350 + (pdLabels.length * 60),
                xaxis: {{ title: '時間 (秒)', range: [0, data.max_time], gridcolor: '#eee' }},
                yaxis: {{
                    tickvals: yTicks,
                    ticktext: yTickTexts,
                    automargin: true,
                    fixedrange: true
                }},
                showlegend: false,
                margin: {{ l: 150, r: 50, t: 100, b: 100 }},
                hovermode: 'closest'
            }};

            Plotly.newPlot('timeline-chart', [...gtTraces, ...pdTraces], layout, {{responsive: true}});

            // 點擊事件
            document.getElementById('timeline-chart').on('plotly_click', (d) => {{
                if (d.points.length > 0) {{
                    const [start, end, label, type] = d.points[0].data.customdata;
                    videoElement.currentTime = start;
                    videoElement.play();
                    document.getElementById('video-status').innerHTML = 
                        `▶ 正在播放 <span class="font-bold text-blue-600">${{type}} [${{label}}]</span> (從 ${{start.toFixed(2)}}s 開始)`;
                }}
            }});
            
            updateVideoPlayer(data.video_path);
        }}

        function updateVideoPlayer(localPath) {{
            if (!localPath || localPath.startsWith('N/A')) return;
            const filename = localPath.split('/').pop();
            videoElement.src = `/videos/${{filename}}`;
            videoElement.load();
        }}

        document.addEventListener('DOMContentLoaded', () => {{
            const selector = document.getElementById('video-selector');
            const ids = Object.keys(ALL_VIDEO_DATA);
            ids.forEach(id => {{
                const opt = document.createElement('option');
                opt.value = id; opt.textContent = id;
                selector.appendChild(opt);
            }});
            selector.addEventListener('change', e => drawTimeline(e.target.value));
            if (ids.length > 0) drawTimeline(ids[0]);
        }});
    </script>
</body>
</html>
"""
    return html_template
# --- 主程序執行 (使用 Flask) ---


def create_app(gt_path, pd_path):
    # 1. 載入原始數據
    try:
        with open(gt_path, 'r', encoding='utf-8') as f: gt_raw = json.load(f)
        with open(pd_path, 'r', encoding='utf-8') as f: pd_raw = json.load(f)
    except Exception as e:
        print(f"載入 JSON 檔案失敗。錯誤: {e}")
        sys.exit(1)

    # 2. 數據解析與結構化
    gt_data, video_paths = get_data_from_json(gt_raw, is_gt=True)
    pd_data, _ = get_data_from_json(pd_raw, is_gt=False)
    structured_data = structure_data_for_javascript(gt_data, pd_data, video_paths)

    if not structured_data:
        print("未找到任何有效的影片數據進行視覺化。")
        sys.exit(1)
    # 3. 找出所有影片檔案的映射 (用於 Flask 靜態服務)
    VIDEO_ROOT_DIR, VIDEO_FILENAME_MAP = find_common_video_root(structured_data)

    if not VIDEO_FILENAME_MAP:
         print(f"⚠️ 警告: 未能從 JSON 中提取到任何影片路徑。")
    else:
        print(f"🎥 捕獲了 {len(VIDEO_FILENAME_MAP)} 個影片路徑。")

    # 4. 創建 Flask 應用程式
    app = Flask(__name__)
    data_json_string = json.dumps(structured_data, indent=4)
    html_template = get_timeline_html_template(data_json_string, video_dir_path=VIDEO_ROOT_DIR, name=gt_path)

    @app.route('/')
    def index():
        """提供視覺化 HTML 頁面"""
        return render_template_string(html_template)

    @app.route('/videos/<filename>')
    def serve_video(filename):
        """提供影片檔案的路由 (支援多資料夾動態尋找)"""
        # 1. 優先從映射表中查出絕對路徑
        full_path = VIDEO_FILENAME_MAP.get(filename)
        
        if full_path and os.path.exists(full_path):
            video_dir = os.path.dirname(full_path)
            return send_from_directory(video_dir, filename)
        
        # 2. 備案：若映射表無效，嘗試從 Common Root 找
        if VIDEO_ROOT_DIR and os.path.exists(os.path.join(VIDEO_ROOT_DIR, filename)):
            return send_from_directory(VIDEO_ROOT_DIR, filename)
            
        return f"Video {filename} not found in mapped paths or root.", 404

    return app

# python visual_results.py -eg
if __name__ == "__main__":
    OUTPUT_DIR = 'output/dummy'

    OUTPUT_DIR = 'output/BlockBased'
    OUTPUT_DIR = 'output/PixelBased'
    OUTPUT_DIR = 'output/PixelBased_bdry_v2'
    OUTPUT_DIR = 'output/PixelBased_bdry_1230'
    OUTPUT_DIR = 'output/PixelBased_bdry_1231'
    OUTPUT_DIR = 'output/PixelBased_bdry_1231_revised'
    # OUTPUT_DIR = 'output/AdaptiveBlockBased'
    # OUTPUT_DIR = 'output/BlockBased_bdry'
    # OUTPUT_DIR = 'output/phash'
    # OUTPUT_DIR = 'output/phash_bdry'
    # OUTPUT_DIR = 'output/yolo'
    OUTPUT_DIR = 'output/PixelBased_bdry_0106'
    OUTPUT_DIR = 'output/Block_0106'
    OUTPUT_DIR = 'output/Block_0112_yolo'
    OUTPUT_DIR = 'output/Block_0119_yolo'
    # OUTPUT_DIR = 'output/Block_0121_cus_yolo'
    # OUTPUT_DIR = 'output/Block_0121_cus_backbone_yolo'
    # OUTPUT_DIR = 'output/Block_0122_yolo'
    # OUTPUT_DIR = 'output/Block_0112_ema_phash'
    # OUTPUT_DIR = 'output/phash_0112'
    # OUTPUT_DIR = 'output/Block_TEST_yolo'
    # OUTPUT_DIR = 'output/Block_0123_mir4_yolo'
    OUTPUT_DIR = 'output/Block_m0_Neck_P4_yolo'
    OUTPUT_DIR = 'output/Block_m0_Backbone_fusion_yolo'

    PORT = "5001"  # 5000-5005
    parser = argparse.ArgumentParser(description="視覺化 Temporal Action Localization 的 Ground Truth 和預測結果。")
    parser.add_argument('--gt', type=str,
                        default="/Users/eason.hung/Documents/Projects/explus/output/ground_truth/data.json",
                        help="Ground Truth JSON 檔案路徑。")
    parser.add_argument('--pd', type=str,
                        default="/Users/eason.hung/Documents/Projects/explus/output/predictions/merge_data.json",
                        help="Prediction JSON 檔案路徑。")
    parser.add_argument('--port', type=int, default=5000, help="Flask 服務器的端口號。")
    parser.add_argument('--config', type=str, default=None,
                        help="run.yaml; derives gt/pd from output.dir")
    parser.add_argument(
        '-eg',
        '--example',
        action='store_true',
        default=False,
    )
    args = parser.parse_args()
    if args.example:
        print(f"""
python visual_results.py \\
    --gt /Users/eason.hung/Documents/Projects/explus/{OUTPUT_DIR}/ground_truth/data.json \\
    --pd /Users/eason.hung/Documents/Projects/explus/{OUTPUT_DIR}/predictions/merge_data.json \\
    --port {PORT}
    """)
        exit()

    # 使用示範數據 (請確保路徑正確或替換為您的數據)
    if 'demo' in args: # 假設您運行的是 demo 模式
        GT = "/Users/eason.hung/Documents/Projects/explus/mmaction2/demo/ground_truth_diff_demo3.json"
        PD = "/Users/eason.hung/Documents/Projects/explus/mmaction2/demo/prediction_diff_demo3.json"
    else:
        # --config derives gt/pd from output.dir; explicit --gt/--pd preserved.
        GT = args.gt
        PD = args.pd
        if args.config:
            from src.config.run_config import RunConfig
            cfg = RunConfig.from_yaml(args.config)
            d_gt, d_pd = cfg.eval_paths()
            GT = d_gt
            PD = d_pd

    app = create_app(GT, PD)

    # 啟動 Flask 服務器
    print("-------------------------------------------------------")
    print(f"🚀 服務器啟動中... 請在瀏覽器中開啟: http://127.0.0.1:{args.port}")
    print("-------------------------------------------------------")
    app.run(debug=True, port=args.port)
