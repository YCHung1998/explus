import json
import os
import sys
import numpy as np
from flask import Flask, render_template_string, send_from_directory, jsonify, send_file, request
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from mmaction2.evaluation.eval_custom import AdvancedDualEvaluator
    from constants import LABEL_MAP, LABEL_NAME2ID, LabelMap
except ImportError:
    print("Error: Could not import project modules. Ensure you're in the project root.")
    sys.exit(1)

# --- Configuration ---
DEFAULT_PORT = 5005
D_DIR = "output/Block_m0_Backbone_fusion_yolo"
# D_DIR = "output/Block_m0_Neck_P4_yolo"

D_GT = os.path.join(D_DIR, "ground_truth/data.json")
D_PD = os.path.join(D_DIR, "predictions/merge_data.json")

# --- HTML TEMPLATE (Includes CSS and JS) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>HS-CODS Evaluation Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; }
        .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 0.75rem; padding: 1.5rem; }
        .metric-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-left: 4px solid #38bdf8; }
        .table-row:hover { background-color: #334155; cursor: pointer; }
        #timeline-chart { width: 100%; height: 450px; }
        video { border-radius: 0.5rem; background: #000; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5); }
        .btn-back { color: #94a3b8; transition: color 0.2s; }
        .btn-back:hover { color: #f8fafc; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }
    </style>
</head>
<body class="p-6">
    <div id="app" class="max-w-[1600px] mx-auto">
        <!-- HEADER -->
        <header class="flex justify-between items-center mb-6">
            <div>
                <h1 class="text-4xl font-bold tracking-tight text-white mb-1">
                    📊 <span class="bg-gradient-to-r from-blue-400 to-teal-400 bg-clip-text text-transparent">HS-CODS</span> Evaluation
                </h1>
                <p class="text-slate-400 text-sm">Interactive Analysis Dashboard v2.1 (Flask Engine)</p>
            </div>
            <div id="back-container" class="hidden">
                <button onclick="showView('summary')" class="btn-back flex items-center space-x-2 font-semibold">
                    <span>← 返回總表</span>
                </button>
            </div>
        </header>

        <!-- PATH CONFIG (New) -->
        <div id="path-config" class="card mb-8 !py-4 flex flex-col gap-4">
            <div class="flex flex-col md:flex-row gap-6 w-full items-end">
                <div class="flex-[3]">
                    <label class="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Ground Truth JSON Path</label>
                    <input type="text" id="gt-path-input" class="w-full bg-slate-900 border border-slate-700 rounded px-4 py-2.5 text-sm text-blue-300 font-mono focus:outline-none focus:border-blue-500 shadow-inner" placeholder="Enter GT path...">
                </div>
                <div class="flex-[3]">
                    <label class="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Prediction JSON Path</label>
                    <input type="text" id="pd-path-input" class="w-full bg-slate-900 border border-slate-700 rounded px-4 py-2.5 text-sm text-amber-300 font-mono focus:outline-none focus:border-blue-500 shadow-inner" placeholder="Enter PD path...">
                </div>
                <div class="flex-none">
                    <button onclick="updateData()" class="bg-blue-600 hover:bg-blue-500 text-white px-8 py-2.5 rounded font-bold text-sm transition-all shadow-lg active:scale-95 whitespace-nowrap">
                        🔄 執行重新評估 (Update)
                    </button>
                </div>
            </div>
        </div>

        <!-- VIEW 1: SUMMARY -->
        <div id="summary-view" class="space-y-8">
            <!-- OVERALL METRICS -->
            <!-- OVERALL METRICS -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                <!-- Group 1: mAP Results -->
                <div class="card metric-card">
                    <p class="text-slate-400 text-xs font-semibold mb-1 uppercase tracking-wider">STAGE 1 AVG MAP</p>
                    <p class="text-2xl font-bold text-white/90" id="s1-map">0.0000</p>
                </div>
                <div class="card metric-card border-l-emerald-500">
                    <p class="text-slate-400 text-xs font-semibold mb-1 uppercase tracking-wider">STAGE 2 AVG MAP</p>
                    <p class="text-2xl font-bold text-white/90" id="s2-map">0.0000</p>
                </div>
                <div class="card metric-card border-l-amber-500">
                    <p class="text-slate-400 text-xs font-semibold mb-1 uppercase tracking-wider">PRECISION (S2)</p>
                    <p class="text-2xl font-bold text-white/90" id="s2-prec">0.0%</p>
                </div>
                <div class="card metric-card border-l-purple-500">
                    <p class="text-slate-400 text-xs font-semibold mb-1 uppercase tracking-wider">RECALL (S2)</p>
                    <p class="text-2xl font-bold text-white/90" id="s2-recall">0.0%</p>
                </div>
            </div>

            <!-- GLOBAL TOTALS (New) -->
            <div id="global-totals" class="grid grid-cols-2 lg:grid-cols-5 gap-4">
                <!-- Totals will be injected by JS -->
            </div>

            <!-- TABS -->
            <div class="flex space-x-4 border-b border-slate-700 pb-px">
                <button onclick="switchTab('stage2')" id="tab-stage2" class="px-6 py-3 font-bold border-b-2 border-blue-500 text-blue-400">🎯 STAGE 2: DECISION</button>
                <button onclick="switchTab('stage1')" id="tab-stage1" class="px-6 py-3 font-bold border-slate-700 text-slate-500 hover:text-slate-300">📈 STAGE 1: POSITIONING</button>
            </div>

            <!-- STAGE 2 TABLE -->
            <div id="table-stage2" class="card overflow-hidden !p-0">
                <div class="overflow-x-auto">
                    <table class="w-full text-left">
                        <thead class="bg-slate-800 text-slate-400 text-sm uppercase tracking-widest font-bold">
                            <tr>
                                <th class="p-4 cursor-pointer hover:text-white" onclick="sortData('id', 'stage2')">Video ID ↕</th>
                                <th class="p-4 text-center cursor-pointer hover:text-white" onclick="sortData('errors', 'stage2')">Errors ↕</th>
                                <th class="p-4 text-center cursor-pointer hover:text-white text-emerald-400" onclick="sortData('just_fit', 'stage2')">Just Fit ↑</th>
                                <th class="p-4 text-center cursor-pointer hover:text-white text-orange-400" onclick="sortData('overcount', 'stage2')">Overcount ↓</th>
                                <th class="p-4 text-center cursor-pointer hover:text-white text-rose-400" onclick="sortData('undercount', 'stage2')">Undercount ↓</th>
                                <th class="p-4 text-center cursor-pointer hover:text-white text-rose-500" onclick="sortData('false_alarm', 'stage2')">False Alarm ↓</th>
                                <th class="p-4 text-center cursor-pointer hover:text-white text-blue-400/70 font-normal" onclick="sortData('s2_mAP', 'stage2')">mAP (F1) ↕</th>
                            </tr>
                        </thead>
                        <tbody id="s2-body" class="divide-y divide-slate-700">
                            <!-- Rows injected by JS -->
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- STAGE 1 TABLE -->
            <div id="table-stage1" class="card overflow-hidden !p-0 hidden">
                <div class="overflow-x-auto">
                    <table class="w-full text-left">
                        <thead class="bg-slate-800 text-slate-400 text-sm uppercase tracking-widest font-bold">
                            <tr>
                                <th class="p-4 cursor-pointer hover:text-white" onclick="sortData('id', 'stage1')">Video ID ↕</th>
                                <th class="p-4 text-center cursor-pointer hover:text-white" onclick="sortData('s1_faults', 'stage1')">Faults ↕</th>
                                <th class="p-4 text-center cursor-pointer hover:text-white text-emerald-400" onclick="sortData('match', 'stage1')">Match ↑</th>
                                <th class="p-4 text-center cursor-pointer hover:text-white text-blue-400" onclick="sortData('not_tight', 'stage1')">Not Tight ↓</th>
                                <th class="p-4 text-center cursor-pointer hover:text-white text-rose-400" onclick="sortData('gt_orphan', 'stage1')">GT Orphan ↓</th>
                                <th class="p-4 text-center cursor-pointer hover:text-white text-rose-500" onclick="sortData('pd_orphan', 'stage1')">PD Orphan ↓</th>
                                <th class="p-4 text-center cursor-pointer hover:text-white text-slate-500 font-normal" onclick="sortData('s1_mAP', 'stage1')">mAP (F1) ↕</th>
                            </tr>
                        </thead>
                        <tbody id="s1-body" class="divide-y divide-slate-700">
                            <!-- Rows injected by JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- VIEW 2: DETAIL -->
        <div id="detail-view" class="hidden">
            <div class="grid grid-cols-3 gap-8">
                <!-- LEFT: Chart -->
                <div class="col-span-2 space-y-6">
                    <div class="card">
                        <h2 class="text-xl font-bold mb-4 flex items-center">
                            <span class="mr-2">🕒</span> 互動時序分析
                        </h2>
                        <div id="timeline-chart"></div>
                        <p class="text-slate-500 text-xs mt-4">提示: 點擊彩色區段，影片會立即跳轉到該起始點。mAP (ActivityNet) 僅供參考，請優先關注計數指標。</p>
                    </div>
                    
                    <div class="grid grid-cols-4 gap-4">
                        <div class="card !p-4 bg-slate-800/50">
                            <p class="text-xs text-slate-500 mb-1">JUST FIT</p>
                            <p id="det-fit" class="text-xl font-bold">0</p>
                        </div>
                        <div class="card !p-4 bg-slate-800/50">
                            <p class="text-xs text-slate-500 mb-1">OVER</p>
                            <p id="det-over" class="text-xl font-bold">0</p>
                        </div>
                        <div class="card !p-4 bg-slate-800/50">
                            <p class="text-xs text-slate-500 mb-1">UNDER</p>
                            <p id="det-under" class="text-xl font-bold text-rose-400">0</p>
                        </div>
                        <div class="card !p-4 bg-slate-800/50">
                            <p class="text-xs text-slate-500 mb-1">FA</p>
                            <p id="det-fa" class="text-xl font-bold text-rose-500">0</p>
                        </div>
                    </div>
                </div>

                <!-- RIGHT: Video & List -->
                <div class="space-y-6">
                    <div class="card">
                        <h2 class="text-lg font-bold mb-4">🎥 影音連動檢索</h2>
                        <video id="v-player" controls class="w-full aspect-video mb-4"></video>
                        <p class="text-slate-400 text-xs break-all" id="det-path"></p>
                    </div>
                    
                    <div class="card !p-0 max-h-[400px] overflow-y-auto">
                        <div class="p-4 border-b border-slate-700 bg-slate-800/50 sticky top-0">
                            <h3 class="text-sm font-bold uppercase tracking-widest text-slate-400">事件清單</h3>
                        </div>
                        <div id="event-list" class="divide-y divide-slate-700">
                             <!-- Event buttons injected by JS -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const DATA = {{ dashboard_data | tojson }};
        const GT_COLOR_MAP = {{ color_map | tojson }};
        const LABEL_MAP = {{ label_map | tojson }};

        function showView(view) {
            document.getElementById('summary-view').classList.toggle('hidden', view !== 'summary');
            document.getElementById('path-config').classList.toggle('hidden', view !== 'summary'); // Hide on detail
            document.getElementById('detail-view').classList.toggle('hidden', view !== 'detail');
            document.getElementById('back-container').classList.toggle('hidden', view !== 'detail');
        }

        let CURRENT_STAGE = 'stage2';
        let SORT_COL = 'errors';
        let SORT_DIR = -1; // -1: desc, 1: asc

        function initSummary() {
            // Set input values
            document.getElementById('gt-path-input').value = DATA.gt_path;
            document.getElementById('pd-path-input').value = DATA.pd_path;

            document.getElementById('s1-map').innerText = DATA.summary.s1.avg_mAP.toFixed(5);
            document.getElementById('s2-map').innerText = DATA.summary.s2.avg_mAP.toFixed(5);
            document.getElementById('s2-prec').innerText = (DATA.summary.s2.precision * 100).toFixed(1) + '%';
            document.getElementById('s2-recall').innerText = (DATA.summary.s2.recall * 100).toFixed(1) + '%';
            
            updateGlobalTotals();
            renderTables();
        }

        function updateGlobalTotals() {
            const container = document.getElementById('global-totals');
            if (CURRENT_STAGE === 'stage2') {
                const s = DATA.summary.s2;
                container.innerHTML = `
                    <div class="card !p-3 bg-slate-800/40 border-slate-700/50">
                        <p class="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Total Just Fit</p>
                        <p class="text-xl font-bold font-mono text-emerald-400">${s.just_fit}</p>
                    </div>
                    <div class="card !p-3 bg-slate-800/40 border-slate-700/50">
                        <p class="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Total Overcount</p>
                        <p class="text-xl font-bold font-mono text-orange-400">${s.overcount}</p>
                    </div>
                    <div class="card !p-3 bg-slate-800/40 border-slate-700/50">
                        <p class="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Total Undercount</p>
                        <p class="text-xl font-bold font-mono text-rose-400">${s.undercount}</p>
                    </div>
                    <div class="card !p-3 bg-slate-800/40 border-slate-700/50">
                        <p class="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Total False Alarm</p>
                        <p class="text-xl font-bold font-mono text-rose-600">${s.false_alarm}</p>
                    </div>
                    <div class="card !p-3 bg-slate-800/40 border-slate-700/50">
                        <p class="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Total Videos</p>
                        <p class="text-xl font-bold font-mono text-slate-400">${DATA.videos.length}</p>
                    </div>
                `;
            } else {
                const s = DATA.summary.s1;
                container.innerHTML = `
                    <div class="card !p-3 bg-slate-800/40 border-slate-700/50">
                        <p class="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Total Match</p>
                        <p class="text-xl font-bold font-mono text-emerald-400">${s.matched}</p>
                    </div>
                    <div class="card !p-3 bg-slate-800/40 border-slate-700/50">
                        <p class="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Total Not Tight</p>
                        <p class="text-xl font-bold font-mono text-blue-400">${s.not_tight}</p>
                    </div>
                    <div class="card !p-3 bg-slate-800/40 border-slate-700/50">
                        <p class="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Total GT Orphan</p>
                        <p class="text-xl font-bold font-mono text-rose-400">${s.gt_orphan}</p>
                    </div>
                    <div class="card !p-3 bg-slate-800/40 border-slate-700/50">
                        <p class="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Total PD Orphan</p>
                        <p class="text-xl font-bold font-mono text-rose-600">${s.pd_orphan}</p>
                    </div>
                    <div class="card !p-3 bg-slate-800/40 border-slate-700/50">
                        <p class="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Total Videos</p>
                        <p class="text-xl font-bold font-mono text-slate-400">${DATA.videos.length}</p>
                    </div>
                `;
            }
        }

        function sortData(col, stage) {
            if (SORT_COL === col) {
                SORT_DIR *= -1;
            } else {
                SORT_COL = col;
                SORT_DIR = -1;
            }
            renderTables();
        }

        function renderTables() {
            // Sort Stage 2
            const sortedVideos = [...DATA.videos].sort((a, b) => {
                let valA, valB;
                if (SORT_COL === 'errors') {
                    valA = a.s2.undercount + a.s2.false_alarm;
                    valB = b.s2.undercount + b.s2.false_alarm;
                } else if (SORT_COL === 'just_fit' || SORT_COL === 'overcount' || SORT_COL === 'undercount' || SORT_COL === 'false_alarm') {
                    valA = a.s2[SORT_COL];
                    valB = b.s2[SORT_COL];
                } else if (SORT_COL === 'match' || SORT_COL === 'not_tight' || SORT_COL === 'gt_orphan' || SORT_COL === 'pd_orphan') {
                    valA = a.s1_counts[SORT_COL];
                    valB = b.s1_counts[SORT_COL];
                } else {
                    valA = a[SORT_COL];
                    valB = b[SORT_COL];
                }

                if (typeof valA === 'string') return SORT_DIR * valA.localeCompare(valB);
                return SORT_DIR * (valA - valB);
            });

            // Populate Stage 2 Body
            const s2Rows = sortedVideos.map(v => {
                const s = v.s2;
                const errors = s.undercount + s.false_alarm;
                return `
                    <tr class="table-row group" onclick="loadDetail('${v.id}')">
                        <td class="p-4 font-medium text-slate-400 text-sm truncate max-w-[300px]">${v.id}</td>
                        <td class="p-4 text-center font-bold text-slate-200 text-xl font-mono">${errors}</td>
                        <td class="p-4 text-center font-bold text-emerald-400 text-2xl font-mono">${s.just_fit}</td>
                        <td class="p-4 text-center font-bold ${s.overcount > 0 ? 'text-orange-500' : 'text-slate-600'} text-2xl font-mono">${s.overcount}</td>
                        <td class="p-4 text-center font-bold ${s.undercount > 0 ? 'text-rose-500' : 'text-slate-600'} text-2xl font-mono">${s.undercount}</td>
                        <td class="p-4 text-center font-bold ${s.false_alarm > 0 ? 'text-rose-600' : 'text-slate-600'} text-2xl font-mono">${s.false_alarm}</td>
                        <td class="p-4 text-center text-slate-500 font-bold text-lg font-mono">${v.s2_mAP.toFixed(4)}</td>
                    </tr>
                `;
            }).join('');
            document.getElementById('s2-body').innerHTML = s2Rows;

            // Populate Stage 1 Body
            const s1Rows = sortedVideos.map(v => {
                const s = v.s1_counts;
                return `
                    <tr class="table-row group" onclick="loadDetail('${v.id}')">
                        <td class="p-4 font-medium text-slate-400 text-sm truncate max-w-[300px]">${v.id}</td>
                        <td class="p-4 text-center font-bold text-slate-200 text-xl font-mono">${v.s1_faults}</td>
                        <td class="p-4 text-center font-bold text-emerald-400 text-2xl font-mono">${s.match}</td>
                        <td class="p-4 text-center font-bold ${s.not_tight > 0 ? 'text-blue-400' : 'text-slate-600'} text-2xl font-mono">${s.not_tight}</td>
                        <td class="p-4 text-center font-bold ${s.gt_orphan > 0 ? 'text-rose-500' : 'text-slate-600'} text-2xl font-mono">${s.gt_orphan}</td>
                        <td class="p-4 text-center font-bold ${s.pd_orphan > 0 ? 'text-rose-600' : 'text-slate-600'} text-2xl font-mono">${s.pd_orphan}</td>
                        <td class="p-4 text-center text-slate-600 font-bold text-lg font-mono">${v.s1_mAP.toFixed(4)}</td>
                    </tr>
                `;
            }).join('');
            document.getElementById('s1-body').innerHTML = s1Rows;
        }

        function switchTab(stage) {
            CURRENT_STAGE = stage;
            SORT_COL = stage === 'stage2' ? 'errors' : 's1_faults';
            SORT_DIR = -1;

            document.getElementById('table-stage1').classList.toggle('hidden', stage !== 'stage1');
            document.getElementById('table-stage2').classList.toggle('hidden', stage !== 'stage2');
            
            const t1 = document.getElementById('tab-stage1');
            const t2 = document.getElementById('tab-stage2');
            
            if (stage === 'stage1') {
                t1.className = 'px-6 py-3 font-bold border-b-2 border-blue-500 text-blue-400';
                t2.className = 'px-6 py-3 font-bold border-slate-700 text-slate-500 hover:text-slate-300';
            } else {
                t2.className = 'px-6 py-3 font-bold border-b-2 border-blue-500 text-blue-400';
                t1.className = 'px-6 py-3 font-bold border-slate-700 text-slate-500 hover:text-slate-300';
            }
            updateGlobalTotals();
            renderTables();
        }

        function loadDetail(vid) {
            const videoData = DATA.videos.find(v => v.id === vid);
            if (!videoData) return;

            showView('detail');
            document.getElementById('det-fit').innerText = videoData.s2.just_fit;
            document.getElementById('det-over').innerText = videoData.s2.overcount;
            document.getElementById('det-under').innerText = videoData.s2.undercount;
            document.getElementById('det-fa').innerText = videoData.s2.false_alarm;
            document.getElementById('det-path').innerText = videoData.path;

            const player = document.getElementById('v-player');
            player.src = `/video/${encodeURIComponent(videoData.path)}`;
            player.load();

            renderChart(videoData);
            renderEventList(videoData);
        }

        function renderChart(v) {
            const traces = [];
            const yLines = ['GT'];
            
            // 1. GT Traces
            v.gt_segments.forEach((seg, i) => {
                const [start, end, label] = seg;
                const labelName = LABEL_MAP[label] || `L${label}`;
                const color = GT_COLOR_MAP[label] || 'rgba(150,150,150,0.5)';
                
                traces.push({
                    x: [start, end],
                    y: ['GT', 'GT'],
                    mode: 'lines',
                    line: { width: 35, color: color },
                    name: `GT: ${labelName}`,
                    customdata: start,
                    hoverinfo: 'text',
                    text: `<b>GT: ${labelName}</b><br>始: ${start.toFixed(2)}s<br>末: ${end.toFixed(2)}s`
                });
            });

            // 2. PD Traces
            v.pd_segments.forEach((seg, i) => {
                const [start, end, score, label] = seg;
                const rowName = `PD: ${label}`;
                if (!yLines.includes(rowName)) yLines.push(rowName);
                
                // Color based on match (pseudo-logic for MVP)
                const color = score > 0.5 ? 'rgba(56, 189, 248, 0.8)' : 'rgba(239, 68, 68, 0.8)';

                traces.push({
                    x: [start, end],
                    y: [rowName, rowName],
                    mode: 'lines',
                    line: { width: 25, color: color },
                    name: `PD: ${label}`,
                    customdata: start,
                    hoverinfo: 'text',
                    text: `<b>PD: ${label}</b><br>Score: ${score.toFixed(3)}<br>範圍: ${start.toFixed(2)}s - ${end.toFixed(2)}s`
                });
            });

            const maxT = Math.max(...traces.map(t => t.x[1]), 0) * 1.1;

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#94a3b8' },
                xaxis: { title: '時間 (秒)', gridcolor: '#334155', range: [0, maxT] },
                yaxis: { categoryorder: 'array', categoryarray: yLines.reverse(), gridcolor: '#1e293b' },
                margin: { l: 80, r: 20, t: 30, b: 50 },
                showlegend: false,
                hovermode: 'closest'
            };

            Plotly.newPlot('timeline-chart', traces, layout, { responsive: true, displayModeBar: false });

            // CLICK EVENT FOR VIDEO SEEKING
            document.getElementById('timeline-chart').on('plotly_click', (data) => {
                if (data.points.length > 0) {
                    const seekTime = data.points[0].data.customdata;
                    const player = document.getElementById('v-player');
                    player.currentTime = seekTime;
                    player.play();
                }
            });
        }

        function renderEventList(v) {
            const container = document.getElementById('event-list');
            let html = '';
            
            v.gt_segments.forEach((seg, i) => {
                const [start, end, label] = seg;
                const labelName = LABEL_MAP[label] || `L${label}`;
                html += `
                    <div class="p-3 hover:bg-slate-700 cursor-pointer text-sm" onclick="seekVideo(${start})">
                        <div class="flex justify-between items-start mb-1">
                            <span class="text-xs font-bold bg-slate-800 px-1.5 py-0.5 rounded text-blue-400">GT</span>
                            <span class="text-slate-500 text-[10px] font-mono">${start.toFixed(2)}s - ${end.toFixed(2)}s</span>
                        </div>
                        <div class="text-slate-200 font-semibold truncate">${labelName}</div>
                    </div>
                `;
            });
            
            v.pd_segments.forEach((seg, i) => {
                const [start, end, score, label] = seg;
                html += `
                    <div class="p-3 hover:bg-slate-700 cursor-pointer text-sm" onclick="seekVideo(${start})">
                        <div class="flex justify-between items-start mb-1">
                            <span class="text-xs font-bold bg-slate-800 px-1.5 py-0.5 rounded text-amber-500">PD</span>
                            <span class="text-slate-500 text-[10px] font-mono">${start.toFixed(2)}s - ${end.toFixed(2)}s</span>
                        </div>
                        <div class="text-slate-200 font-semibold truncate">${label} (S:${score.toFixed(2)})</div>
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }

        async function updateData() {
            const gt = document.getElementById('gt-path-input').value;
            const pd = document.getElementById('pd-path-input').value;
            
            const btn = document.querySelector('#path-config button');
            btn.innerText = "重新評估中...";
            btn.disabled = true;

            try {
                const resp = await fetch('/update_data', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ gt_path: gt, pd_path: pd })
                });
                const result = await resp.json();
                if (result.success) {
                    location.reload();
                } else {
                    alert("更新失敗: " + result.error);
                }
            } catch (e) {
                alert("伺服器錯誤: " + e);
            } finally {
                btn.innerText = "更新資料 (Update)";
                btn.disabled = false;
            }
        }

        function seekVideo(t) {
            const player = document.getElementById('v-player');
            player.currentTime = t;
            player.play();
        }

        document.addEventListener('DOMContentLoaded', initSummary);
    </script>
</body>
</html>
"""

# --- Backend Logic (Evaluation Integration) ---

def wrap_eval_data(gt_path, pd_path):
    """Orchestrates evaluation and packages data for the frontend."""
    # 1. Evaluate
    evaluator = AdvancedDualEvaluator(gt_path, pd_path, verbose=False)
    
    # Stage 1
    evaluator.set_eval_mode(mode="stage1", tiou_thresholds=np.linspace(0.5, 0.95, 10))
    _, s1_avg_map = evaluator.evaluate()
    s1_bench = evaluator.evaluate_low_level_benchmark()
    
    # Stage 2
    evaluator.set_eval_mode(
        mode="stage2",
        target_labels=[LABEL_NAME2ID[LabelMap.TRIGGER], LABEL_NAME2ID[LabelMap.NEGATIVE_UNSTABLE_EXTERNAL_DISTURBANCES]],
        tiou_thresholds=np.array([0.5]),
        shifts=[0.5, 2.0]
    )
    _, s2_avg_map = evaluator.evaluate()
    s2_bench = evaluator.evaluate_trigger_benchmark()

    # 2. Package Summary Metrics (following logic in eval_custom.py)
    s1_sum = s1_bench["summary"]
    s2_sum = s2_bench["summary"]

    # Stage 1 Precision/Recall Calculation
    s1_tp = s1_sum["matched"]
    s1_fn = s1_sum["gt_orphan"] + s1_sum["not_tight"]
    s1_fp = s1_sum["pd_orphan"] + s1_sum["not_tight"]
    s1_prec = s1_tp / (s1_tp + s1_fp) if (s1_tp + s1_fp) > 0 else 0.0
    s1_rec = s1_tp / (s1_tp + s1_fn) if (s1_tp + s1_fn) > 0 else 0.0

    # Stage 2 Precision/Recall Calculation
    s2_total_actions = s2_sum["just_fit"] + s2_sum["overcount"] + s2_sum["undercount"]
    s2_denominator = s2_sum["just_fit"] + s2_sum["overcount"] + s2_sum["false_alarm"] + 1e-6
    s2_prec = s2_sum["just_fit"] / s2_denominator
    s2_rec = (s2_sum["just_fit"] + s2_sum["overcount"]) / (s2_total_actions + 1e-6)

    dashboard_data = {
        "gt_path": gt_path,
        "pd_path": pd_path,
        "summary": {
            "s1": {
                "avg_mAP": s1_avg_map, 
                "precision": s1_prec,
                "recall": s1_rec,
                "matched": s1_sum["matched"],
                "not_tight": s1_sum["not_tight"],
                "gt_orphan": s1_sum["gt_orphan"],
                "pd_orphan": s1_sum["pd_orphan"],
            },
            "s2": {
                "avg_mAP": s2_avg_map, 
                "just_fit": s2_sum["just_fit"],
                "overcount": s2_sum["overcount"],
                "undercount": s2_sum["undercount"],
                "false_alarm": s2_sum["false_alarm"],
                "precision": s2_prec,
                "recall": s2_rec
            }
        },
        "videos": []
    }

    # Load raw JSONs for segments
    with open(gt_path, 'r') as f: gt_raw = json.load(f)
    with open(pd_path, 'r') as f: pd_raw = json.load(f)

    # Re-map terminology for frontend
    all_video_ids = set(s1_bench["details"].keys()) | set(s2_bench["details"].keys())

    for vid in sorted(all_video_ids):
        s2_stat = s2_bench["details"].get(vid, {
            "just_fit": 0, "overcount": 0, "undercount": 0, "false_alarm": 0,
            "total_pds": 0, "total_gts": 0, "miss_rate": 0, "false_alarm_rate": 0
        })
        s1_stat = s1_bench["details"].get(vid, {"raw_counts": {}, "metrics": {}})
        s1_rc = s1_stat.get("raw_counts", {})
        s1_met = s1_stat.get("metrics", {})
        
        # For Stage 2, metrics like F1 are not in the details dict, 
        # but miss_rate and false_alarm_rate are.
        s2_f1 = 1.0 - (s2_stat.get("miss_rate", 0) + s2_stat.get("false_alarm_rate", 0)) / 2.0
        
        video_entry = {
            "id": vid,
            "path": gt_raw.get(vid, {}).get("video_local_path", "N/A"),
            "s2": s2_stat,
            "s2_mAP": s2_f1, # Proxy for per-video performance
            "s1_mAP": s1_met.get("F1-Score", 0),
            "s1_faults": s1_rc.get("gt_orphan (Completely Missed)", 0) + s1_rc.get("pd_orphan (Hallucination)", 0),
            "s1_counts": {
                "match": s1_rc.get("matched (Perfect Hit)", 0),
                "not_tight": s1_rc.get("not_tight (Localization Error)", 0),
                "gt_orphan": s1_rc.get("gt_orphan (Completely Missed)", 0),
                "pd_orphan": s1_rc.get("pd_orphan (Hallucination)", 0),
            },
            "gt_segments": [[a["segment"][0], a["segment"][1], int(a["label"])] for a in gt_raw.get(vid, {}).get("annotations", [])],
            "pd_segments": [[a["segment"][0], a["segment"][1], a.get("score", 1.0), str(a.get("label", "0"))] for a in pd_raw.get("results", {}).get(vid, [])]
        }
        dashboard_data["videos"].append(video_entry)

    # Sort by Stage 2 errors by default
    dashboard_data["videos"].sort(key=lambda x: x["s2"]["undercount"] + x["s2"]["false_alarm"], reverse=True)
    
    return dashboard_data

# --- Flask App Creation ---

app = Flask(__name__)

# Cache the evaluation data globally
GLOBAL_DATA = None

@app.route('/')
def index():
    color_map = {
        1: "rgba(100, 100, 100, 0.4)",  # Stable
        2: "rgba(54, 162, 235, 0.6)",   # Positive Unstable
        3: "rgba(251, 113, 133, 0.6)",  # Red/Rose: External Disturbances
        4: "rgba(250, 204, 21, 0.6)",   # Yellow: Overexposure
        5: "rgba(45, 212, 191, 0.6)",   # Teal: Trigger
    }
    label_map_raw = {int(k): v.value for k, v in LABEL_MAP.items()}
    
    return render_template_string(
        HTML_TEMPLATE, 
        dashboard_data=GLOBAL_DATA, 
        color_map=color_map,
        label_map=label_map_raw
    )

@app.route('/update_data', methods=['POST'])
def update_data():
    global GLOBAL_DATA
    try:
        data = request.json
        gt_path = data.get('gt_path')
        pd_path = data.get('pd_path')
        
        if not gt_path or not pd_path:
            return jsonify({"success": False, "error": "路徑不能為空"})
        
        print(f"🔄 Re-evaluating with: \n GT: {gt_path} \n PD: {pd_path}")
        GLOBAL_DATA = wrap_eval_data(gt_path, pd_path)
        return jsonify({"success": True})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

@app.route('/video/<path:filepath>')
def serve_video(filepath):
    """Serves local video files safely."""
    # 1. Handle absolute paths (Flask/Browser might strip leading /)
    full_path = filepath
    if not full_path.startswith('/'):
        # Check if adding / makes it a valid absolute path
        potential_abs = '/' + full_path
        if os.path.exists(potential_abs):
            full_path = potential_abs
    
    # 2. Check existence
    if os.path.exists(full_path):
        return send_file(full_path)
    
    # 3. Fallback: Try relative to project root
    rel_path = os.path.join(PROJECT_ROOT, filepath)
    if os.path.exists(rel_path):
        return send_file(rel_path)
        
    return f"Video file not found at: {full_path}", 404

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HS-CODS High-Performance Dashboard")
    parser.add_argument("--gt", type=str, default=D_GT, help="Path to Ground Truth JSON")
    parser.add_argument("--pd", type=str, default=D_PD, help="Path to Prediction JSON")
    parser.add_argument("--config", type=str, default=None,
                        help="run.yaml; derives gt/pd from output.dir")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to run Flask on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0 for Docker)")
    args = parser.parse_args()

    # --config derives gt/pd from output.dir, but only when the user did not
    # override --gt/--pd (which default to D_GT/D_PD).
    gt_path, pd_path = args.gt, args.pd
    if args.config:
        from src.config.run_config import RunConfig
        cfg = RunConfig.from_yaml(args.config)
        d_gt, d_pd = cfg.eval_paths()
        if args.gt == D_GT:
            gt_path = d_gt
        if args.pd == D_PD:
            pd_path = d_pd

    print(f"🚀 Initializing Evaluation Engine...")
    try:
        GLOBAL_DATA = wrap_eval_data(gt_path, pd_path)
        print(f"✅ Evaluation complete. Processed {len(GLOBAL_DATA['videos'])} videos.")
        print(f"🔗 Dashboard is running on:")
        print(f"   - Local: http://localhost:{args.port}")
        print(f"   - Network: http://{args.host}:{args.port}")
        print(f"   - Docker: Use your server IP with mapped port (e.g., http://YOUR_SERVER_IP:41240)")
        app.run(host=args.host, port=args.port, debug=False)
    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        import traceback
        traceback.print_exc()

"""
# mac
python eval_dashboard.py \
--gt /Users/eason.hung/Documents/Projects/explus/output/Block_m0_Neck_P4_yolo/ground_truth/data.json  \
--pd /Users/eason.hung/Documents/Projects/explus/output/Block_m0_Neck_P4_yolo/predictions/merge_data.json \
--port 5004

# mac
python eval_dashboard.py \
--gt output/Block_m0_Backbone_fusion_yolo/ground_truth/data.json  \
--pd output/Block_m0_Backbone_fusion_yolo/predictions/merge_data.json \
--port 5005


# ssh_124 eX
python eval_dashboard.py \
--gt output/Block_m0_Backbone_fusion_yolo/ground_truth/data.json  \
--pd output/Block_m0_Backbone_fusion_yolo/predictions/merge_data.json \
--port 41247

# ssh_124 eX
python eval_dashboard.py \
--gt /vol/08822801/AutoTrigger/TMP_output/Block_m0_Neck_P4_yolo/ground_truth/data.json  \
--pd /vol/08822801/AutoTrigger/TMP_output/Block_m0_Neck_P4_yolo/predictions/merge_data.json \
--port 41246

python eval_dashboard.py \
--gt /vol/08822801/AutoTrigger/TMP_output/Block_m0_Backbone_fusion_yolo/ground_truth/data.json  \
--pd /vol/08822801/AutoTrigger/TMP_output/Block_m0_Backbone_fusion_yolo/predictions/merge_data.json \
--port 41247
"""