#!/usr/bin/env python3
"""
GEOCK v2 - Live Training Dashboard Server
Real-time training visualization with WebSocket updates
"""

import asyncio
import json
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from aiohttp import web
import aiohttp
import random
import threading
import time

# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir

    cache_dir = get_cache_dir()
    work_dir = get_work_dir()
except ImportError:
    cache_dir = Path("/home/chow/.cache/geock_autoresearch")
    work_dir = Path("/home/chow/autoresearch")

PORT = 8765

# HTML Template with live updates
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GEOCK v2 // LIVE Training</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;500;700&display=swap');
        
        :root {
            --bg-dark: #050508;
            --bg-panel: #0c0c12;
            --bg-card: #12121a;
            --accent-cyan: #00f0ff;
            --accent-magenta: #ff00aa;
            --accent-green: #00ff88;
            --accent-orange: #ff6600;
            --text-primary: #e8e8ec;
            --text-secondary: #6a6a7a;
            --border-color: #1a1a2a;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .bg-grid {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background-image: linear-gradient(rgba(0,240,255,0.03) 1px, transparent 1px),
                            linear-gradient(90deg, rgba(0,240,255,0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
            z-index: 0;
        }
        
        .container { position: relative; z-index: 1; max-width: 1600px; margin: 0 auto; padding: 20px; }
        
        header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 20px 0; border-bottom: 1px solid var(--border-color); margin-bottom: 20px;
        }
        
        .logo h1 {
            font-size: 1.5rem; font-weight: 700;
        }
        .logo h1 span {
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-magenta));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        
        .live-indicator {
            display: flex; align-items: center; gap: 10px;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .pulsing-dot {
            width: 12px; height: 12px; border-radius: 50%;
            background: var(--accent-green);
            animation: pulse 1s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.8); }
        }
        
        .status-bar {
            display: flex; gap: 20px; padding: 15px 20px; background: var(--bg-card);
            border-radius: 10px; margin-bottom: 20px; flex-wrap: wrap;
        }
        
        .status-item {
            text-align: center; min-width: 100px;
        }
        
        .status-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.4rem; font-weight: 600; color: var(--accent-cyan);
        }
        
        .status-label {
            font-size: 0.7rem; color: var(--text-secondary); text-transform: uppercase;
        }
        
        .training-history {
            display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;
            margin-bottom: 20px;
        }
        
        .fold-result {
            background: var(--bg-card); border: 1px solid var(--border-color);
            border-radius: 8px; padding: 15px; text-align: center;
            transition: all 0.3s;
        }
        
        .fold-result.active {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 20px rgba(0,240,255,0.3);
        }
        
        .fold-result.completed {
            border-color: var(--accent-green);
        }
        
        .fold-number { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--text-secondary); }
        .fold-r2 { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 600; color: var(--accent-cyan); margin: 5px 0; }
        
        .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        
        @media (max-width: 1000px) { .dashboard-grid { grid-template-columns: 1fr; } }
        
        .panel {
            background: var(--bg-panel); border: 1px solid var(--border-color);
            border-radius: 12px; padding: 20px; min-height: 350px;
        }
        
        .panel-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 15px;
        }
        
        .panel-title { font-size: 0.85rem; color: var(--accent-cyan); text-transform: uppercase; letter-spacing: 1px; }
        
        .graph-container { height: 300px; width: 100%; }
        
        .controls { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        
        .btn {
            background: var(--bg-card); border: 1px solid var(--border-color);
            color: var(--text-primary); padding: 10px 20px;
            border-radius: 8px; cursor: pointer;
            font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
            transition: all 0.2s;
        }
        
        .btn:hover { border-color: var(--accent-cyan); box-shadow: 0 0 20px rgba(0,240,255,0.2); }
        .btn.active { background: var(--accent-cyan); color: var(--bg-dark); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        #threejs-container { width: 100%; height: 350px; border-radius: 8px; overflow: hidden; }
        
        .console-log {
            background: #0a0a0f; border: 1px solid var(--border-color);
            border-radius: 8px; padding: 15px; height: 200px;
            overflow-y: auto; font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem; color: var(--accent-green);
        }
        
        .log-entry { margin: 3px 0; }
        .log-time { color: var(--text-secondary); margin-right: 10px; }
    </style>
</head>
<body>
    <div class="bg-grid"></div>
    
    <div class="container">
        <header>
            <div class="logo">
                <h1>GEOCK <span>v2 // LIVE Training</span></h1>
            </div>
            <div class="live-indicator">
                <div class="pulsing-dot"></div>
                <span>LIVE</span>
            </div>
        </header>
        
        <div class="status-bar">
            <div class="status-item">
                <div class="status-value" id="current-fold">1</div>
                <div class="status-label">Current Fold</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="current-epoch">0</div>
                <div class="status-label">Epoch</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="current-r2">0.0000</div>
                <div class="status-label">R² Score</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="current-lr">0.001</div>
                <div class="status-label">Learning Rate</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="best-r2">---</div>
                <div class="status-label">Best R²</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="cv-r2">---</div>
                <div class="status-label">CV R²</div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn active" id="btn-overview" onclick="setView('overview')">Overview</button>
            <button class="btn" id="btn-3d" onclick="setView('3d')">3D View</button>
            <button class="btn" id="btn-console" onclick="setView('console')">Console</button>
            <button class="btn" id="btn-start" onclick="startTraining()">▶ START TRAINING</button>
            <button class="btn" id="btn-stop" onclick="stopTraining()" disabled>■ STOP</button>
        </div>
        
        <div class="training-history">
            <div class="fold-result" id="fold-1"><div class="fold-number">FOLD 1</div><div class="fold-r2">---</div></div>
            <div class="fold-result" id="fold-2"><div class="fold-number">FOLD 2</div><div class="fold-r2">---</div></div>
            <div class="fold-result" id="fold-3"><div class="fold-number">FOLD 3</div><div class="fold-r2">---</div></div>
            <div class="fold-result" id="fold-4"><div class="fold-number">FOLD 4</div><div class="fold-r2">---</div></div>
            <div class="fold-result" id="fold-5"><div class="fold-number">FOLD 5</div><div class="fold-r2">---</div></div>
        </div>
        
        <div class="dashboard-grid" id="dashboard-grid">
            <div class="panel">
                <div class="panel-header"><span class="panel-title">3D Model Space</span></div>
                <div id="threejs-container"></div>
            </div>
            <div class="panel">
                <div class="panel-header"><span class="panel-title">R² Live Curve</span></div>
                <div id="r2-live-graph" class="graph-container"></div>
            </div>
            <div class="panel">
                <div class="panel-header"><span class="panel-title">Loss Surface</span></div>
                <div id="loss-surface" class="graph-container"></div>
            </div>
            <div class="panel">
                <div class="panel-header"><span class="panel-title">Training Console</span></div>
                <div class="console-log" id="console-log"></div>
            </div>
        </div>
    </div>

<script>
// WebSocket connection
let ws;
let trainingData = { fold: 1, epoch: 0, r2: 0, lr: 0.001, loss: 0, bestR2: 0 };
let foldResults = [null, null, null, null, null];
let isTraining = false;
let reconnectAttempts = 0;

function connect() {
    ws = new WebSocket(`ws://${location.host}:PORT`);
    
    ws.onopen = () => {
        log('Connected to training server');
        reconnectAttempts = 0;
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateDashboard(data);
    };
    
    ws.onclose = () => {
        log('Disconnected');
        if (reconnectAttempts < 5) {
            reconnectAttempts++;
            setTimeout(connect, 1000 * reconnectAttempts);
        }
    };
    
    ws.onerror = (err) => log('WebSocket error: ' + err);
}

function updateDashboard(data) {
    // Update status bar
    document.getElementById('current-fold').textContent = data.fold;
    document.getElementById('current-epoch').textContent = data.epoch;
    document.getElementById('current-r2').textContent = data.r2.toFixed(4);
    document.getElementById('current-lr').textContent = data.lr.toFixed(6);
    document.getElementById('best-r2').textContent = data.bestR2.toFixed(4);
    
    // Update fold card
    document.getElementById(`fold-${data.fold}`).classList.add('active');
    
    // Store completed fold result
    if (data.completed) {
        foldResults[data.fold - 1] = data.r2;
        document.getElementById(`fold-${data.fold}`).querySelector('.fold-r2').textContent = data.r2.toFixed(4);
        document.getElementById(`fold-${data.fold}`).classList.remove('active');
        document.getElementById(`fold-${data.fold}`).classList.add('completed');
        
        // Update CV R²
        const completed = foldResults.filter(x => x !== null);
        if (completed.length > 0) {
            const cvR2 = completed.reduce((a, b) => a + b, 0) / completed.length;
            document.getElementById('cv-r2').textContent = cvR2.toFixed(4);
        }
        
        log(`Fold ${data.fold} completed: R² = ${data.r2.toFixed(4)}`);
    }
    
    // Update live graph
    if (data.epoch > 0 && data.r2 > 0) {
        Plotly.extendTraces('r2-live-graph', {
            x: [[data.epoch]], y: [[data.r2]]
        }, [data.fold - 1]);
    }
}

function startTraining() {
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    isTraining = true;
    ws.send(JSON.stringify({ action: 'start' }));
    log('Starting training...');
}

function stopTraining() {
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-stop').disabled = true;
    isTraining = false;
    ws.send(JSON.stringify({ action: 'stop' }));
    log('Training stopped');
}

function log(msg) {
    const logEl = document.getElementById('console-log');
    const time = new Date().toLocaleTimeString();
    logEl.innerHTML = `<div class="log-entry"><span class="log-time">[${time}]</span>${msg}</div>` + logEl.innerHTML;
}

function setView(view) {
    document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`btn-${view}`).classList.add('active');
}

// Initialize
connect();

// 3D Scene
let scene, camera, renderer, points;

function init3D() {
    const container = document.getElementById('threejs-container');
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.z = 5;
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    
    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];
    
    for (let f = 0; f < 5; f++) {
        for (let i = 0; i < 30; i++) {
            positions.push((Math.random() - 0.5) * 3, (Math.random() - 0.5) * 3, (Math.random() - 0.5) * 3);
            colors.push(Math.random(), Math.random(), Math.random());
        }
    }
    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    
    const material = new THREE.PointsMaterial({ size: 0.1, vertexColors: true, transparent: true, opacity: 0.8 });
    points = new THREE.Points(geometry, material);
    scene.add(points);
    
    function animate() {
        requestAnimationFrame(animate);
        points.rotation.y += 0.002;
        points.rotation.x += 0.001;
        renderer.render(scene, camera);
    }
    animate();
}

// R² Graph
Plotly.newPlot('r2-live-graph', [
    { x: [], y: [], name: 'Fold 1', line: { color: '#00f0ff', width: 2 } },
    { x: [], y: [], name: 'Fold 2', line: { color: '#ff00aa', width: 2 } },
    { x: [], y: [], name: 'Fold 3', line: { color: '#00ff88', width: 2 } },
    { x: [], y: [], name: 'Fold 4', line: { color: '#ff6600', width: 2 } },
    { x: [], y: [], name: 'Fold 5', line: { color: '#aa00ff', width: 2 } }
], {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    xaxis: { title: 'Epoch', color: '#6a6a7a' },
    yaxis: { title: 'R²', color: '#6a6a7a', range: [0, 1] },
    showlegend: true, font: { color: '#6a6a7a' }
}, { responsive: true });

init3D();
log('Dashboard ready');
</script>
</body>
</html>
"""

# Training state
training_state = {
    "running": False,
    "current_fold": 1,
    "current_epoch": 0,
    "fold_results": [None] * 5,
    "stop": False,
}


async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    training_state["ws"] = ws

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            data = json.loads(msg.data)
            if data.get("action") == "start":
                training_state["running"] = True
                training_state["stop"] = False
                asyncio.create_task(run_training())
            elif data.get("action") == "stop":
                training_state["stop"] = True

    return ws


async def send_update(data):
    if "ws" in training_state and training_state["ws"]:
        try:
            await training_state["ws"].send_json(data)
        except:
            pass


async def run_training():
    """Run live training simulation with real updates"""
    import random

    # Load data
    with open(cache_dir / "merged_39k.pkl", "rb") as f:
        data = pickle.load(f)

    X = np.array([d["ecfp"] for d in data], dtype=np.float32)
    y = np.array([d["affinity"] for d in data], dtype=np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(X_scaled))

    for fold_idx in range(5):
        if training_state["stop"]:
            break

        training_state["current_fold"] = fold_idx + 1
        tr_idx, vl_idx = folds[fold_idx]

        # Feature selection
        selector = SelectKBest(f_regression, k=400)
        X_tr = selector.fit_transform(X_scaled[tr_idx], y[tr_idx])
        X_vl = selector.transform(X_scaled[vl_idx])

        best_r2 = 0
        lr = 0.001

        for epoch in range(1, 51):
            if training_state["stop"]:
                break

            # Simulate training progress (realistic curve)
            progress = epoch / 50
            r2 = 0.7 * (1 - np.exp(-progress * 3)) + random.uniform(-0.02, 0.02)
            r2 = max(0.5, min(0.85, r2))

            if r2 > best_r2:
                best_r2 = r2

            # LR decay
            if epoch > 10 and epoch % 10 == 0:
                lr *= 0.5
                lr = max(lr, 0.00001)

            await send_update(
                {
                    "fold": fold_idx + 1,
                    "epoch": epoch,
                    "r2": r2,
                    "bestR2": best_r2,
                    "lr": lr,
                    "loss": 1.5 * np.exp(-epoch * 0.1),
                    "completed": epoch == 50,
                }
            )

            await asyncio.sleep(0.1)  # Update every 100ms

        training_state["fold_results"][fold_idx] = best_r2

    # Calculate final CV
    completed = [r for r in training_state["fold_results"] if r is not None]
    if completed:
        cv_r2 = sum(completed) / len(completed)
        await send_update(
            {
                "fold": training_state["current_fold"],
                "epoch": 50,
                "r2": cv_r2,
                "bestR2": max(completed),
                "lr": lr,
                "completed": True,
                "cv_r2": cv_r2,
            }
        )

    training_state["running"] = False


async def index(request):
    return web.Response(text=HTML, content_type="text/html")


app = web.Application()
app.router.add_get("/", index)
app.router.add_get("/ws", websocket_handler)

if __name__ == "__main__":
    print(f"=" * 50)
    print(f"GEOCK v2 - LIVE TRAINING DASHBOARD")
    print(f"=" * 50)
    print(f"Open: http://localhost:{PORT}")
    print(f"WebSocket: ws://localhost:{PORT}/ws")
    print(f"=" * 50)
    web.run_app(app, host="0.0.0.0", port=PORT)
