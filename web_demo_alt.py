#!/usr/bin/env python3
"""
DreamWalk Simple Web Demo (Alternative Port)
A lightweight web-based dashboard for DreamWalk neural interface
"""

import asyncio
import json
import random
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="DreamWalk Neural Interface", version="1.0.0")

class NeuralState:
    def __init__(self):
        self.valence = 0.0
        self.arousal = 0.5
        self.dominance = 0.0
        self.motif_tags = ["calm", "peaceful"]
        self.eeg_data = []
    
    def update(self):
        """Update neural state with realistic changes"""
        # Simulate gradual changes
        self.valence += random.uniform(-0.1, 0.1)
        self.arousal += random.uniform(-0.05, 0.05)
        self.dominance += random.uniform(-0.08, 0.08)
        
        # Clamp values
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(-1.0, min(1.0, self.dominance))
        
        # Update motifs
        motifs = []
        if self.valence > 0.3:
            motifs.append("positive")
        elif self.valence < -0.3:
            motifs.append("negative")
        
        if self.arousal > 0.6:
            motifs.append("energetic")
        elif self.arousal < 0.3:
            motifs.append("calm")
        
        if self.dominance > 0.3:
            motifs.append("confident")
        elif self.dominance < -0.3:
            motifs.append("submissive")
        
        if not motifs:
            motifs = ["neutral", "balanced"]
        
        self.motif_tags = motifs
        
        # Generate EEG data
        t = np.linspace(0, 1, 10)
        alpha_amp = 20 * (1 - self.arousal)
        beta_amp = 15 * self.arousal
        theta_amp = 10 * abs(self.valence)
        
        signal = (alpha_amp * np.sin(2 * np.pi * 10 * t) +
                 beta_amp * np.sin(2 * np.pi * 20 * t) +
                 theta_amp * np.sin(2 * np.pi * 6 * t) +
                 np.random.normal(0, 2, len(t)))
        
        self.eeg_data.extend(signal.tolist())
        if len(self.eeg_data) > 100:
            self.eeg_data = self.eeg_data[-100:]
    
    def get_mood(self):
        """Determine mood from neural state"""
        if self.valence > 0 and self.arousal > 0.5:
            return "Excited and Joyful"
        elif self.valence > 0 and self.arousal <= 0.5:
            return "Calm and Content"
        elif self.valence <= 0 and self.arousal > 0.5:
            return "Anxious and Agitated"
        else:
            return "Sad and Withdrawn"
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "valence": round(self.valence, 2),
            "arousal": round(self.arousal, 2),
            "dominance": round(self.dominance, 2),
            "motif_tags": self.motif_tags,
            "mood": self.get_mood(),
            "eeg_data": self.eeg_data[-20:],  # Last 2 seconds
            "timestamp": datetime.now().isoformat()
        }

# Global neural state
neural_state = NeuralState()

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DreamWalk Neural Interface</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
        <style>
            :root {
                color-scheme: dark;
                --bg: #0a0a0a;
                --surface: #161616;
                --surface-2: #1c1c1c;
                --border: rgba(255, 255, 255, 0.08);
                --text: #ededed;
                --text-muted: #8a8a8a;
                --accent: #ffffff;
                --accent-dim: rgba(255, 255, 255, 0.08);
                --ok: #ededed;
                --bad: #4a4a4a;
                --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
                --font-sans: 'Space Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                --font-mono: 'Space Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
            }

            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                min-height: 100vh;
                padding: 2.5rem;
                font-family: var(--font-sans);
                color: var(--text);
                background-color: var(--bg);
                background-image:
                    radial-gradient(circle at 88% 8%, rgba(255, 255, 255, 0.04), transparent 45%),
                    repeating-linear-gradient(0deg, rgba(255, 255, 255, 0.02) 0px, rgba(255, 255, 255, 0.02) 1px, transparent 1px, transparent 48px),
                    repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.02) 0px, rgba(255, 255, 255, 0.02) 1px, transparent 1px, transparent 48px);
            }

            .container {
                max-width: 1280px;
                margin: 0 auto;
            }

            .header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 1.5rem;
                padding-bottom: 1.5rem;
                margin-bottom: 2.5rem;
                border-bottom: 1px solid var(--border);
            }

            .header h1 {
                margin: 0;
                font-size: 1.75rem;
                font-weight: 700;
                letter-spacing: -0.02em;
            }

            .header p {
                margin: 0.5rem 0 0;
                color: var(--text-muted);
                font-size: 0.9rem;
            }

            .connection-status {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.4rem 0.75rem;
                border: 1px solid var(--border);
                border-radius: 6px;
                font-family: var(--font-mono);
                font-size: 0.7rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: var(--text-muted);
                white-space: nowrap;
            }

            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--text-muted);
            }

            .connection-status.connected .status-dot {
                background: var(--ok);
                box-shadow: 0 0 8px rgba(255, 255, 255, 0.35);
                animation: pulse 2s var(--ease-out) infinite;
            }

            .connection-status.disconnected .status-dot {
                background: var(--bad);
                box-shadow: none;
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.35; }
            }

            .dashboard {
                display: grid;
                grid-template-columns: minmax(280px, 380px) 1fr;
                gap: 1.5rem;
                margin-bottom: 1.5rem;
            }

            .panel {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1.75rem;
                transition: transform 200ms var(--ease-out), border-color 200ms var(--ease-out);
                animation: fadeIn 500ms var(--ease-out) both;
            }

            .panel:nth-of-type(2) {
                animation-delay: 60ms;
            }

            .panel:hover {
                transform: translateY(-2px);
                border-color: rgba(255, 255, 255, 0.18);
            }

            .panel h2 {
                margin: 0 0 1.5rem;
                padding-bottom: 0.9rem;
                border-bottom: 1px solid var(--border);
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: var(--text-muted);
            }

            .metric {
                padding: 0.85rem 0;
                border-bottom: 1px solid var(--border);
            }

            .metric:last-of-type {
                border-bottom: none;
            }

            .metric-head {
                display: flex;
                justify-content: space-between;
                align-items: baseline;
            }

            .metric-label {
                font-size: 0.85rem;
                color: var(--text-muted);
            }

            .metric-value {
                font-family: var(--font-mono);
                font-variant-numeric: tabular-nums;
                font-size: 1.2rem;
                font-weight: 700;
                color: var(--text);
            }

            .metric-track {
                position: relative;
                height: 4px;
                margin-top: 0.6rem;
                background: rgba(255, 255, 255, 0.06);
                border-radius: 2px;
                overflow: hidden;
            }

            .metric-track.bipolar::after {
                content: '';
                position: absolute;
                top: 0;
                bottom: 0;
                left: 50%;
                width: 1px;
                background: rgba(255, 255, 255, 0.18);
            }

            .metric-fill {
                position: absolute;
                top: 0;
                height: 100%;
                background: var(--accent);
                border-radius: 2px;
                transition: left 400ms var(--ease-out), width 400ms var(--ease-out);
            }

            .motifs {
                margin-top: 1.5rem;
            }

            .motifs-label {
                display: block;
                margin-bottom: 0.6rem;
                font-size: 0.7rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: var(--text-muted);
            }

            .motif-tag {
                display: inline-block;
                margin: 0 0.4rem 0.4rem 0;
                padding: 0.3rem 0.6rem;
                border-radius: 4px;
                background: var(--accent-dim);
                color: var(--accent);
                font-family: var(--font-mono);
                font-size: 0.7rem;
                letter-spacing: 0.04em;
            }

            .mood-section {
                margin-top: 1.5rem;
                padding-top: 1.5rem;
                border-top: 1px solid var(--border);
            }

            .mood-value {
                margin-top: 0.6rem;
                padding-left: 0.9rem;
                border-left: 3px solid var(--accent);
                font-size: 1.3rem;
                font-weight: 600;
                letter-spacing: -0.01em;
            }

            .chart-container {
                height: 320px;
                border: 1px solid var(--border);
                border-radius: 8px;
                background: var(--surface-2);
                overflow: hidden;
            }

            .chart-container canvas {
                display: block;
                width: 100%;
                height: 100%;
            }

            .status-bar {
                padding: 0.9rem;
                border: 1px solid var(--border);
                border-radius: 8px;
                background: var(--surface);
                text-align: center;
                font-family: var(--font-mono);
                font-size: 0.75rem;
                color: var(--text-muted);
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(8px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @media (max-width: 860px) {
                .dashboard {
                    grid-template-columns: 1fr;
                }
            }

            @media (prefers-reduced-motion: reduce) {
                .panel {
                    animation: none;
                }
                .connection-status.connected .status-dot {
                    animation: none;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header class="header">
                <div>
                    <h1>DreamWalk Neural Interface</h1>
                    <p>Real-time neural signal processing and dreamscape generation</p>
                </div>
                <div class="connection-status" id="connectionStatus">
                    <span class="status-dot"></span>
                    <span class="status-text">Connecting</span>
                </div>
            </header>

            <main class="dashboard">
                <section class="panel">
                    <h2>Neural state</h2>

                    <div class="metric">
                        <div class="metric-head">
                            <span class="metric-label">Valence</span>
                            <span class="metric-value" id="valence">0.00</span>
                        </div>
                        <div class="metric-track bipolar">
                            <div class="metric-fill" id="valence-bar"></div>
                        </div>
                    </div>

                    <div class="metric">
                        <div class="metric-head">
                            <span class="metric-label">Arousal</span>
                            <span class="metric-value" id="arousal">0.50</span>
                        </div>
                        <div class="metric-track">
                            <div class="metric-fill" id="arousal-bar"></div>
                        </div>
                    </div>

                    <div class="metric">
                        <div class="metric-head">
                            <span class="metric-label">Dominance</span>
                            <span class="metric-value" id="dominance">0.00</span>
                        </div>
                        <div class="metric-track bipolar">
                            <div class="metric-fill" id="dominance-bar"></div>
                        </div>
                    </div>

                    <div class="motifs">
                        <span class="motifs-label">Neural motifs</span>
                        <div id="motifs">
                            <span class="motif-tag">calm</span>
                            <span class="motif-tag">peaceful</span>
                        </div>
                    </div>

                    <div class="mood-section">
                        <span class="motifs-label">Current state</span>
                        <div class="mood-value" id="mood">Calm and Content</div>
                    </div>
                </section>

                <section class="panel">
                    <h2>EEG signal visualization</h2>
                    <div class="chart-container">
                        <canvas id="eegChart"></canvas>
                    </div>
                </section>
            </main>

            <div class="status-bar" id="status">
                Demo running, generating neural data
            </div>
        </div>

        <script>
            let ws;
            let eegData = [];

            function connect() {
                ws = new WebSocket('ws://localhost:8001/ws');

                ws.onopen = function() {
                    const el = document.getElementById('connectionStatus');
                    el.className = 'connection-status connected';
                    el.querySelector('.status-text').textContent = 'Connected';
                };

                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };

                ws.onclose = function() {
                    const el = document.getElementById('connectionStatus');
                    el.className = 'connection-status disconnected';
                    el.querySelector('.status-text').textContent = 'Disconnected';
                    setTimeout(connect, 3000);
                };
            }

            function setBar(id, value, bipolar) {
                const el = document.getElementById(id);
                if (bipolar) {
                    const pct = Math.max(-1, Math.min(1, value)) * 50;
                    if (pct >= 0) {
                        el.style.left = '50%';
                        el.style.width = pct + '%';
                    } else {
                        el.style.left = (50 + pct) + '%';
                        el.style.width = (-pct) + '%';
                    }
                } else {
                    const pct = Math.max(0, Math.min(1, value)) * 100;
                    el.style.left = '0%';
                    el.style.width = pct + '%';
                }
            }

            function updateDashboard(data) {
                // Update neural state
                document.getElementById('valence').textContent = data.valence.toFixed(2);
                document.getElementById('arousal').textContent = data.arousal.toFixed(2);
                document.getElementById('dominance').textContent = data.dominance.toFixed(2);
                document.getElementById('mood').textContent = data.mood;

                setBar('valence-bar', data.valence, true);
                setBar('arousal-bar', data.arousal, false);
                setBar('dominance-bar', data.dominance, true);

                // Update motifs
                const motifsDiv = document.getElementById('motifs');
                motifsDiv.innerHTML = data.motif_tags.map(tag =>
                    `<span class="motif-tag">${tag}</span>`
                ).join('');

                // Update EEG chart
                eegData = data.eeg_data;
                updateChart();

                // Update status
                document.getElementById('status').textContent =
                    `Last update: ${new Date().toLocaleTimeString()} - ${data.mood}`;
            }

            function updateChart() {
                const canvas = document.getElementById('eegChart');
                const ctx = canvas.getContext('2d');
                const width = canvas.width = canvas.clientWidth;
                const height = canvas.height = canvas.clientHeight;

                ctx.clearRect(0, 0, width, height);

                ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
                ctx.lineWidth = 1;
                for (let i = 1; i < 4; i++) {
                    const y = (height / 4) * i;
                    ctx.beginPath();
                    ctx.moveTo(0, y);
                    ctx.lineTo(width, y);
                    ctx.stroke();
                }

                if (eegData.length > 1) {
                    ctx.strokeStyle = '#ededed';
                    ctx.lineWidth = 2;
                    ctx.shadowColor = 'rgba(255, 255, 255, 0.25)';
                    ctx.shadowBlur = 8;
                    ctx.beginPath();

                    const stepX = width / (eegData.length - 1);
                    const centerY = height / 2;
                    const scaleY = height / 120; // Scale for +/-60 range

                    for (let i = 0; i < eegData.length; i++) {
                        const x = i * stepX;
                        const y = centerY - (eegData[i] * scaleY);

                        if (i === 0) {
                            ctx.moveTo(x, y);
                        } else {
                            ctx.lineTo(x, y);
                        }
                    }

                    ctx.stroke();
                    ctx.shadowBlur = 0;
                }
            }

            // Initialize
            connect();

            // Handle window resize
            window.addEventListener('resize', updateChart);
        </script>
    </body>
    </html>
    """

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data"""
    await websocket.accept()
    
    try:
        while True:
            # Update neural state
            neural_state.update()
            
            # Send data to client
            data = neural_state.to_dict()
            await websocket.send_text(json.dumps(data))
            
            # Wait 1 second before next update
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.get("/api/state")
async def get_state():
    """REST API endpoint for current neural state"""
    neural_state.update()
    return neural_state.to_dict()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "dreamwalk-demo"}

if __name__ == "__main__":
    print("Starting DreamWalk Web Dashboard Demo...")
    print("Open your browser to: http://localhost:8001")
    print("Press Ctrl+C to stop")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
