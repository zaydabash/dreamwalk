#!/usr/bin/env python3
"""
DreamWalk Web Dashboard Demo
A simple web-based dashboard showing neural signal processing
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
        self.time_points = []
    
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
    <html>
    <head>
        <title>DreamWalk Neural Interface</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            .header h1 {
                font-size: 2.5em;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
                margin: 10px 0;
            }
            .dashboard {
                display: grid;
                grid-template-columns: 1fr 2fr;
                gap: 20px;
                margin-bottom: 20px;
            }
            .panel {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 20px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .panel h2 {
                margin-top: 0;
                color: #fff;
                border-bottom: 2px solid rgba(255, 255, 255, 0.3);
                padding-bottom: 10px;
            }
            .metric {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin: 15px 0;
                padding: 10px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 8px;
            }
            .metric-label {
                font-weight: 500;
            }
            .metric-value {
                font-size: 1.2em;
                font-weight: bold;
                color: #4CAF50;
            }
            .motifs {
                margin: 15px 0;
            }
            .motif-tag {
                display: inline-block;
                background: rgba(76, 175, 80, 0.3);
                color: #4CAF50;
                padding: 5px 10px;
                margin: 2px;
                border-radius: 15px;
                font-size: 0.9em;
            }
            .mood {
                text-align: center;
                font-size: 1.5em;
                font-weight: bold;
                color: #FFD700;
                margin: 20px 0;
                padding: 15px;
                background: rgba(255, 215, 0, 0.1);
                border-radius: 10px;
                border: 2px solid rgba(255, 215, 0, 0.3);
            }
            .chart-container {
                height: 300px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                padding: 10px;
            }
            .status {
                text-align: center;
                margin-top: 20px;
                padding: 10px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                font-size: 0.9em;
                opacity: 0.8;
            }
            .connection-status {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 10px 15px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: bold;
            }
            .connected {
                background: rgba(76, 175, 80, 0.8);
                color: white;
            }
            .disconnected {
                background: rgba(244, 67, 54, 0.8);
                color: white;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 DreamWalk Neural Interface</h1>
                <p>Real-time neural signal processing and dreamscape generation</p>
            </div>
            
            <div class="connection-status" id="connectionStatus">Connecting...</div>
            
            <div class="dashboard">
                <div class="panel">
                    <h2>Neural State</h2>
                    
                    <div class="metric">
                        <span class="metric-label">Valence (Emotional Tone)</span>
                        <span class="metric-value" id="valence">0.00</span>
                    </div>
                    
                    <div class="metric">
                        <span class="metric-label">Arousal (Energy Level)</span>
                        <span class="metric-value" id="arousal">0.50</span>
                    </div>
                    
                    <div class="metric">
                        <span class="metric-label">Dominance (Control)</span>
                        <span class="metric-value" id="dominance">0.00</span>
                    </div>
                    
                    <div class="motifs">
                        <strong>Neural Motifs:</strong><br>
                        <div id="motifs">
                            <span class="motif-tag">calm</span>
                            <span class="motif-tag">peaceful</span>
                        </div>
                    </div>
                    
                    <div class="mood" id="mood">Calm and Content</div>
                </div>
                
                <div class="panel">
                    <h2>EEG Signal Visualization</h2>
                    <div class="chart-container">
                        <canvas id="eegChart" width="100%" height="100%"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="status" id="status">
                Demo running... Generating neural data
            </div>
        </div>
        
        <script>
            let ws;
            let chart;
            let eegData = [];
            
            function connect() {
                ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onopen = function() {
                    document.getElementById('connectionStatus').textContent = 'Connected';
                    document.getElementById('connectionStatus').className = 'connection-status connected';
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                ws.onclose = function() {
                    document.getElementById('connectionStatus').textContent = 'Disconnected';
                    document.getElementById('connectionStatus').className = 'connection-status disconnected';
                    setTimeout(connect, 3000);
                };
            }
            
            function updateDashboard(data) {
                // Update neural state
                document.getElementById('valence').textContent = data.valence.toFixed(2);
                document.getElementById('arousal').textContent = data.arousal.toFixed(2);
                document.getElementById('dominance').textContent = data.dominance.toFixed(2);
                document.getElementById('mood').textContent = data.mood;
                
                // Update motifs
                const motifsDiv = document.getElementById('motifs');
                motifsDiv.innerHTML = data.motif_tags.map(tag => 
                    `<span class="motif-tag">${tag}</span>`
                ).join(' ');
                
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
                const width = canvas.width = canvas.offsetWidth;
                const height = canvas.height = canvas.offsetHeight;
                
                ctx.clearRect(0, 0, width, height);
                
                if (eegData.length > 1) {
                    ctx.strokeStyle = '#4CAF50';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    
                    const stepX = width / (eegData.length - 1);
                    const centerY = height / 2;
                    const scaleY = height / 100; // Scale for ±50 range
                    
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
    print("Open your browser to: http://localhost:8000")
    print("Press Ctrl+C to stop")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
