#!/usr/bin/env python3
"""
Simple DreamWalk Demo Script
Generates mock neural data and displays it in a basic dashboard
"""

import asyncio
import json
import random
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

class DreamWalkDemo:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DreamWalk - Neural Dreamscape Generator Demo")
        self.root.geometry("1200x800")
        
        # Neural state variables
        self.valence = 0.0  # -1 to 1
        self.arousal = 0.5   # 0 to 1
        self.dominance = 0.0 # -1 to 1
        self.motif_tags = []
        
        # EEG simulation data
        self.eeg_data = []
        self.time_points = []
        
        self.setup_ui()
        self.start_demo()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="DreamWalk Neural Interface Demo", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Neural State
        left_frame = ttk.LabelFrame(main_frame, text="Neural State", padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Valence display
        ttk.Label(left_frame, text="Valence (Emotional Tone):").grid(row=0, column=0, sticky=tk.W)
        self.valence_var = tk.StringVar(value="0.00")
        ttk.Label(left_frame, textvariable=self.valence_var, font=("Arial", 12, "bold")).grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Arousal display
        ttk.Label(left_frame, text="Arousal (Energy Level):").grid(row=1, column=0, sticky=tk.W)
        self.arousal_var = tk.StringVar(value="0.50")
        ttk.Label(left_frame, textvariable=self.arousal_var, font=("Arial", 12, "bold")).grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        # Dominance display
        ttk.Label(left_frame, text="Dominance (Control):").grid(row=2, column=0, sticky=tk.W)
        self.dominance_var = tk.StringVar(value="0.00")
        ttk.Label(left_frame, textvariable=self.dominance_var, font=("Arial", 12, "bold")).grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        
        # Motif tags
        ttk.Label(left_frame, text="Neural Motifs:").grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        self.motif_var = tk.StringVar(value="calm, peaceful")
        ttk.Label(left_frame, textvariable=self.motif_var, font=("Arial", 10)).grid(row=4, column=0, columnspan=2, sticky=tk.W)
        
        # Mood description
        ttk.Label(left_frame, text="Current Mood:").grid(row=5, column=0, sticky=tk.W, pady=(10, 0))
        self.mood_var = tk.StringVar(value="Calm and Content")
        ttk.Label(left_frame, textvariable=self.mood_var, font=("Arial", 12, "bold"), 
                 foreground="blue").grid(row=6, column=0, columnspan=2, sticky=tk.W)
        
        # Right panel - EEG Visualization
        right_frame = ttk.LabelFrame(main_frame, text="EEG Signal Visualization", padding="10")
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title("Simulated EEG Signals")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Amplitude (μV)")
        self.ax.grid(True, alpha=0.3)
        
        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Demo running... Generating neural data")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def generate_neural_state(self):
        """Generate realistic neural state changes"""
        # Simulate gradual changes with some randomness
        self.valence += random.uniform(-0.1, 0.1)
        self.arousal += random.uniform(-0.05, 0.05)
        self.dominance += random.uniform(-0.08, 0.08)
        
        # Clamp values to valid ranges
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(-1.0, min(1.0, self.dominance))
        
        # Update motif tags based on state
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
        
        # Determine mood
        if self.valence > 0 and self.arousal > 0.5:
            mood = "Excited and Joyful"
        elif self.valence > 0 and self.arousal <= 0.5:
            mood = "Calm and Content"
        elif self.valence <= 0 and self.arousal > 0.5:
            mood = "Anxious and Agitated"
        else:
            mood = "Sad and Withdrawn"
        
        return mood
    
    def generate_eeg_data(self):
        """Generate simulated EEG data"""
        # Generate time points
        current_time = time.time()
        self.time_points.append(current_time)
        
        # Keep only last 10 seconds of data
        if len(self.time_points) > 100:  # Assuming 10Hz sampling
            self.time_points = self.time_points[-100:]
        
        # Generate EEG-like signal with multiple frequency components
        t = np.linspace(0, 1, 10)  # 1 second of data at 10Hz
        
        # Alpha waves (8-13 Hz) - dominant when relaxed
        alpha_freq = 10
        alpha_amp = 20 * (1 - self.arousal)  # Higher when calm
        
        # Beta waves (13-30 Hz) - dominant when alert
        beta_freq = 20
        beta_amp = 15 * self.arousal  # Higher when aroused
        
        # Theta waves (4-8 Hz) - associated with emotions
        theta_freq = 6
        theta_amp = 10 * abs(self.valence)  # Higher with strong emotions
        
        # Generate the signal
        signal = (alpha_amp * np.sin(2 * np.pi * alpha_freq * t) +
                 beta_amp * np.sin(2 * np.pi * beta_freq * t) +
                 theta_amp * np.sin(2 * np.pi * theta_freq * t) +
                 np.random.normal(0, 2, len(t)))  # Add noise
        
        self.eeg_data.extend(signal.tolist())
        
        # Keep only last 10 seconds
        if len(self.eeg_data) > 100:
            self.eeg_data = self.eeg_data[-100:]
    
    def update_display(self):
        """Update the display with new data"""
        # Update neural state
        mood = self.generate_neural_state()
        
        # Update UI variables
        self.valence_var.set(f"{self.valence:.2f}")
        self.arousal_var.set(f"{self.arousal:.2f}")
        self.dominance_var.set(f"{self.dominance:.2f}")
        self.motif_var.set(", ".join(self.motif_tags))
        self.mood_var.set(mood)
        
        # Generate new EEG data
        self.generate_eeg_data()
        
        # Update plot
        self.ax.clear()
        if len(self.eeg_data) > 0:
            time_axis = np.linspace(0, len(self.eeg_data) / 10, len(self.eeg_data))
            self.ax.plot(time_axis, self.eeg_data, 'b-', linewidth=1)
            self.ax.set_title(f"EEG Signal - Mood: {mood}")
            self.ax.set_xlabel("Time (seconds)")
            self.ax.set_ylabel("Amplitude (μV)")
            self.ax.grid(True, alpha=0.3)
            self.ax.set_ylim(-50, 50)
        
        self.canvas.draw()
        
        # Update status
        self.status_var.set(f"Last update: {datetime.now().strftime('%H:%M:%S')} - {mood}")
    
    def start_demo(self):
        """Start the demo animation"""
        def animate(frame):
            self.update_display()
            self.root.after(1000, lambda: animate(frame + 1))  # Update every second
        
        animate(0)
    
    def run(self):
        """Run the demo"""
        self.root.mainloop()

def main():
    """Main function to run the demo"""
    print("Starting DreamWalk Neural Interface Demo...")
    print("This demo simulates real-time neural signal processing")
    print("and shows how brain activity translates to virtual environments.")
    print("\nPress Ctrl+C to stop the demo")
    
    try:
        demo = DreamWalkDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Error running demo: {e}")

if __name__ == "__main__":
    main()
