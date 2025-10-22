# How to Take Screenshots of DreamWalk Demo

## Web Dashboard Screenshot

The web demo is now running at: **http://localhost:8000**

### Steps to take a screenshot:

1. **Open your web browser** and navigate to: `http://localhost:8000`

2. **Wait for the page to load** - you should see:
   - "DreamWalk Neural Interface" header
   - Neural State panel with Valence, Arousal, Dominance metrics
   - EEG Signal Visualization chart
   - Real-time updating data

3. **Take a screenshot**:
   - **macOS**: Press `Cmd + Shift + 4`, then drag to select the browser window
   - **Windows**: Press `Win + Shift + S`, then select the browser window
   - **Linux**: Use `gnome-screenshot` or `scrot` command

4. **Save the screenshot** as `docs/screenshots/dashboard.png`

### What you should see in the screenshot:

- **Header**: "DreamWalk Neural Interface" with gradient background
- **Neural State Panel**: 
  - Valence: -1.00 to 1.00 (emotional tone)
  - Arousal: 0.00 to 1.00 (energy level)  
  - Dominance: -1.00 to 1.00 (control)
  - Neural Motifs: tags like "calm", "positive", "energetic"
  - Current Mood: "Calm and Content", "Excited and Joyful", etc.
- **EEG Chart**: Real-time waveform visualization
- **Status**: Last update timestamp and current mood
- **Connection Status**: "Connected" indicator in top-right

### Alternative: Desktop Demo Screenshot

If the web demo isn't working, you can also screenshot the desktop demo:

1. The desktop demo should be running in a separate window
2. Look for a tkinter window titled "DreamWalk - Neural Dreamscape Generator Demo"
3. Take a screenshot of that window instead

## Expected Visual Elements

The dashboard should show:
- **Professional gradient background** (blue to purple)
- **Real-time updating metrics** with green values
- **Animated EEG waveform** chart
- **Color-coded mood indicators**
- **Modern, clean interface** design

This screenshot will demonstrate that DreamWalk actually works and processes neural data in real-time!
