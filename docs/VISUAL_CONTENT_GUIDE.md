# Visual Content Guide for DreamWalk

This document outlines the visual content needed to showcase DreamWalk effectively.

## Required Screenshots

### 1. Web Dashboard (`docs/screenshots/dashboard.png`)
**What to capture:**
- Real-time EEG signal waveforms
- Emotion classification charts (valence, arousal, dominance)
- Neural motif detection display
- World state parameters (biome type, weather intensity, etc.)
- Service health monitoring
- Live metrics and performance data

**How to create:**
1. Run `./setup.sh` to start all services
2. Run `./run_demo.sh` to generate mock data
3. Open http://localhost:8000 in browser
4. Take screenshot of the dashboard showing live data

### 2. VR Environment (`docs/screenshots/vr_dreamscape.png`)
**What to capture:**
- Unity VR scene showing generated world
- Different biome types (forest, desert, mountains, etc.)
- Dynamic lighting and weather effects
- Procedural terrain and objects
- VR interface elements

**How to create:**
1. Import Unity project from `unity/DreamWalkVR/`
2. Set up VR headset (Quest 3, Vive, etc.)
3. Run the demo and capture VR footage
4. Take screenshots of different biome types

### 3. Neural Signal Processing (`docs/screenshots/eeg_signals.png`)
**What to capture:**
- Raw EEG waveforms from multiple channels
- Spectral analysis (frequency bands)
- Artifact detection and removal
- Feature extraction results
- Signal quality metrics

**How to create:**
1. Use mock EEG data or real OpenBCI device
2. Capture signal processing pipeline output
3. Show before/after artifact removal
4. Display feature extraction results

### 4. World Generation (`docs/screenshots/world_generation.png`)
**What to capture:**
- Procedural terrain generation
- Biome switching based on neural state
- Texture generation process
- Weather and lighting changes
- Object placement algorithms

**How to create:**
1. Show Unity scene generation in action
2. Capture different biome types
3. Demonstrate real-time morphing
4. Show texture generation process

## Additional Visual Content

### 5. Architecture Diagram (`docs/screenshots/architecture.png`)
**What to create:**
- Professional system architecture diagram
- Service communication flow
- Data pipeline visualization
- Technology stack overview

### 6. Demo Video (`docs/demo_video.mp4`)
**What to include:**
- Quick setup demonstration
- Live EEG processing
- World generation in action
- VR experience walkthrough
- Dashboard monitoring

### 7. Research Applications (`docs/screenshots/research.png`)
**What to show:**
- Neuroscience research use cases
- Therapeutic applications
- Educational visualization
- Creative expression examples

## Tools for Creating Visual Content

### Screenshots
- **macOS**: Screenshot app or Cmd+Shift+4
- **Windows**: Snipping Tool or Win+Shift+S
- **Linux**: GNOME Screenshot or scrot

### Screen Recording
- **macOS**: QuickTime Player or ScreenFlow
- **Windows**: OBS Studio or Camtasia
- **Linux**: OBS Studio or SimpleScreenRecorder

### Diagram Creation
- **Draw.io**: Free online diagram tool
- **Lucidchart**: Professional diagramming
- **Figma**: Design and prototyping
- **Mermaid**: Code-based diagrams

### Video Editing
- **DaVinci Resolve**: Professional free editor
- **Adobe Premiere**: Professional editing
- **Final Cut Pro**: macOS professional editor
- **OpenShot**: Free cross-platform editor

## Recommended Screenshot Specifications

### Dimensions
- **Web Dashboard**: 1920x1080 (Full HD)
- **VR Screenshots**: 1920x1080 or 4K
- **Mobile**: 1080x1920 (Portrait)

### Format
- **PNG**: For screenshots with transparency
- **JPG**: For photos and complex images
- **SVG**: For diagrams and logos

### Optimization
- Compress images for web
- Use appropriate file sizes
- Maintain aspect ratios
- Include alt text for accessibility

## Placeholder Images

Until real screenshots are created, you can use placeholder images:

```markdown
![Web Dashboard](https://via.placeholder.com/800x600/1a1a1a/ffffff?text=Web+Dashboard)
![VR Environment](https://via.placeholder.com/800x600/2d2d2d/ffffff?text=VR+Dreamscape)
![EEG Processing](https://via.placeholder.com/800x600/3d3d3d/ffffff?text=Neural+Signals)
![World Generation](https://via.placeholder.com/800x600/4d4d4d/ffffff?text=World+Generation)
```

## Next Steps

1. **Create screenshots** using the guide above
2. **Add images** to `docs/screenshots/` directory
3. **Update README** with actual image paths
4. **Create demo video** for maximum impact
5. **Add architecture diagram** for technical overview

## Impact of Visual Content

Visual content will significantly improve:
- **Project credibility** and professionalism
- **User understanding** of the system
- **GitHub repository** appearance and engagement
- **Research community** interest and adoption
- **Investor/employer** impression of technical skills
