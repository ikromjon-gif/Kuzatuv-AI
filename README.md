# ⚽ Kuzatuv AI — Football Analytics

> **Computer vision for player tracking, team classification, ball tracking, and motion analysis.**

Kuzatuv AI is a **video-based football analytics prototype** built with YOLO11, ByteTrack, OpenCV, and HSV color analysis. The system processes recorded football match videos and generates an annotated output video with persistent player IDs, team classification, ball tracking, camera-motion compensation, trajectories, and relative motion statistics.

> **Current scope:** batch analysis of uploaded/recorded videos in Google Colab. This repository is not a real-time production system.

## 🚀 Features

- 👤 **Player Detection** — YOLO11 detects players in each frame.
- 🆔 **Persistent Player IDs** — ByteTrack maintains player identities across frames.
- 👕 **Team Classification** — HSV-based jersey color analysis identifies team categories.
- 🇺🇿 **Uzbekistan Team** — Blue jersey classification.
- 🇮🇷 **Iran Team** — White jersey classification.
- 🧤 **Goalkeeper Detection** — Light-green jersey classification.
- 🟥 **Referee Detection** — Red jersey classification.
- ⚽ **Ball Detection & Tracking** — Detects candidate football positions and maintains a smoothed trajectory.
- ☄️ **Ball Motion Visualization** — Meteor-style visual effect for the tracked ball.
- 🎯 **Player Trajectories** — Visualizes recent player movement paths.
- 🎥 **Annotated Video Output** — Exports a processed football video.
- 🎥 **Camera Motion Estimation** — Estimates camera translation and scale using optical flow and affine transformation.
- ✂️ **Camera Cut Detection** — Uses frame histogram differences to detect abrupt scene changes.
- 🏃 **Camera-Compensated Motion** — Estimates relative player movement after compensating for camera motion.
- 📈 **Acceleration & Direction** — Calculates relative acceleration and movement direction.
- 📊 **CSV Statistics** — Exports per-player tracking and motion statistics.

## 🧠 Processing Pipeline

```text
Football Video
      │
      ▼
YOLO11 Object Detection
      │
      ├───────────────┐
      ▼               ▼
Player Detection    Ball Detection
      │               │
      ▼               ▼
ByteTrack         Ball Tracking
      │               │
      ▼               ▼
Persistent IDs     Smoothing / Motion
      │
      ▼
HSV Jersey Color Classification
      │
      ▼
Team / Role Classification
      │
      ▼
Camera Motion Estimation
      │
      ▼
Camera-Compensated Player Motion
      │
      ├── Trajectory
      ├── Relative Speed
      ├── Acceleration
      └── Direction
      │
      ▼
OpenCV Visualization
      │
      ├── Annotated Video
      └── CSV Player Statistics
```

## 📊 Output

For each sufficiently tracked player, the generated CSV can include:

| Metric | Description |
|---|---|
| `Track_ID` | Original ByteTrack identifier |
| `Player_ID` | Stable display number |
| `Team` | Classified team/role |
| `Frames_Tracked` | Number of processed frames for the player |
| `Distance_px` | Relative distance in pixels |
| `Average_Relative_Speed_px_s` | Average relative speed in pixels/second |
| `Max_Relative_Speed_px_s` | Maximum relative speed |
| `Average_Acceleration_px_s2` | Average absolute acceleration |
| `Max_Acceleration_px_s2` | Maximum absolute acceleration |
| `Dominant_Direction` | Most frequent movement direction |

### ⚠️ Important metric limitation

**Speed and distance are relative pixel measurements.** They are not real-world km/h or kilometers. Converting them to physical units requires pitch calibration, camera geometry, and a suitable spatial reference such as homography.

## 🛠️ Technology Stack

- **Python**
- **YOLO11 / Ultralytics** — object detection and tracking
- **ByteTrack** — multi-object tracking
- **OpenCV** — video processing, optical flow, camera estimation, and visualization
- **HSV Color Analysis** — jersey/team classification
- **NumPy** — numerical and coordinate calculations
- **Pandas** — player statistics and CSV export
- **PyTorch** — GPU availability/device selection
- **Google Colab** — execution environment and GPU acceleration

## ⚙️ How It Works

1. Upload a recorded football match video in Google Colab.
2. The script reads the video metadata and selects GPU when available.
3. YOLO11 detects and tracks objects using ByteTrack.
4. Player jersey regions are analyzed in HSV color space.
5. Stable player IDs are assigned for visualization.
6. Camera motion is estimated from optical flow between frames.
7. Player movement is compensated for estimated camera motion.
8. Trajectories, relative speed, acceleration, and direction are calculated.
9. The ball is smoothed and rendered with a motion effect.
10. The final annotated MP4 and player-statistics CSV are exported.

## 💻 Run in Google Colab

The current implementation is designed as a **Google Colab notebook/script workflow**.

Install dependencies:

```bash
pip install -q ultralytics opencv-python-headless pandas
```

Then run the notebook cells and upload a football video when prompted.

The script produces:

```text
/content/football_analytics_V2.mp4
/content/football_player_statistics_V2.csv
```

## 🎯 Example Use Cases

- Football match video analysis
- Player tracking research
- Sports computer vision experiments
- Team/role classification research
- Camera-motion-aware movement analysis
- AI-based sports analytics prototypes

## 🔮 Roadmap

Potential future development for Kuzatuv AI includes:

- 📊 Player heatmaps
- ⚽ Ball possession estimation
- 🔄 Pass detection
- 🎯 Shot detection
- 🗺️ Tactical formation analysis
- 📈 Advanced match statistics
- 🏟️ Pitch calibration and real-world distance/speed estimation
- 🌐 FastAPI backend
- 🖥️ Web analytics dashboard
- ⚡ Real-time processing

## ⚠️ Limitations

- Team classification is based on jersey-color heuristics and can be affected by lighting, shadows, occlusion, and similar colors.
- Camera compensation is an estimate and may become unreliable during rapid camera movement or scene cuts.
- Ball tracking can fail during occlusion, very fast motion, or ambiguous detections.
- Current speed and distance values are **pixel-based relative metrics**, not physical measurements.
- The current implementation is a **recorded-video prototype running in Google Colab**, not a deployed real-time service.

## 👨‍💻 Author

**Ikromjon Tojiboev**  
Master's in Computer Engineering  
AI / Machine Learning / Computer Vision

---

⭐ **Kuzatuv AI** is an evolving sports-computer-vision project focused on turning football video into structured player and motion analytics.