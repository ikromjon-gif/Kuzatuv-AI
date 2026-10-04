# ⚽ Kuzatuv AI — Football Computer Vision Analytics

> **YOLO11 + ByteTrack pipeline for player tracking, team classification, ball tracking, and motion analysis.**

Kuzatuv AI converts recorded football video into structured tracking and motion statistics.

> **Current scope:** batch processing in Google Colab. This is not yet a real-time production system.

## Pipeline

```
Football Video
      |
      v
YOLO11 Detection
      |
 +----+-------------+
 |                  |
 v                  v
Players             Ball
 |                  |
 v                  v
ByteTrack       Ball Tracking
 |
 v
Persistent IDs
 |
 v
HSV Jersey Analysis
 |
 v
Team / Role Classification
 |
 v
Camera Motion Estimation
 |
 v
Camera-Compensated Motion
 |
 +-- Trajectory
 +-- Relative Speed
 +-- Acceleration
 +-- Direction
 |
 v
Annotated Video + CSV
```

## Features

- YOLO11 object detection
- ByteTrack multi-object tracking
- Persistent player IDs
- HSV-based jersey/team classification
- Ball tracking and smoothing
- Optical-flow camera-motion estimation
- Camera-cut detection
- Camera-compensated player motion
- Trajectories, relative speed, acceleration and direction
- Annotated MP4 output
- Per-player CSV statistics

## Output Metrics

| Metric | Meaning |
|---|---|
| Track_ID | ByteTrack identifier |
| Player_ID | Stable display ID |
| Team | Classified team/role |
| Frames_Tracked | Number of tracked frames |
| Distance_px | Relative pixel distance |
| Average_Relative_Speed_px_s | Average relative speed |
| Max_Relative_Speed_px_s | Maximum relative speed |
| Average_Acceleration_px_s2 | Average acceleration |
| Dominant_Direction | Most frequent direction |

**Metric caveat:** speed and distance are relative pixel measurements, not km/h or kilometers. Real-world measurement requires pitch calibration and camera geometry/homography.

## Stack

Python · YOLO11/Ultralytics · ByteTrack · OpenCV · NumPy · Pandas · PyTorch · Google Colab

## Run

```bash
pip install -q ultralytics opencv-python-headless pandas
```

Run the notebook/script in Colab and provide a recorded football video.

Typical outputs:

```
football_analytics_V2.mp4
football_player_statistics_V2.csv
```

## Computer Vision Components

**Detection:** YOLO11 identifies relevant objects frame by frame.

**Tracking:** ByteTrack maintains persistent identities.

**Team classification:** HSV analysis samples jersey regions using color heuristics.

**Camera compensation:** optical-flow motion estimation reduces camera-motion effects when calculating relative movement.

## Limitations

- Jersey-color classification is sensitive to lighting and occlusion.
- Tracking can degrade during heavy occlusion or missed detections.
- Ball tracking is affected by fast motion and occlusion.
- Camera compensation may fail during rapid movement or scene cuts.
- Current speed/distance values are pixel-based relative metrics.
- Current implementation is a Colab video-processing prototype.

## Roadmap

- Pitch calibration and real-world distance estimation
- Player heatmaps
- Possession and pass detection
- Shot detection
- Tactical formation analysis
- FastAPI inference service
- Web analytics dashboard
- Real-time processing

**Author:** Ikromjon Tojiboev