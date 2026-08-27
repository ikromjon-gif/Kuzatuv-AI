# ============================================================
# ⚽ AI FOOTBALL ANALYTICS V2.1
#
# YOLO11 + ByteTrack
# HSV Team Classification
# Stable Player IDs
# Ball Tracking + Meteor Effect
#
# V2.1:
# - Camera motion estimation
# - Pan / zoom detection
# - Camera cut detection
# - Player trajectories
# - Camera-compensated relative speed
# - Acceleration
# - Direction
# - Pixel distance
# - Player CSV statistics
#
# IMPORTANT:
# Speed / distance are relative pixel metrics.
# Real km/h and km require pitch calibration.
# ============================================================


# ============================================================
# 1. INSTALL
# ============================================================

!pip install -q ultralytics opencv-python-headless pandas


# ============================================================
# 2. IMPORTS
# ============================================================

import cv2
import math
import torch
import numpy as np
import pandas as pd

from collections import deque, defaultdict
from google.colab import files
from ultralytics import YOLO


# ============================================================
# 3. CONFIG
# ============================================================

MODEL_NAME = "yolo11x.pt"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

PERSON_CONF = 0.25
BALL_CONF = 0.10


# ------------------------------------------------------------
# Trajectory
# ------------------------------------------------------------

MAX_TRAJECTORY_POINTS = 35


# ------------------------------------------------------------
# Camera motion
# ------------------------------------------------------------

CAMERA_MIN_FEATURES = 30
CAMERA_MIN_INLIERS = 12

CAMERA_MAX_TRANSLATION = 250.0

CAMERA_MIN_SCALE = 0.75
CAMERA_MAX_SCALE = 1.35

CAMERA_CUT_HIST_THRESHOLD = 0.55


# ------------------------------------------------------------
# Motion
# ------------------------------------------------------------

MAX_RELATIVE_SPEED = 2500.0


# ------------------------------------------------------------
# Ball
# ------------------------------------------------------------

MAX_BALL_JUMP = 180

BALL_SMOOTHING = 0.65

BALL_MAX_MISSING = 4


print("=" * 70)
print("⚽ AI FOOTBALL ANALYTICS V2.1")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 4. UPLOAD VIDEO
# ============================================================

print("\n🎥 Upload football video")

uploaded = files.upload()

VIDEO_PATH = list(
    uploaded.keys()
)[0]

print(
    "VIDEO:",
    VIDEO_PATH
)


# ============================================================
# 5. VIDEO INFO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

FPS = cap.get(
    cv2.CAP_PROP_FPS
)

WIDTH = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

HEIGHT = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

TOTAL_FRAMES = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

cap.release()


if FPS <= 0:
    FPS = 30.0


DURATION = (
    TOTAL_FRAMES / FPS
)


print("\nVIDEO INFO")

print(
    "FPS:",
    FPS
)

print(
    "Resolution:",
    WIDTH,
    "x",
    HEIGHT
)

print(
    "Frames:",
    TOTAL_FRAMES
)

print(
    "Duration:",
    round(
        DURATION,
        1
    ),
    "seconds"
)


# ============================================================
# 6. LOAD YOLO
# ============================================================

print(
    "\n🚀 Loading YOLO11..."
)

model = YOLO(
    MODEL_NAME
)

print(
    "YOLO READY!"
)


# ============================================================
# 7. OUTPUT
# ============================================================

OUTPUT_PATH = (
    "/content/"
    "football_analytics_V2.mp4"
)

CSV_PATH = (
    "/content/"
    "football_player_statistics_V2.csv"
)


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


out = cv2.VideoWriter(

    OUTPUT_PATH,

    fourcc,

    FPS,

    (
        WIDTH,
        HEIGHT
    )
)


# ============================================================
# 8. COLORS
# ============================================================

COLORS = {

    "IRAN":
        (245, 245, 245),

    "UZBEKISTAN":
        (255, 80, 0),

    "GOALKEEPER":
        (120, 255, 120),

    "REFEREE":
        (40, 40, 255),

    "BALL":
        (0, 220, 255),

    "UNKNOWN":
        (0, 220, 255),

    "WHITE":
        (255, 255, 255),

    "GREEN":
        (50, 220, 80),

    "YELLOW":
        (0, 220, 255)
}


# ============================================================
# 9. PLAYER MEMORY
# ============================================================

PLAYER_HISTORY = {}

PLAYER_STABLE_CLASS = {}

PLAYER_DISPLAY_NUMBER = {}

NEXT_PLAYER_NUMBER = 1


# ============================================================
# 10. PLAYER MOTION MEMORY
# ============================================================

PLAYER_TRAJECTORIES = defaultdict(

    lambda: deque(
        maxlen=MAX_TRAJECTORY_POINTS
    )
)

PLAYER_LAST_POSITION = {}

PLAYER_LAST_TIME = {}

PLAYER_LAST_SPEED = defaultdict(
    float
)


# ============================================================
# 11. PLAYER STATISTICS
# ============================================================

PLAYER_STATS = defaultdict(

    lambda: {

        "team":
            "UNKNOWN",

        "frames":
            0,

        "distance_px":
            0.0,

        "speed_sum_px_s":
            0.0,

        "speed_samples":
            0,

        "max_speed_px_s":
            0.0,

        "acceleration_sum":
            0.0,

        "acceleration_samples":
            0,

        "max_acceleration":
            0.0,

        "directions":
            defaultdict(int),

        "last_direction":
            "STATIONARY"
    }
)


# ============================================================
# 12. CAMERA MEMORY
# ============================================================

PREVIOUS_FRAME = None

PREVIOUS_GRAY = None


# IMPORTANT:
# Initialize these BEFORE processing.

camera_matrix = np.eye(
    2,
    3,
    dtype=np.float32
)

camera_status = "INITIALIZING"

camera_scale = 1.0

camera_dx = 0.0

camera_dy = 0.0

CAMERA_STATUS = "INITIALIZING"

is_camera_cut = False


# ============================================================
# 13. BALL MEMORY
# ============================================================

BALL_POSITION = None

BALL_VELOCITY = np.array(
    [0.0, 0.0],
    dtype=np.float32
)

BALL_MISSING = 0


# ============================================================
# 14. JERSEY REGION
# ============================================================

def get_jersey_region(

    frame,
    x1,
    y1,
    x2,
    y2

):

    w = x2 - x1
    h = y2 - y1


    if w < 10 or h < 20:

        return None


    rx1 = x1 + int(
        w * 0.18
    )

    rx2 = x2 - int(
        w * 0.18
    )

    ry1 = y1 + int(
        h * 0.20
    )

    ry2 = y1 + int(
        h * 0.55
    )


    rx1 = max(
        0,
        rx1
    )

    ry1 = max(
        0,
        ry1
    )

    rx2 = min(
        frame.shape[1],
        rx2
    )

    ry2 = min(
        frame.shape[0],
        ry2
    )


    crop = frame[
        ry1:ry2,
        rx1:rx2
    ]


    if crop.size == 0:

        return None


    return crop


# ============================================================
# 15. TEAM CLASSIFICATION
# ============================================================

def classify_player(

    frame,
    x1,
    y1,
    x2,
    y2

):

    crop = get_jersey_region(

        frame,
        x1,
        y1,
        x2,
        y2
    )


    if crop is None:

        return "UNKNOWN"


    hsv = cv2.cvtColor(

        crop,

        cv2.COLOR_BGR2HSV
    )


    total = (

        crop.shape[0]
        *
        crop.shape[1]
    )


    if total == 0:

        return "UNKNOWN"


    # WHITE
    white_mask = cv2.inRange(

        hsv,

        np.array([
            0,
            0,
            160
        ]),

        np.array([
            180,
            75,
            255
        ])
    )


    # BLUE
    blue_mask = cv2.inRange(

        hsv,

        np.array([
            90,
            70,
            40
        ]),

        np.array([
            145,
            255,
            255
        ])
    )


    # GREEN
    green_mask = cv2.inRange(

        hsv,

        np.array([
            35,
            35,
            70
        ]),

        np.array([
            90,
            255,
            255
        ])
    )


    # RED
    red1 = cv2.inRange(

        hsv,

        np.array([
            0,
            70,
            60
        ]),

        np.array([
            12,
            255,
            255
        ])
    )


    red2 = cv2.inRange(

        hsv,

        np.array([
            165,
            70,
            60
        ]),

        np.array([
            180,
            255,
            255
        ])
    )


    red_mask = cv2.bitwise_or(
        red1,
        red2
    )


    scores = {

        "IRAN":
            cv2.countNonZero(
                white_mask
            ) / total,

        "UZBEKISTAN":
            cv2.countNonZero(
                blue_mask
            ) / total,

        "GOALKEEPER":
            cv2.countNonZero(
                green_mask
            ) / total,

        "REFEREE":
            cv2.countNonZero(
                red_mask
            ) / total
    }


    best_class = max(

        scores,

        key=scores.get
    )


    if scores[
        best_class
    ] < 0.12:

        return "UNKNOWN"


    return best_class


# ============================================================
# 16. STABLE TEAM CLASS
# ============================================================

def get_stable_class(

    track_id,
    detected_class

):

    if track_id == -1:

        return detected_class


    if track_id not in PLAYER_HISTORY:

        PLAYER_HISTORY[
            track_id
        ] = {

            "IRAN": 0,

            "UZBEKISTAN": 0,

            "GOALKEEPER": 0,

            "REFEREE": 0,

            "UNKNOWN": 0
        }


    PLAYER_HISTORY[
        track_id
    ][
        detected_class
    ] += 1


    valid_classes = [

        "IRAN",
        "UZBEKISTAN",
        "GOALKEEPER",
        "REFEREE"
    ]


    best_class = max(

        valid_classes,

        key=lambda x:

        PLAYER_HISTORY[
            track_id
        ][x]
    )


    best_count = (

        PLAYER_HISTORY[
            track_id
        ][best_class]
    )


    if best_count < 2:

        stable = detected_class

    else:

        stable = best_class


    PLAYER_STABLE_CLASS[
        track_id
    ] = stable


    return stable


# ============================================================
# 17. DISPLAY ID
# ============================================================

def get_display_number(
    track_id
):

    global NEXT_PLAYER_NUMBER


    if track_id == -1:

        return 0


    if track_id not in PLAYER_DISPLAY_NUMBER:

        PLAYER_DISPLAY_NUMBER[
            track_id
        ] = NEXT_PLAYER_NUMBER


        NEXT_PLAYER_NUMBER += 1


    return PLAYER_DISPLAY_NUMBER[
        track_id
    ]


# ============================================================
# 18. HISTOGRAM CAMERA CUT
# ============================================================

def histogram_distance(

    frame_a,
    frame_b

):

    hsv_a = cv2.cvtColor(
        frame_a,
        cv2.COLOR_BGR2HSV
    )

    hsv_b = cv2.cvtColor(
        frame_b,
        cv2.COLOR_BGR2HSV
    )


    hist_a = cv2.calcHist(

        [hsv_a],

        [0, 1],

        None,

        [32, 32],

        [0, 180, 0, 256]
    )


    hist_b = cv2.calcHist(

        [hsv_b],

        [0, 1],

        None,

        [32, 32],

        [0, 180, 0, 256]
    )


    cv2.normalize(
        hist_a,
        hist_a
    )

    cv2.normalize(
        hist_b,
        hist_b
    )


    correlation = cv2.compareHist(

        hist_a,

        hist_b,

        cv2.HISTCMP_CORREL
    )


    return (
        1.0 -
        correlation
    )


# ============================================================
# 19. CAMERA MOTION
# ============================================================

def estimate_camera_motion(

    previous_gray,
    current_gray

):

    # --------------------------------------------------------
    # First frame
    # --------------------------------------------------------

    if previous_gray is None:

        return (

            np.eye(
                2,
                3,
                dtype=np.float32
            ),

            "INITIALIZING",

            1.0,

            0.0,

            0.0,

            False
        )


    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    points_prev = cv2.goodFeaturesToTrack(

        previous_gray,

        maxCorners=500,

        qualityLevel=0.01,

        minDistance=10,

        blockSize=7
    )


    if (

        points_prev is None
        or
        len(points_prev)
        < CAMERA_MIN_FEATURES

    ):

        return (

            np.eye(
                2,
                3,
                dtype=np.float32
            ),

            "LOW_FEATURES",

            1.0,

            0.0,

            0.0,

            False
        )


    # --------------------------------------------------------
    # Optical flow
    # --------------------------------------------------------

    points_curr, status, error = (

        cv2.calcOpticalFlowPyrLK(

            previous_gray,

            current_gray,

            points_prev,

            None,

            winSize=(
                21,
                21
            ),

            maxLevel=3,

            criteria=(

                cv2.TERM_CRITERIA_EPS
                |
                cv2.TERM_CRITERIA_COUNT,

                30,

                0.01
            )
        )
    )


    if points_curr is None:

        return (

            np.eye(
                2,
                3,
                dtype=np.float32
            ),

            "NO_FLOW",

            1.0,

            0.0,

            0.0,

            False
        )


    good_prev = points_prev[
        status.flatten() == 1
    ]


    good_curr = points_curr[
        status.flatten() == 1
    ]


    if len(good_prev) < CAMERA_MIN_FEATURES:

        return (

            np.eye(
                2,
                3,
                dtype=np.float32
            ),

            "LOW_INLIERS",

            1.0,

            0.0,

            0.0,

            False
        )


    # --------------------------------------------------------
    # Affine camera transform
    # --------------------------------------------------------

    matrix, inliers = (

        cv2.estimateAffinePartial2D(

            good_prev,

            good_curr,

            method=cv2.RANSAC,

            ransacReprojThreshold=3.0
        )
    )


    if matrix is None:

        return (

            np.eye(
                2,
                3,
                dtype=np.float32
            ),

            "FAILED",

            1.0,

            0.0,

            0.0,

            False
        )


    if inliers is None:

        inlier_count = 0

    else:

        inlier_count = int(
            inliers.sum()
        )


    if inlier_count < CAMERA_MIN_INLIERS:

        return (

            matrix.astype(
                np.float32
            ),

            "UNRELIABLE",

            1.0,

            float(
                matrix[0, 2]
            ),

            float(
                matrix[1, 2]
            ),

            False
        )


    a = float(
        matrix[0, 0]
    )

    b = float(
        matrix[0, 1]
    )


    scale = math.sqrt(

        a * a +
        b * b
    )


    dx = float(
        matrix[0, 2]
    )

    dy = float(
        matrix[1, 2]
    )


    reliable = True


    if (

        abs(dx)
        >
        CAMERA_MAX_TRANSLATION

        or

        abs(dy)
        >
        CAMERA_MAX_TRANSLATION

        or

        scale < CAMERA_MIN_SCALE

        or

        scale > CAMERA_MAX_SCALE

    ):

        status_text = (
            "LARGE_CAMERA_MOVE"
        )

        reliable = False

    else:

        status_text = (
            "CAMERA_MOVING"
        )


    return (

        matrix.astype(
            np.float32
        ),

        status_text,

        scale,

        dx,

        dy,

        reliable
    )


# ============================================================
# 20. CAMERA-COMPENSATED MOTION
# ============================================================

def get_camera_compensated_motion(

    previous_point,
    current_point,
    camera_matrix

):

    if (

        previous_point is None
        or
        current_point is None

    ):

        return None


    try:

        inverse_matrix = (

            cv2.invertAffineTransform(

                camera_matrix
            )
        )

    except:

        return None


    current = np.array(

        [
            current_point[0],
            current_point[1],
            1.0
        ],

        dtype=np.float32
    )


    compensated = (

        inverse_matrix @
        current
    )


    compensated_x = float(
        compensated[0]
    )

    compensated_y = float(
        compensated[1]
    )


    dx = (

        compensated_x
        -
        previous_point[0]
    )


    dy = (

        compensated_y
        -
        previous_point[1]
    )


    return (
        dx,
        dy
    )


# ============================================================
# 21. DIRECTION
# ============================================================

def get_direction(

    dx,
    dy,

    threshold=2.0

):

    if (

        abs(dx) < threshold
        and
        abs(dy) < threshold

    ):

        return "STATIONARY"


    angle = math.degrees(

        math.atan2(
            -dy,
            dx
        )
    )


    if angle < 0:

        angle += 360


    if (
        angle >= 337.5
        or
        angle < 22.5
    ):

        return "RIGHT"


    if angle < 67.5:

        return "UP-RIGHT"


    if angle < 112.5:

        return "UP"


    if angle < 157.5:

        return "UP-LEFT"


    if angle < 202.5:

        return "LEFT"


    if angle < 247.5:

        return "DOWN-LEFT"


    if angle < 292.5:

        return "DOWN"


    return "DOWN-RIGHT"


# ============================================================
# 22. PLAYER MOTION
# ============================================================

def update_player_motion(

    track_id,
    team,
    position,
    frame_number,
    camera_matrix,
    camera_reliable

):

    if track_id == -1:

        return {

            "speed": 0.0,

            "acceleration": 0.0,

            "direction":
                "UNKNOWN",

            "valid":
                False
        }


    stats = PLAYER_STATS[
        track_id
    ]


    stats[
        "team"
    ] = team


    stats[
        "frames"
    ] += 1


    current_time = (

        frame_number /
        FPS
    )


    previous_position = (

        PLAYER_LAST_POSITION.get(
            track_id
        )
    )


    previous_time = (

        PLAYER_LAST_TIME.get(
            track_id
        )
    )


    # --------------------------------------------------------
    # First detection
    # --------------------------------------------------------

    if previous_position is None:

        PLAYER_LAST_POSITION[
            track_id
        ] = position

        PLAYER_LAST_TIME[
            track_id
        ] = current_time

        PLAYER_TRAJECTORIES[
            track_id
        ].append(position)


        return {

            "speed": 0.0,

            "acceleration": 0.0,

            "direction":
                "STATIONARY",

            "valid":
                False
        }


    dt = (

        current_time
        -
        previous_time
    )


    if (

        dt <= 0
        or
        dt > 1.0

    ):

        PLAYER_LAST_POSITION[
            track_id
        ] = position

        PLAYER_LAST_TIME[
            track_id
        ] = current_time


        return {

            "speed": 0.0,

            "acceleration": 0.0,

            "direction":
                "RESET",

            "valid":
                False
        }


    # --------------------------------------------------------
    # CAMERA CUT
    # --------------------------------------------------------

    if is_camera_cut:

        PLAYER_LAST_POSITION[
            track_id
        ] = position

        PLAYER_LAST_TIME[
            track_id
        ] = current_time

        PLAYER_LAST_SPEED[
            track_id
        ] = 0.0

        PLAYER_TRAJECTORIES[
            track_id
        ].clear()

        PLAYER_TRAJECTORIES[
            track_id
        ].append(position)


        return {

            "speed": 0.0,

            "acceleration": 0.0,

            "direction":
                "CAMERA CUT",

            "valid":
                False
        }


    # --------------------------------------------------------
    # If camera estimation is unreliable:
    #
    # DO NOT claim precise player speed.
    # --------------------------------------------------------

    if not camera_reliable:

        PLAYER_LAST_POSITION[
            track_id
        ] = position

        PLAYER_LAST_TIME[
            track_id
        ] = current_time

        PLAYER_LAST_SPEED[
            track_id
        ] = 0.0

        PLAYER_TRAJECTORIES[
            track_id
        ].append(position)


        return {

            "speed": 0.0,

            "acceleration": 0.0,

            "direction":
                "CAMERA",

            "valid":
                False
        }


    # --------------------------------------------------------
    # Camera compensated movement
    # --------------------------------------------------------

    motion = get_camera_compensated_motion(

        previous_position,

        position,

        camera_matrix
    )


    if motion is None:

        PLAYER_LAST_POSITION[
            track_id
        ] = position

        PLAYER_LAST_TIME[
            track_id
        ] = current_time

        return {

            "speed": 0.0,

            "acceleration": 0.0,

            "direction":
                "UNKNOWN",

            "valid":
                False
        }


    dx, dy = motion


    displacement = math.sqrt(

        dx * dx +
        dy * dy
    )


    speed = (

        displacement /
        dt
    )


    # --------------------------------------------------------
    # Impossible jump
    # --------------------------------------------------------

    if speed > MAX_RELATIVE_SPEED:

        PLAYER_LAST_POSITION[
            track_id
        ] = position

        PLAYER_LAST_TIME[
            track_id
        ] = current_time

        PLAYER_LAST_SPEED[
            track_id
        ] = 0.0


        return {

            "speed": 0.0,

            "acceleration": 0.0,

            "direction":
                "JUMP",

            "valid":
                False
        }


    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    previous_speed = PLAYER_LAST_SPEED[
        track_id
    ]


    acceleration = (

        speed -
        previous_speed
    ) / dt


    direction = get_direction(

        dx,
        dy
    )


    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    stats[
        "distance_px"
    ] += displacement


    stats[
        "speed_sum_px_s"
    ] += speed


    stats[
        "speed_samples"
    ] += 1


    stats[
        "max_speed_px_s"
    ] = max(

        stats[
            "max_speed_px_s"
        ],

        speed
    )


    stats[
        "acceleration_sum"
    ] += abs(
        acceleration
    )


    stats[
        "acceleration_samples"
    ] += 1


    stats[
        "max_acceleration"
    ] = max(

        stats[
            "max_acceleration"
        ],

        abs(
            acceleration
        )
    )


    stats[
        "directions"
    ][
        direction
    ] += 1


    stats[
        "last_direction"
    ] = direction


    # --------------------------------------------------------
    # Save state
    # --------------------------------------------------------

    PLAYER_LAST_POSITION[
        track_id
    ] = position

    PLAYER_LAST_TIME[
        track_id
    ] = current_time

    PLAYER_LAST_SPEED[
        track_id
    ] = speed


    PLAYER_TRAJECTORIES[
        track_id
    ].append(position)


    return {

        "speed":
            speed,

        "acceleration":
            acceleration,

        "direction":
            direction,

        "valid":
            True
    }


# ============================================================
# 23. DRAW PLAYER CIRCLE
# ============================================================

def draw_player_circle(

    frame,

    x1,
    y1,
    x2,
    y2,

    color

):

    player_w = (
        x2 - x1
    )


    foot_x = int(

        (
            x1 +
            x2
        ) / 2
    )


    foot_y = int(
        y2
    )


    ellipse_w = max(

        10,

        int(
            player_w *
            0.32
        )
    )


    ellipse_h = max(

        3,

        int(
            ellipse_w *
            0.28
        )
    )


    cv2.ellipse(

        frame,

        (
            foot_x,
            foot_y
        ),

        (
            ellipse_w,
            ellipse_h
        ),

        0,

        0,

        360,

        color,

        2,

        cv2.LINE_AA
    )


# ============================================================
# 24. TRAJECTORY
# ============================================================

def draw_player_trajectory(

    frame,

    track_id,

    color

):

    points = list(

        PLAYER_TRAJECTORIES[
            track_id
        ]
    )


    if len(points) < 2:

        return


    for i in range(

        1,
        len(points)
    ):

        p1 = (

            int(
                points[i - 1][0]
            ),

            int(
                points[i - 1][1]
            )
        )


        p2 = (

            int(
                points[i][0]
            ),

            int(
                points[i][1]
            )
        )


        cv2.line(

            frame,

            p1,

            p2,

            color,

            2,

            cv2.LINE_AA
        )


# ============================================================
# 25. ID BADGE
# ============================================================

def draw_id_badge(

    frame,

    x1,
    y1,
    x2,

    player_class,

    display_number

):

    color = COLORS.get(

        player_class,

        COLORS[
            "UNKNOWN"
        ]
    )


    if player_class == "IRAN":

        label = (
            f"IRN "
            f"{display_number}"
        )

    elif player_class == "UZBEKISTAN":

        label = (
            f"UZB "
            f"{display_number}"
        )

    elif player_class == "GOALKEEPER":

        label = "GK"

    elif player_class == "REFEREE":

        label = "REF"

    else:

        label = (
            f"P "
            f"{display_number}"
        )


    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )


    scale = 0.52
    thickness = 2


    (
        text_w,
        text_h
    ), _ = cv2.getTextSize(

        label,

        font,

        scale,

        thickness
    )


    pad_x = 9
    pad_y = 6


    badge_w = (
        text_w +
        pad_x * 2
    )

    badge_h = (
        text_h +
        pad_y * 2
    )


    center_x = int(

        (
            x1 +
            x2
        ) / 2
    )


    bx1 = (
        center_x -
        badge_w // 2
    )


    by2 = max(

        badge_h + 5,

        y1 - 8
    )


    bx2 = (
        bx1 +
        badge_w
    )


    by1 = (
        by2 -
        badge_h
    )


    cv2.rectangle(

        frame,

        (
            bx1,
            by1
        ),

        (
            bx2,
            by2
        ),

        (
            18,
            18,
            18
        ),

        -1
    )


    cv2.rectangle(

        frame,

        (
            bx1,
            by1
        ),

        (
            bx2,
            by2
        ),

        color,

        2
    )


    cv2.putText(

        frame,

        label,

        (
            bx1 + pad_x,

            by2 - pad_y
        ),

        font,

        scale,

        COLORS[
            "WHITE"
        ],

        thickness,

        cv2.LINE_AA
    )


# ============================================================
# 26. BALL SELECTION
# ============================================================

def select_best_ball(

    candidates,

    previous_position

):

    if len(candidates) == 0:

        return None


    if previous_position is None:

        return max(

            candidates,

            key=lambda x:
            x[2]
        )


    best = None

    best_score = -999999


    for (

        x,
        y,
        conf

    ) in candidates:


        distance = math.dist(

            previous_position,

            (
                x,
                y
            )
        )


        score = (

            conf * 100
            -
            distance * 0.08
        )


        if score > best_score:

            best_score = score

            best = (
                x,
                y,
                conf
            )


    return best


# ============================================================
# 27. METEOR BALL
# ============================================================

def draw_meteor_ball(

    frame,

    position,

    velocity

):

    if position is None:

        return


    x = int(
        position[0]
    )

    y = int(
        position[1]
    )


    vx = velocity[0]

    vy = velocity[1]


    speed = math.sqrt(

        vx * vx +
        vy * vy
    )


    if speed > 2:

        dx = (
            vx /
            speed
        )

        dy = (
            vy /
            speed
        )


        tail_length = min(

            45,

            max(
                15,
                int(
                    speed * 3
                )
            )
        )


        for i in range(
            6,
            0,
            -1
        ):

            ratio = (
                i /
                6.0
            )


            tx = int(

                x -
                dx *
                tail_length *
                ratio
            )


            ty = int(

                y -
                dy *
                tail_length *
                ratio
            )


            radius = max(

                2,

                int(
                    7 *
                    (
                        1 -
                        ratio
                    )
                )
            )


            cv2.circle(

                frame,

                (
                    tx,
                    ty
                ),

                radius + 2,

                (
                    0,
                    100,
                    255
                ),

                -1,

                cv2.LINE_AA
            )


            cv2.circle(

                frame,

                (
                    tx,
                    ty
                ),

                radius,

                (
                    0,
                    220,
                    255
                ),

                -1,

                cv2.LINE_AA
            )


    cv2.circle(

        frame,

        (
            x,
            y
        ),

        11,

        (
            0,
            150,
            255
        ),

        2,

        cv2.LINE_AA
    )


    cv2.circle(

        frame,

        (
            x,
            y
        ),

        7,

        (
            0,
            230,
            255
        ),

        -1,

        cv2.LINE_AA
    )


    cv2.circle(

        frame,

        (
            x,
            y
        ),

        3,

        (
            255,
            255,
            255
        ),

        -1,

        cv2.LINE_AA
    )


# ============================================================
# 28. TIME BADGE
# ============================================================

def draw_time_badge(

    frame,

    frame_number

):

    current_seconds = (

        frame_number /
        FPS
    )


    minutes = int(
        current_seconds // 60
    )

    seconds = int(
        current_seconds % 60
    )


    total_minutes = int(
        DURATION // 60
    )

    total_seconds = int(
        DURATION % 60
    )


    text = (

        f"{minutes:02d}:"
        f"{seconds:02d} / "
        f"{total_minutes:02d}:"
        f"{total_seconds:02d}"
    )


    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )


    scale = 0.55
    thickness = 1


    (
        tw,
        th
    ), _ = cv2.getTextSize(

        text,

        font,

        scale,

        thickness
    )


    x1 = (
        WIDTH -
        tw -
        30
    )

    y1 = (
        HEIGHT -
        42
    )

    x2 = (
        WIDTH -
        10
    )

    y2 = (
        HEIGHT -
        10
    )


    cv2.rectangle(

        frame,

        (
            x1,
            y1
        ),

        (
            x2,
            y2
        ),

        (
            20,
            20,
            20
        ),

        -1
    )


    cv2.putText(

        frame,

        text,

        (
            x1 + 10,
            y2 - 10
        ),

        font,

        scale,

        COLORS[
            "WHITE"
        ],

        thickness,

        cv2.LINE_AA
    )


# ============================================================
# 29. CAMERA STATUS
# ============================================================

def draw_camera_status(
    frame
):

    if CAMERA_STATUS == "CAMERA_MOVING":

        color = COLORS[
            "GREEN"
        ]

    elif CAMERA_STATUS == "CAMERA CUT":

        color = (
            0,
            0,
            255
        )

    else:

        color = (
            0,
            180,
            255
        )


    text = (
        f"CAM: "
        f"{CAMERA_STATUS}"
    )


    cv2.rectangle(

        frame,

        (
            WIDTH - 275,
            15
        ),

        (
            WIDTH - 15,
            52
        ),

        (
            15,
            15,
            15
        ),

        -1
    )


    cv2.putText(

        frame,

        text,

        (
            WIDTH - 260,
            40
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        color,

        1,

        cv2.LINE_AA
    )


# ============================================================
# 30. START TRACKING
# ============================================================

print("\n" + "=" * 70)

print(
    "🚀 STARTING AI FOOTBALL ANALYTICS V2.1"
)

print("=" * 70)


frame_number = 0


results = model.track(

    source=VIDEO_PATH,

    tracker="bytetrack.yaml",

    persist=True,

    stream=True,

    verbose=False,

    device=DEVICE,

    conf=PERSON_CONF
)


# ============================================================
# 31. PROCESS VIDEO
# ============================================================

for result in results:

    frame_number += 1


    frame = result.orig_img.copy()


    # ========================================================
    # CAMERA ANALYSIS
    # ========================================================

    current_gray = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2GRAY
    )


    (
        camera_matrix,
        camera_status,
        camera_scale,
        camera_dx,
        camera_dy,
        camera_reliable
    ) = estimate_camera_motion(

        PREVIOUS_GRAY,

        current_gray
    )


    # ========================================================
    # CAMERA CUT
    # ========================================================

    is_camera_cut = False


    if PREVIOUS_FRAME is not None:

        try:

            hist_dist = histogram_distance(

                PREVIOUS_FRAME,

                frame
            )


            if (
                hist_dist
                >
                CAMERA_CUT_HIST_THRESHOLD
            ):

                is_camera_cut = True

        except Exception:

            is_camera_cut = False


    if is_camera_cut:

        CAMERA_STATUS = (
            "CAMERA CUT"
        )


        PLAYER_LAST_POSITION.clear()

        PLAYER_LAST_TIME.clear()

        PLAYER_LAST_SPEED.clear()


        for track_id in PLAYER_TRAJECTORIES:

            PLAYER_TRAJECTORIES[
                track_id
            ].clear()


    else:

        CAMERA_STATUS = camera_status


    PREVIOUS_GRAY = (
        current_gray.copy()
    )

    PREVIOUS_FRAME = (
        frame.copy()
    )


    # ========================================================
    # COUNTERS
    # ========================================================

    iran_count = 0

    uzbekistan_count = 0

    goalkeeper_count = 0

    referee_count = 0


    ball_candidates = []


    # ========================================================
    # DETECTIONS
    # ========================================================

    if result.boxes is not None:

        boxes = (

            result.boxes
            .xyxy
            .cpu()
            .numpy()
        )


        classes = (

            result.boxes
            .cls
            .int()
            .cpu()
            .tolist()
        )


        confidences = (

            result.boxes
            .conf
            .cpu()
            .numpy()
        )


        if result.boxes.id is not None:

            track_ids = (

                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

        else:

            track_ids = [

                -1
                for _ in boxes
            ]


        # ====================================================
        # EACH DETECTION
        # ====================================================

        for (

            box,
            cls_id,
            conf,
            track_id

        ) in zip(

            boxes,

            classes,

            confidences,

            track_ids
        ):


            x1, y1, x2, y2 = map(
                int,
                box
            )


            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                WIDTH - 1,
                x2
            )

            y2 = min(
                HEIGHT - 1,
                y2
            )


            # =================================================
            # PLAYER
            # =================================================

            if cls_id == 0:

                player_w = (
                    x2 - x1
                )

                player_h = (
                    y2 - y1
                )


                if (

                    player_w < 20
                    or
                    player_h < 40

                ):

                    continue


                detected_class = classify_player(

                    frame,

                    x1,
                    y1,
                    x2,
                    y2
                )


                stable_class = get_stable_class(

                    track_id,

                    detected_class
                )


                display_number = get_display_number(

                    track_id
                )


                # ------------------------------------------------
                # Team counts
                # ------------------------------------------------

                if stable_class == "IRAN":

                    iran_count += 1

                elif stable_class == "UZBEKISTAN":

                    uzbekistan_count += 1

                elif stable_class == "GOALKEEPER":

                    goalkeeper_count += 1

                elif stable_class == "REFEREE":

                    referee_count += 1


                # ------------------------------------------------
                # Foot position
                # ------------------------------------------------

                foot_x = (
                    x1 +
                    x2
                ) / 2


                foot_y = float(
                    y2
                )


                position = (

                    foot_x,
                    foot_y
                )


                # ------------------------------------------------
                # Motion
                # ------------------------------------------------

                motion = update_player_motion(

                    track_id,

                    stable_class,

                    position,

                    frame_number,

                    camera_matrix,

                    camera_reliable
                    and
                    not is_camera_cut
                )


                # ------------------------------------------------
                # Visual
                # ------------------------------------------------

                color = COLORS.get(

                    stable_class,

                    COLORS[
                        "UNKNOWN"
                    ]
                )


                draw_player_trajectory(

                    frame,

                    track_id,

                    color
                )


                draw_player_circle(

                    frame,

                    x1,
                    y1,
                    x2,
                    y2,

                    color
                )


                draw_id_badge(

                    frame,

                    x1,
                    y1,
                    x2,

                    stable_class,

                    display_number
                )


                # ------------------------------------------------
                # Motion text
                # ------------------------------------------------

                if motion["valid"]:

                    motion_text = (

                        f"{motion['speed']:.0f}"
                        f" px/s "
                        f"{motion['direction']}"
                    )

                else:

                    motion_text = (
                        motion[
                            "direction"
                        ]
                    )


                info_x = int(

                    (
                        x1 +
                        x2
                    ) / 2
                )


                info_y = min(

                    HEIGHT - 5,

                    y2 + 22
                )


                (
                    tw,
                    th
                ), _ = cv2.getTextSize(

                    motion_text,

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.35,

                    1
                )


                cv2.rectangle(

                    frame,

                    (
                        info_x -
                        tw // 2 -
                        4,

                        info_y -
                        th -
                        4
                    ),

                    (
                        info_x +
                        tw // 2 +
                        4,

                        info_y +
                        4
                    ),

                    (
                        10,
                        10,
                        10
                    ),

                    -1
                )


                cv2.putText(

                    frame,

                    motion_text,

                    (
                        info_x -
                        tw // 2,

                        info_y
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.35,

                    COLORS[
                        "WHITE"
                    ],

                    1,

                    cv2.LINE_AA
                )


            # =================================================
            # BALL
            # =================================================

            elif cls_id == 32:

                if conf >= BALL_CONF:

                    ball_x = int(

                        (
                            x1 +
                            x2
                        ) / 2
                    )


                    ball_y = int(

                        (
                            y1 +
                            y2
                        ) / 2
                    )


                    ball_candidates.append(

                        (
                            ball_x,
                            ball_y,
                            float(conf)
                        )
                    )


    # ========================================================
    # BALL TRACKING
    # ========================================================

    selected = select_best_ball(

        ball_candidates,

        BALL_POSITION
    )


    if selected is not None:

        new_position = np.array(

            [
                float(
                    selected[0]
                ),

                float(
                    selected[1]
                )
            ],

            dtype=np.float32
        )


        if BALL_POSITION is None:

            BALL_POSITION = (
                new_position
            )

            BALL_VELOCITY = np.array(

                [
                    0.0,
                    0.0
                ],

                dtype=np.float32
            )

            BALL_MISSING = 0


        else:

            distance = np.linalg.norm(

                new_position -
                BALL_POSITION
            )


            if distance > MAX_BALL_JUMP:

                BALL_POSITION = (
                    new_position
                )

                BALL_VELOCITY = np.array(

                    [
                        0.0,
                        0.0
                    ],

                    dtype=np.float32
                )

                BALL_MISSING = 0


            else:

                previous_ball = (
                    BALL_POSITION.copy()
                )


                BALL_POSITION = (

                    BALL_SMOOTHING *
                    new_position

                    +

                    (
                        1 -
                        BALL_SMOOTHING
                    )
                    *
                    BALL_POSITION
                )


                new_velocity = (

                    BALL_POSITION -
                    previous_ball
                )


                BALL_VELOCITY = (

                    0.7 *
                    BALL_VELOCITY

                    +

                    0.3 *
                    new_velocity
                )


                BALL_MISSING = 0


    else:

        BALL_MISSING += 1


        if (
            BALL_MISSING
            >
            BALL_MAX_MISSING
        ):

            BALL_POSITION = None

            BALL_VELOCITY = np.array(

                [
                    0.0,
                    0.0
                ],

                dtype=np.float32
            )


    # ========================================================
    # DRAW BALL
    # ========================================================

    if BALL_POSITION is not None:

        draw_meteor_ball(

            frame,

            BALL_POSITION,

            BALL_VELOCITY
        )


    # ========================================================
    # DASHBOARD
    # ========================================================

    cv2.rectangle(

        frame,

        (
            10,
            10
        ),

        (
            370,
            190
        ),

        (
            12,
            12,
            12
        ),

        -1
    )


    cv2.putText(

        frame,

        "AI FOOTBALL ANALYTICS V2",

        (
            22,
            38
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.62,

        COLORS[
            "WHITE"
        ],

        2,

        cv2.LINE_AA
    )


    cv2.line(

        frame,

        (
            20,
            50
        ),

        (
            350,
            50
        ),

        (
            70,
            70,
            70
        ),

        1
    )


    dashboard = [

        (
            "IRAN",
            iran_count,
            COLORS[
                "IRAN"
            ]
        ),

        (
            "UZBEKISTAN",
            uzbekistan_count,
            COLORS[
                "UZBEKISTAN"
            ]
        ),

        (
            "GOALKEEPER",
            goalkeeper_count,
            COLORS[
                "GOALKEEPER"
            ]
        ),

        (
            "REFEREE",
            referee_count,
            COLORS[
                "REFEREE"
            ]
        )
    ]


    y = 78


    for (

        name,
        count,
        color

    ) in dashboard:


        cv2.circle(

            frame,

            (
                28,
                y - 5
            ),

            7,

            color,

            -1,

            cv2.LINE_AA
        )


        cv2.putText(

            frame,

            name,

            (
                45,
                y
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            color,

            1,

            cv2.LINE_AA
        )


        cv2.putText(

            frame,

            str(count),

            (
                335,
                y
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            COLORS[
                "WHITE"
            ],

            2,

            cv2.LINE_AA
        )


        y += 24


    cv2.putText(

        frame,

        "MOTION: CAMERA COMPENSATED",

        (
            22,
            172
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.35,

        COLORS[
            "GREEN"
        ],

        1,

        cv2.LINE_AA
    )


    # ========================================================
    # CAMERA STATUS
    # ========================================================

    draw_camera_status(
        frame
    )


    # ========================================================
    # TIME
    # ========================================================

    draw_time_badge(

        frame,

        frame_number
    )


    # ========================================================
    # WRITE
    # ========================================================

    out.write(
        frame
    )


    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_number % 100 == 0:

        progress = (

            frame_number /
            TOTAL_FRAMES
        ) * 100


        print(

            f"Processed: "
            f"{frame_number}/"
            f"{TOTAL_FRAMES} "
            f"("
            f"{progress:.1f}%"
            ") | "
            f"Camera: "
            f"{CAMERA_STATUS}"
        )


# ============================================================
# 32. RELEASE
# ============================================================

out.release()


# ============================================================
# 33. PLAYER REPORT
# ============================================================

rows = []


for (

    track_id,
    stats

) in PLAYER_STATS.items():


    if stats[
        "frames"
    ] < 5:

        continue


    if stats[
        "speed_samples"
    ] > 0:

        avg_speed = (

            stats[
                "speed_sum_px_s"
            ]

            /

            stats[
                "speed_samples"
            ]
        )

    else:

        avg_speed = 0.0


    if stats[
        "acceleration_samples"
    ] > 0:

        avg_acceleration = (

            stats[
                "acceleration_sum"
            ]

            /

            stats[
                "acceleration_samples"
            ]
        )

    else:

        avg_acceleration = 0.0


    display_id = (

        PLAYER_DISPLAY_NUMBER.get(

            track_id,

            track_id
        )
    )


    directions = stats[
        "directions"
    ]


    if len(directions) > 0:

        dominant_direction = max(

            directions,

            key=directions.get
        )

    else:

        dominant_direction = (
            "UNKNOWN"
        )


    rows.append({

        "Track_ID":
            track_id,

        "Player_ID":
            display_id,

        "Team":
            stats[
                "team"
            ],

        "Frames_Tracked":
            stats[
                "frames"
            ],

        "Distance_px":
            round(

                stats[
                    "distance_px"
                ],

                1
            ),

        "Average_Relative_Speed_px_s":
            round(
                avg_speed,
                1
            ),

        "Max_Relative_Speed_px_s":
            round(

                stats[
                    "max_speed_px_s"
                ],

                1
            ),

        "Average_Acceleration_px_s2":
            round(
                avg_acceleration,
                2
            ),

        "Max_Acceleration_px_s2":
            round(

                stats[
                    "max_acceleration"
                ],

                2
            ),

        "Dominant_Direction":
            dominant_direction
    })


stats_df = pd.DataFrame(
    rows
)


if not stats_df.empty:

    stats_df = stats_df.sort_values(

        "Distance_px",

        ascending=False
    )


stats_df.to_csv(

    CSV_PATH,

    index=False
)


# ============================================================
# 34. RESULTS
# ============================================================

print("\n" + "=" * 70)

print(
    "🎉 AI FOOTBALL ANALYTICS V2.1 FINISHED"
)

print("=" * 70)

print(
    "\n🎥 VIDEO:"
)

print(
    OUTPUT_PATH
)

print(
    "\n📊 CSV:"
)

print(
    CSV_PATH
)


if not stats_df.empty:

    print(
        "\n📊 PLAYER PERFORMANCE"
    )

    display(
        stats_df
    )


# ============================================================
# 35. DOWNLOAD
# ============================================================

print(
    "\n⬇️ Downloading results..."
)

files.download(
    OUTPUT_PATH
)

files.download(
    CSV_PATH
)

print(
    "\n✅ ALL DONE!"
)
