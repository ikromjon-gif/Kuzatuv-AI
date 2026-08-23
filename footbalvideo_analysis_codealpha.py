# ============================================================
# ⚽ AI FOOTBALL ANALYTICS V2
# YOLO11 + ByteTrack + HSV TEAM CLASSIFICATION
# + MOUSE PITCH POINT SELECTION
# + HOMOGRAPHY
# + HEATMAP
# + SPEED
# + DISTANCE
# ============================================================

# ============================================================
# 1. INSTALL
# ============================================================

!pip install -q ultralytics opencv-python-headless lap pandas matplotlib


# ============================================================
# 2. IMPORTS
# ============================================================

import cv2
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from google.colab import files
from IPython.display import display, clear_output
from PIL import Image
from ultralytics import YOLO


# ============================================================
# 3. CONFIG
# ============================================================

MODEL_NAME = "yolo11x.pt"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

PERSON_CONF = 0.25
BALL_CONF = 0.10

# Ball
MAX_BALL_JUMP = 180
BALL_SMOOTHING = 0.65
BALL_MAX_MISSING = 4

# Standard football pitch assumption
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Maximum physically reasonable player speed
MAX_REASONABLE_SPEED = 40.0


print("=" * 70)
print("⚽ AI FOOTBALL ANALYTICS V2")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 4. UPLOAD VIDEO
# ============================================================

print("\n🎥 Upload your football video")

uploaded = files.upload()

VIDEO_PATH = list(uploaded.keys())[0]

print("VIDEO:", VIDEO_PATH)


# ============================================================
# 5. VIDEO INFO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

FPS = cap.get(cv2.CAP_PROP_FPS)

WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()

if FPS <= 0:
    FPS = 30.0

DURATION = TOTAL_FRAMES / FPS

print("\nVIDEO INFO")
print("FPS:", FPS)
print("Resolution:", WIDTH, "x", HEIGHT)
print("Frames:", TOTAL_FRAMES)
print("Duration:", round(DURATION, 2), "seconds")


# ============================================================
# 6. LOAD YOLO
# ============================================================

print("\n🚀 Loading YOLO11...")

model = YOLO(MODEL_NAME)

print("YOLO READY!")


# ============================================================
# 7. LOAD FIRST FRAME
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

ret, first_frame = cap.read()

cap.release()

if not ret:
    raise RuntimeError("Could not read first frame.")


# ============================================================
# 8. INTERACTIVE PITCH POINT SELECTION
# ============================================================

print("\n" + "=" * 70)
print("📐 PITCH MAPPING")
print("=" * 70)

print("""
You will select 4 points using your mouse.

ORDER:

1 → TOP-LEFT
2 → TOP-RIGHT
3 → BOTTOM-RIGHT
4 → BOTTOM-LEFT

Click four points on the football pitch.

After the 4th point, press ENTER.
Press R to reset and select again.
""")

selected_points = []


# ------------------------------------------------------------
# Resize only for display
# Original coordinates are automatically restored
# ------------------------------------------------------------

MAX_DISPLAY_WIDTH = 1200

display_scale = 1.0

if WIDTH > MAX_DISPLAY_WIDTH:

    display_scale = (
        MAX_DISPLAY_WIDTH /
        WIDTH
    )

display_width = int(
    WIDTH * display_scale
)

display_height = int(
    HEIGHT * display_scale
)


display_frame = cv2.resize(
    first_frame,
    (
        display_width,
        display_height
    )
)


window_name = "PITCH MAPPING - CLICK 4 POINTS"


def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global selected_points

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(selected_points) >= 4:
            return

        # Convert display coordinates
        # back to original video coordinates

        original_x = int(
            x / display_scale
        )

        original_y = int(
            y / display_scale
        )

        selected_points.append(
            (
                original_x,
                original_y
            )
        )

        print(
            f"Point {len(selected_points)}: "
            f"({original_x}, {original_y})"
        )


cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    display_width,
    display_height
)

cv2.setMouseCallback(
    window_name,
    mouse_callback
)


while True:

    canvas = display_frame.copy()


    # --------------------------------------------------------
    # Draw selected points
    # --------------------------------------------------------

    labels = [
        "TOP-LEFT",
        "TOP-RIGHT",
        "BOTTOM-RIGHT",
        "BOTTOM-LEFT"
    ]


    for i, point in enumerate(
        selected_points
    ):

        original_x, original_y = point

        x = int(
            original_x *
            display_scale
        )

        y = int(
            original_y *
            display_scale
        )


        cv2.circle(
            canvas,
            (x, y),
            8,
            (0, 255, 255),
            -1
        )


        cv2.putText(

            canvas,

            f"{i+1} {labels[i]}",

            (
                x + 10,
                y - 10
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 255, 255),

            2,

            cv2.LINE_AA
        )


    # --------------------------------------------------------
    # Draw lines between selected points
    # --------------------------------------------------------

    if len(selected_points) >= 2:

        for i in range(
            len(selected_points) - 1
        ):

            p1 = selected_points[i]
            p2 = selected_points[i + 1]


            p1_display = (

                int(
                    p1[0] *
                    display_scale
                ),

                int(
                    p1[1] *
                    display_scale
                )
            )


            p2_display = (

                int(
                    p2[0] *
                    display_scale
                ),

                int(
                    p2[1] *
                    display_scale
                )
            )


            cv2.line(

                canvas,

                p1_display,

                p2_display,

                (0, 255, 255),

                2,

                cv2.LINE_AA
            )


    # Close polygon
    if len(selected_points) == 4:

        p1 = selected_points[3]
        p2 = selected_points[0]

        cv2.line(

            canvas,

            (
                int(
                    p1[0] *
                    display_scale
                ),

                int(
                    p1[1] *
                    display_scale
                )
            ),

            (
                int(
                    p2[0] *
                    display_scale
                ),

                int(
                    p2[1] *
                    display_scale
                )
            ),

            (0, 255, 255),

            2,

            cv2.LINE_AA
        )


    # --------------------------------------------------------
    # Instructions
    # --------------------------------------------------------

    instruction = (
        f"Selected: "
        f"{len(selected_points)}/4 | "
        f"ENTER = confirm | "
        f"R = reset"
    )


    cv2.rectangle(

        canvas,

        (10, 10),

        (650, 50),

        (15, 15, 15),

        -1
    )


    cv2.putText(

        canvas,

        instruction,

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2,

        cv2.LINE_AA
    )


    cv2.imshow(
        window_name,
        canvas
    )


    key = cv2.waitKey(50) & 0xFF


    # ENTER
    if key in [13, 10]:

        if len(selected_points) == 4:
            break

        else:

            print(
                "❌ Please select exactly 4 points."
            )


    # R = RESET
    elif key in [
        ord("r"),
        ord("R")
    ]:

        selected_points = []

        print(
            "🔄 Points reset."
        )


    # ESC
    elif key == 27:

        cv2.destroyAllWindows()

        raise RuntimeError(
            "Pitch point selection cancelled."
        )


cv2.destroyAllWindows()


# ============================================================
# 9. CREATE HOMOGRAPHY
# ============================================================

src_points = np.float32(
    selected_points
)


# Real football pitch coordinates
#
# 1 → TOP-LEFT
# 2 → TOP-RIGHT
# 3 → BOTTOM-RIGHT
# 4 → BOTTOM-LEFT

dst_points = np.float32([

    [0, 0],

    [PITCH_LENGTH, 0],

    [PITCH_LENGTH, PITCH_WIDTH],

    [0, PITCH_WIDTH]
])


H = cv2.getPerspectiveTransform(

    src_points,

    dst_points
)


print("\n✅ Homography created.")

print(
    "Selected points:"
)

for i, p in enumerate(
    selected_points
):

    print(
        f"{i+1}: {p}"
    )


# ============================================================
# 10. COLORS
# ============================================================

COLORS = {

    "IRAN": (245, 245, 245),

    "UZBEKISTAN": (255, 80, 0),

    "GOALKEEPER": (120, 255, 120),

    "REFEREE": (40, 40, 255),

    "BALL": (0, 220, 255),

    "UNKNOWN": (0, 220, 255),

    "DARK": (18, 18, 18),

    "WHITE": (255, 255, 255)
}


# ============================================================
# 11. PLAYER MEMORY
# ============================================================

PLAYER_HISTORY = {}

PLAYER_STABLE_CLASS = {}

PLAYER_DISPLAY_NUMBER = {}

NEXT_PLAYER_NUMBER = 1


# ============================================================
# 12. ANALYTICS MEMORY
# ============================================================

PLAYER_STATS = defaultdict(
    lambda: {

        "team": "UNKNOWN",

        "frames": 0,

        "distance_m": 0.0,

        "max_speed_kmh": 0.0,

        "speed_sum": 0.0,

        "speed_samples": 0,

        "positions": [],

        "last_pitch_position": None,

        "last_frame": None
    }
)


# ============================================================
# 13. JERSEY REGION
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
# 14. TEAM CLASSIFICATION
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
        crop.shape[0] *
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


    best_score = scores[
        best_class
    ]


    if best_score < 0.12:
        return "UNKNOWN"


    return best_class


# ============================================================
# 15. STABLE CLASS
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


    best_count = PLAYER_HISTORY[
        track_id
    ][best_class]


    if best_count < 2:

        stable = detected_class

    else:

        stable = best_class


    PLAYER_STABLE_CLASS[
        track_id
    ] = stable


    return stable


# ============================================================
# 16. DISPLAY NUMBER
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
# 17. PIXEL → REAL PITCH
# ============================================================

def pixel_to_pitch(
    x,
    y
):

    point = np.array(

        [[[x, y]]],

        dtype=np.float32
    )


    transformed = cv2.perspectiveTransform(

        point,

        H
    )


    pitch_x = float(
        transformed[0][0][0]
    )

    pitch_y = float(
        transformed[0][0][1]
    )


    return (
        pitch_x,
        pitch_y
    )


# ============================================================
# 18. FOOT POINT
# ============================================================

def get_foot_point(
    x1,
    y1,
    x2,
    y2
):

    return (

        (x1 + x2) / 2,

        float(y2)
    )


# ============================================================
# 19. UPDATE ANALYTICS
# ============================================================

def update_player_analytics(

    track_id,

    team,

    foot_x,

    foot_y,

    frame_number

):

    if track_id == -1:
        return


    stats = PLAYER_STATS[
        track_id
    ]


    stats[
        "team"
    ] = team


    stats[
        "frames"
    ] += 1


    pitch_x, pitch_y = pixel_to_pitch(

        foot_x,

        foot_y
    )


    current_position = np.array(

        [
            pitch_x,
            pitch_y
        ],

        dtype=np.float32
    )


    # Keep only reasonable pitch positions
    if not (

        -10 <= pitch_x <=
        PITCH_LENGTH + 10

        and

        -10 <= pitch_y <=
        PITCH_WIDTH + 10

    ):

        return


    stats[
        "positions"
    ].append(

        (
            pitch_x,
            pitch_y
        )
    )


    previous_position = stats[
        "last_pitch_position"
    ]


    previous_frame = stats[
        "last_frame"
    ]


    if (

        previous_position is not None

        and

        previous_frame is not None

    ):

        frame_gap = (

            frame_number -
            previous_frame
        )


        if (

            frame_gap > 0

            and

            frame_gap <= 3

        ):

            dt = (
                frame_gap /
                FPS
            )


            displacement = np.linalg.norm(

                current_position -
                previous_position
            )


            max_displacement = (

                MAX_REASONABLE_SPEED /
                3.6
            ) * dt


            if displacement <= (

                max_displacement * 1.5

            ):

                speed_ms = (
                    displacement /
                    dt
                )


                speed_kmh = (
                    speed_ms *
                    3.6
                )


                stats[
                    "distance_m"
                ] += displacement


                stats[
                    "speed_sum"
                ] += speed_kmh


                stats[
                    "speed_samples"
                ] += 1


                stats[
                    "max_speed_kmh"
                ] = max(

                    stats[
                        "max_speed_kmh"
                    ],

                    speed_kmh
                )


    stats[
        "last_pitch_position"
    ] = current_position


    stats[
        "last_frame"
    ] = frame_number


# ============================================================
# 20. DRAW PLAYER CIRCLE
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
        (x1 + x2) / 2
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
# 21. ID BADGE
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

        if display_number > 0:

            label += (
                f" {display_number}"
            )

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


    font_scale = 0.55
    thickness = 2


    (
        text_w,
        text_h
    ), _ = cv2.getTextSize(

        label,

        font,

        font_scale,

        thickness
    )


    padding_x = 10
    padding_y = 7


    badge_w = (
        text_w +
        padding_x * 2
    )


    badge_h = (
        text_h +
        padding_y * 2
    )


    center_x = int(
        (x1 + x2) / 2
    )


    badge_x1 = (
        center_x -
        badge_w // 2
    )


    badge_y2 = max(

        badge_h + 5,

        y1 - 8
    )


    badge_x2 = (
        badge_x1 +
        badge_w
    )


    badge_y1 = (
        badge_y2 -
        badge_h
    )


    cv2.rectangle(

        frame,

        (
            badge_x1,
            badge_y1
        ),

        (
            badge_x2,
            badge_y2
        ),

        (18, 18, 18),

        -1
    )


    cv2.rectangle(

        frame,

        (
            badge_x1,
            badge_y1
        ),

        (
            badge_x2,
            badge_y2
        ),

        color,

        2
    )


    cv2.putText(

        frame,

        label,

        (
            badge_x1 +
            padding_x,

            badge_y2 -
            padding_y
        ),

        font,

        font_scale,

        COLORS[
            "WHITE"
        ],

        thickness,

        cv2.LINE_AA
    )


# ============================================================
# 22. BALL TRACKER
# ============================================================

BALL_POSITION = None

BALL_VELOCITY = np.array(
    [0.0, 0.0]
)

BALL_MISSING = 0


def select_best_ball(

    ball_candidates,

    previous_position

):

    if len(
        ball_candidates
    ) == 0:

        return None


    if previous_position is None:

        return max(

            ball_candidates,

            key=lambda x: x[2]
        )


    best_candidate = None

    best_score = -999999


    for (

        x,
        y,
        conf

    ) in ball_candidates:


        distance = math.dist(

            previous_position,

            (x, y)
        )


        score = (

            conf * 100
            -
            distance * 0.08
        )


        if score > best_score:

            best_score = score

            best_candidate = (

                x,
                y,
                conf,
                distance
            )


    return best_candidate


# ============================================================
# 23. DRAW METEOR BALL
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

        dx = vx / speed
        dy = vy / speed


        tail_length = min(

            45,

            max(
                15,
                int(speed * 3)
            )
        )


        for i in range(
            6,
            0,
            -1
        ):

            ratio = i / 6.0


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
                    (1 - ratio)
                )
            )


            cv2.circle(

                frame,

                (
                    tx,
                    ty
                ),

                radius + 2,

                (0, 100, 255),

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

                (0, 220, 255),

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

        (0, 150, 255),

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

        (0, 230, 255),

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

        (255, 255, 255),

        -1,

        cv2.LINE_AA
    )


# ============================================================
# 24. TIME
# ============================================================

def format_time(
    seconds
):

    minutes = int(
        seconds // 60
    )

    seconds = int(
        seconds % 60
    )

    return (
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def draw_time_badge(

    frame,

    frame_number

):

    current_seconds = (

        frame_number /
        FPS
    )


    current_time = format_time(

        current_seconds
    )


    total_time = format_time(

        DURATION
    )


    text = (

        f"{current_time} / "
        f"{total_time}"
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

    y1 = HEIGHT - 42

    x2 = WIDTH - 10

    y2 = HEIGHT - 10


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

        (20, 20, 20),

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
# 25. OUTPUT
# ============================================================

OUTPUT_PATH = (
    "/content/"
    "football_analytics_V2.mp4"
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
# 26. START ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("🚀 STARTING FOOTBALL ANALYTICS V2")
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
# 27. PROCESS
# ============================================================

for result in results:

    frame_number += 1

    frame = result.orig_img.copy()


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
                # Counts
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
                # Foot point
                # ------------------------------------------------

                foot_x, foot_y = get_foot_point(

                    x1,
                    y1,
                    x2,
                    y2
                )


                # ------------------------------------------------
                # Analytics
                # ------------------------------------------------

                update_player_analytics(

                    track_id,

                    stable_class,

                    foot_x,

                    foot_y,

                    frame_number
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


            # =================================================
            # BALL
            # =================================================

            elif cls_id == 32:

                if conf >= BALL_CONF:

                    ball_x = int(

                        (
                            x1 + x2
                        ) / 2
                    )


                    ball_y = int(

                        (
                            y1 + y2
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
    # BALL SELECTION
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
            ]
        )


        if BALL_POSITION is None:

            BALL_POSITION = (
                new_position
            )

            BALL_VELOCITY = np.array(
                [0.0, 0.0]
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
                    [0.0, 0.0]
                )

                BALL_MISSING = 0


            else:

                previous = (
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
                    previous
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

            BALL_MISSING >
            BALL_MAX_MISSING

        ):

            BALL_POSITION = None

            BALL_VELOCITY = np.array(
                [0.0, 0.0]
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

        (10, 10),

        (350, 180),

        (12, 12, 12),

        -1
    )


    cv2.putText(

        frame,

        "AI FOOTBALL ANALYTICS V2",

        (22, 38),

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

        (20, 50),

        (340, 50),

        (70, 70, 70),

        1
    )


    dashboard_data = [

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

    ) in dashboard_data:


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
                315,
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

        "HEATMAP | SPEED | DISTANCE",

        (
            22,
            170
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.38,

        (200, 200, 200),

        1,

        cv2.LINE_AA
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
            f"({progress:.1f}%)"
        )


# ============================================================
# 28. RELEASE
# ============================================================

out.release()


# ============================================================
# 29. PLAYER STATISTICS
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
                "speed_sum"
            ]
            /
            stats[
                "speed_samples"
            ]
        )

    else:

        avg_speed = 0.0


    display_number = (
        PLAYER_DISPLAY_NUMBER.get(

            track_id,

            track_id
        )
    )


    rows.append({

        "Track_ID":
            track_id,

        "Player_ID":
            display_number,

        "Team":
            stats[
                "team"
            ],

        "Frames_Tracked":
            stats[
                "frames"
            ],

        "Distance_km":
            round(

                stats[
                    "distance_m"
                ] / 1000,

                3
            ),

        "Average_Speed_kmh":
            round(
                avg_speed,
                2
            ),

        "Max_Speed_kmh":
            round(

                stats[
                    "max_speed_kmh"
                ],

                2
            )
    })


stats_df = pd.DataFrame(
    rows
)


if not stats_df.empty:

    stats_df = stats_df.sort_values(

        "Distance_km",

        ascending=False
    )


STATS_PATH = (
    "/content/"
    "player_statistics_V2.csv"
)


stats_df.to_csv(

    STATS_PATH,

    index=False
)


# ============================================================
# 30. HEATMAP
# ============================================================

print(
    "\n🔥 Creating heatmap..."
)


all_x = []
all_y = []


for (

    track_id,
    stats

) in PLAYER_STATS.items():


    for (

        px,
        py

    ) in stats[
        "positions"
    ]:


        if (

            0 <= px <=
            PITCH_LENGTH

            and

            0 <= py <=
            PITCH_WIDTH

        ):

            all_x.append(
                px
            )

            all_y.append(
                py
            )


HEATMAP_PATH = (
    "/content/"
    "football_heatmap_V2.png"
)


plt.figure(
    figsize=(12, 7)
)


plt.xlim(
    0,
    PITCH_LENGTH
)

plt.ylim(
    0,
    PITCH_WIDTH
)


plt.gca().set_aspect(
    "equal"
)


# Pitch outline

plt.plot(
    [0, PITCH_LENGTH],
    [0, 0]
)

plt.plot(
    [0, PITCH_LENGTH],
    [
        PITCH_WIDTH,
        PITCH_WIDTH
    ]
)

plt.plot(
    [0, 0],
    [0, PITCH_WIDTH]
)

plt.plot(
    [PITCH_LENGTH, PITCH_LENGTH],
    [0, PITCH_WIDTH]
)


# Halfway line

plt.plot(

    [
        PITCH_LENGTH / 2,
        PITCH_LENGTH / 2
    ],

    [
        0,
        PITCH_WIDTH
    ]
)


# Center circle

theta = np.linspace(
    0,
    2 * np.pi,
    200
)


center_x = (

    PITCH_LENGTH / 2
    +
    9.15 *
    np.cos(theta)
)


center_y = (

    PITCH_WIDTH / 2
    +
    9.15 *
    np.sin(theta)
)


plt.plot(

    center_x,

    center_y
)


if len(all_x) > 10:

    plt.hexbin(

        all_x,

        all_y,

        gridsize=35,

        cmap="hot",

        mincnt=1
    )


    plt.colorbar(
        label="Player Presence"
    )


plt.title(
    "AI Football Analytics — Player Heatmap"
)


plt.xlabel(
    "Pitch Length (meters)"
)


plt.ylabel(
    "Pitch Width (meters)"
)


plt.tight_layout()


plt.savefig(

    HEATMAP_PATH,

    dpi=180
)


plt.show()


# ============================================================
# 31. FINAL
# ============================================================

print("\n" + "=" * 70)
print("🎉 AI FOOTBALL ANALYTICS V2 FINISHED")
print("=" * 70)

print(
    "\n🎥 VIDEO:",
    OUTPUT_PATH
)

print(
    "📊 CSV:",
    STATS_PATH
)

print(
    "🔥 HEATMAP:",
    HEATMAP_PATH
)


# ============================================================
# 32. DISPLAY STATISTICS
# ============================================================

if not stats_df.empty:

    print(
        "\n📊 PLAYER PERFORMANCE"
    )

    display(
        stats_df
    )

else:

    print(
        "No sufficient player tracking data found."
    )


# ============================================================
# 33. DOWNLOAD
# ============================================================

print(
    "\n⬇️ Downloading results..."
)


files.download(
    OUTPUT_PATH
)

files.download(
    STATS_PATH
)

files.download(
    HEATMAP_PATH
)


print(
    "\n✅ ALL DONE!"
)
