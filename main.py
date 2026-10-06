import argparse
import math
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
DISPLAY_W, DISPLAY_H = 1000, 700  # size of the main image window
MIN_ZOOM, MAX_ZOOM = 1.0, 6.0
SMOOTHING = 0.25  # 0..1, higher = snappier, lower = smoother
PINCH_THRESHOLD = 0.30  # thumb-index / palm size ratio for "touching"
L_SHAPE_RANGE = (0.35, 1.70)  # ratio range mapped to MIN_ZOOM..MAX_ZOOM
FIST_HOLD_SECONDS = 1.5

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def make_demo_image(w=1600, h=1000):
    """Generate a colorful test image so the program works without any file."""
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        img[y, :, 0] = int(255 * y / h)
        img[y, :, 2] = 255 - int(255 * y / h)
    for x in range(0, w, 100):
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)
    for y in range(0, h, 100):
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 1)
    rng = np.random.default_rng(7)
    for _ in range(40):
        c = tuple(int(v) for v in rng.integers(40, 255, 3))
        p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(img, p, int(rng.integers(20, 90)), c, -1)
    cv2.putText(
        img,
        "HAND CONTROL DEMO",
        (330, 520),
        cv2.FONT_HERSHEY_DUPLEX,
        3,
        (255, 255, 255),
        6,
        cv2.LINE_AA,
    )
    return img


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def palm_size(lm):
    """Wrist -> middle finger base. Used to make distances independent of
    how far the hand is from the camera."""
    return max(dist(lm[0], lm[9]), 1e-6)


def fingers_up(lm):
    """Returns (index, middle, ring, pinky) booleans. Works for an upright hand."""
    tips = (8, 12, 16, 20)
    pips = (6, 10, 14, 18)
    return tuple(lm[t].y < lm[p].y for t, p in zip(tips, pips))


def classify_single_hand(lm):
    index, middle, ring, pinky = fingers_up(lm)
    pinch_ratio = dist(lm[4], lm[8]) / palm_size(lm)

    if index and not (middle or ring or pinky):
        return "ZOOM", pinch_ratio
    if middle and ring and pinky and pinch_ratio < PINCH_THRESHOLD:
        return "PAN", pinch_ratio
    if not (index or middle or ring or pinky):
        return "FIST", pinch_ratio
    return "IDLE", pinch_ratio


def render_view(img, zoom, cx, cy):
    """Crop the image around (cx, cy) in normalized coords and scale it to the window."""
    h, w = img.shape[:2]
    view_w, view_h = w / zoom, h / zoom
    x0 = np.clip(cx * w - view_w / 2, 0, w - view_w)
    y0 = np.clip(cy * h - view_h / 2, 0, h - view_h)
    crop = img[int(y0) : int(y0 + view_h), int(x0) : int(x0 + view_w)]
    return cv2.resize(crop, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)


def draw_hud(canvas, mode, zoom):
    cv2.rectangle(canvas, (0, 0), (DISPLAY_W, 50), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        f"Mode: {mode}   Zoom: {zoom:.2f}x",
        (15, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    help_text = "2 hands: zoom | L-shape: zoom | OK sign: pan | fist: reset | q: quit"
    cv2.putText(
        canvas,
        help_text,
        (15, DISPLAY_H - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Control an image with your hand")
    parser.add_argument("image", nargs="?", help="path to an image (optional)")
    parser.add_argument("--cam", type=int, default=0, help="camera index")
    args = parser.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"Could not read image: {args.image}")
    else:
        img = make_demo_image()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        sys.exit("Could not open the webcam. Try --cam 1")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    # View state
    zoom = 1.0  # smoothed zoom actually shown
    target_zoom = 1.0  # zoom we are moving toward
    cx, cy = 0.5, 0.5  # view center in normalized image coords

    # Gesture state
    two_hand_start_dist = None
    two_hand_start_zoom = None
    pan_prev = None
    fist_since = None
    mode = "IDLE"

    hands = mp_hands.Hands(
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    cv2.namedWindow("Hand Control", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror so it feels natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            hand_list = result.multi_hand_landmarks or []
            new_mode = "IDLE"

            # ---------------- two hands: zoom ----------------
            if len(hand_list) == 2:
                new_mode = "TWO-HAND ZOOM"
                a = hand_list[0].landmark[8]
                b = hand_list[1].landmark[8]
                d = math.hypot(a.x - b.x, a.y - b.y)

                if two_hand_start_dist is None:
                    two_hand_start_dist = max(d, 1e-3)
                    two_hand_start_zoom = target_zoom
                target_zoom = two_hand_start_zoom * (d / two_hand_start_dist)
                pan_prev = None
                fist_since = None

            # ---------------- one hand ----------------
            elif len(hand_list) == 1:
                two_hand_start_dist = None
                lm = hand_list[0].landmark
                gesture, ratio = classify_single_hand(lm)

                if gesture == "ZOOM":
                    new_mode = "ONE-HAND ZOOM"
                    lo, hi = L_SHAPE_RANGE
                    t = np.clip((ratio - lo) / (hi - lo), 0, 1)
                    target_zoom = MIN_ZOOM + t * (MAX_ZOOM - MIN_ZOOM)
                    pan_prev = None
                    fist_since = None

                elif gesture == "PAN":
                    new_mode = "PAN"
                    # use midpoint of thumb and index as the "grab" point
                    px = (lm[4].x + lm[8].x) / 2
                    py = (lm[4].y + lm[8].y) / 2
                    if pan_prev is not None:
                        dx, dy = px - pan_prev[0], py - pan_prev[1]
                        # dragging right moves the image right -> center moves left
                        cx -= dx / max(zoom, 1e-6)
                        cy -= dy / max(zoom, 1e-6)
                    pan_prev = (px, py)
                    fist_since = None

                elif gesture == "FIST":
                    new_mode = "FIST (hold to reset)"
                    pan_prev = None
                    if fist_since is None:
                        fist_since = time.time()
                    elif time.time() - fist_since > FIST_HOLD_SECONDS:
                        target_zoom, cx, cy = 1.0, 0.5, 0.5
                        fist_since = None
                else:
                    pan_prev = None
                    fist_since = None
            else:
                two_hand_start_dist = None
                pan_prev = None
                fist_since = None

            mode = new_mode

            # ---------------- update view ----------------
            target_zoom = float(np.clip(target_zoom, MIN_ZOOM, MAX_ZOOM))
            zoom += (target_zoom - zoom) * SMOOTHING
            half = 0.5 / zoom
            cx = float(np.clip(cx, half, 1 - half))
            cy = float(np.clip(cy, half, 1 - half))

            canvas = render_view(img, zoom, cx, cy)
            draw_hud(canvas, mode, zoom)

            # ---------------- webcam preview (picture-in-picture) ----------------
            for hl in hand_list:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            pip = cv2.resize(frame, (280, 158))
            canvas[
                DISPLAY_H - 158 - 40 : DISPLAY_H - 40,
                DISPLAY_W - 280 - 10 : DISPLAY_W - 10,
            ] = pip

            cv2.imshow("Hand Control", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                target_zoom, cx, cy = 1.0, 0.5, 0.5
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
