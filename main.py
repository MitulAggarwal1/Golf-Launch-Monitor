import os
import math
import cv2
import numpy as np
import argparse
from ultralytics import YOLO

# =========================
# DEFAULTS (base settings)
# =========================
# Measurement and detection parameters
SCALE_M_PER_PX = 0.003893   # Conversion: pixel length to metres, set by calibration
G = 9.81                    # Acceleration due to gravity (metres per second squared)
DEFAULT_IMG_SIZE = 960      # Default image size used for inference
DEFAULT_MIN_WIDTH = 960     # Minimum allowed width for resize before model prediction
CONF_THRES = 0.20           # YOLO detection confidence threshold (raise to reduce false positives)
IOU_THRES = 0.60            # Intersection-over-union threshold for non-max suppression
VID_STRIDE = 1              # Number of frames to skip between detections (1=every frame)
SHOW_PREVIEW_600 = True     # Whether to display a live 600x600 preview window
WRITE_ANNOTATED_MP4 = True  # Whether to save an output annotated video
OUT_PATH = r".\ball_annotated.mp4"  # Path to output annotated video

# Model paths: trained detector (YOLO, prefers OpenVINO format if available)
OPENVINO_DIR = r"C:\Users\mitul\runs\detect\train2\weights\best_openvino_model"
BEST_PT_PATH = r"C:\Users\mitul\runs\detect\train2\weights\best.pt"

# =========================
# LOADERS AND HELPERS
# =========================

def load_detector():
    """
    Loads the YOLO detector, using OpenVINO if available for speed,
    otherwise PyTorch weights. Raises error if not found.
    """
    if os.path.isdir(OPENVINO_DIR):
        print(f"[MODEL] Loading OpenVINO: {OPENVINO_DIR}")
        return YOLO(OPENVINO_DIR)
    if os.path.isfile(BEST_PT_PATH):
        print(f"[MODEL] Loading PyTorch: {BEST_PT_PATH}")
        return YOLO(BEST_PT_PATH)
    raise FileNotFoundError("No model found. Check OPENVINO_DIR or BEST_PT_PATH.")

def dump_metadata(cap):
    """
    Prints and returns the video's FPS, frame count, and calculated duration.
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_video = frame_count / fps if fps > 0 else float("nan")
    print(f"[META] FPS: {fps:.3f}")
    print(f"[META] Frames: {frame_count}")
    print(f"[META] Duration: {duration_video:.3f} s")
    return fps, frame_count, duration_video

def find_launch_idx(points, thresh_px=5.0):
    """
    Find index where the tracked ball moves more than 'thresh_px' pixels from
    the initial point, indicating launch (returns index).
    """
    if len(points) < 2:
        return 0
    x0, y0 = points[0]
    for i in range(1, len(points)):
        dx = points[i][0] - x0
        dy = points[i][1] - y0
        if (dx*dx + dy*dy) ** 0.5 > thresh_px:
            return i
    return 0

def pick_best_ball(result, roi=None, max_box=200):
    """
    From a YOLO result, choose the default highest-confidence small box ideally
    corresponding to the ball. If region of interest is specified, only select
    a box whose centre falls inside it.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best = None
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    if roi is not None:
        rx, ry, rw, rh = roi
    else:
        rx = ry = rw = rh = None

    for (x1, y1, x2, y2), c in zip(boxes, confs):
        w = x2 - x1
        h = y2 - y1
        if w > max_box or h > max_box:
            continue
        if rx is not None:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if not (rx < cx < rx + rw and ry < cy < ry + rh):
                continue
        if best is None or c > best[0]:
            best = (c, x1, y1, x2, y2)
    return best

def estimate_slowmo_factor_from_gravity(points, fps, scale_m_per_px, vid_stride=1, min_pts=7):
    """
    (Advanced) Estimate slow motion playback factor so the observed vertical acceleration matches real gravity;
    useful when user timing data is missing.
    """
    if len(points) < min_pts:
        return None
    ys = np.array([p[1] for p in points], dtype=float)
    ddy = ys[2:] - 2 * ys[1:-1] + ys[:-2]
    if len(ddy) < 3:
        return None
    ay_pix = float(np.median(np.abs(ddy)))
    if ay_pix < 1e-6:
        return None
    dt_video_step = (vid_stride / max(fps, 1e-6))
    s = math.sqrt((ay_pix * scale_m_per_px) / G) / dt_video_step
    return float(np.clip(s, 0.03, 1.5))

def parse_args():
    """
    Parse command-line arguments for video input, swing timings, and model settings.
    """
    p = argparse.ArgumentParser(description="Golf ball launch monitor (YOLO + time analysis).")
    p.add_argument("--video", type=str, default=None, help="Path to input video file")
    p.add_argument("--swing-real", type=float, default=None, help="Actual swing time in seconds")
    p.add_argument("--swing-video", type=float, default=None, help="Swing time length in video seconds")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE, help="YOLO inference size")
    p.add_argument("--minw", type=int, default=DEFAULT_MIN_WIDTH, help="Minimum frame width for detection")
    p.add_argument("--stride", type=int, default=VID_STRIDE, help="Frame skip interval for processing")
    return p.parse_args()

def prompt_if_needed(args):
    """
    Prompt user interactively to fill in any missing required arguments.
    """
    if not args.video:
        args.video = input("Enter video path: ").strip().strip('"')
    if args.swing_real is None:
        s = input("Enter REAL swing duration (s) or press Enter to skip: ").strip()
        args.swing_real = float(s) if s else None
    if args.swing_video is None:
        s = input("Enter VIDEO swing duration (s) or press Enter to skip: ").strip()
        args.swing_video = float(s) if s else None
    return args

# =========================
# MAIN EXECUTION
# =========================
def main():
    # Parse and, if necessary, prompt for user input settings
    args = parse_args()
    args = prompt_if_needed(args)

    # Check video file availability
    VIDEO_PATH = args.video
    if not VIDEO_PATH or not os.path.isfile(VIDEO_PATH):
        print(f"[ERR ] Video not found: {VIDEO_PATH}")
        return

    # Set model/image size and stride from arguments
    IMG_SIZE  = int(args.imgsz)
    MIN_WIDTH = int(args.minw)
    VID_STRIDE = max(1, int(args.stride))

    # Load YOLO detection model (OpenVINO preferred for speed)
    detector = load_detector()

    # Open video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERR ] Could not open video: {VIDEO_PATH}")
        return

    # Display and log video metadata
    fps, frame_count, duration_video = dump_metadata(cap)

    # Calculate swing slow factor if both real and video durations are available
    KNOWN_SLOW_FACTOR = None
    if args.swing_real is not None and args.swing_video is not None and args.swing_video > 0:
        KNOWN_SLOW_FACTOR = float(args.swing_real) / float(args.swing_video)
        print(f"[TIME] Known slow factor = {KNOWN_SLOW_FACTOR:.6f}")

    # Have user select the analysis region of interest (ROI) on the first frame
    ret, frame0 = cap.read()
    if not ret:
        print("[ERR ] Can't read first frame.")
        cap.release()
        return

    preview0 = cv2.resize(frame0, (600, 600))
    print("[INFO] Select ROI (ENTER to confirm, c to cancel)")
    roi_disp = cv2.selectROI("ROI selector", preview0, False, False)
    cv2.destroyWindow("ROI selector")

    # Map ROI from 600x600 preview space back to the original frame size
    sx0 = frame0.shape[1] / 600.0
    sy0 = frame0.shape[0] / 600.0
    roi = (int(roi_disp[0] * sx0), int(roi_disp[1] * sy0),
           int(roi_disp[2] * sx0), int(roi_disp[3] * sy0))

    # Prepare annotated output video recorder if required
    writer = None
    if WRITE_ANNOTATED_MP4:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = (fps / max(VID_STRIDE, 1))
        writer = cv2.VideoWriter(OUT_PATH, fourcc, out_fps,
                                 (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        print(f"[OUT ] Writing annotated video to {os.path.abspath(OUT_PATH)}")

    # Reset video stream to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    centers = []  # Record detected ball centres
    frame_idx = 0

    print("[INFO] Press 'q' to stop early.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Use stride to optionally skip frames for faster processing
        if (frame_idx - 1) % max(VID_STRIDE, 1) != 0:
            continue

        # If input width is below threshold, scale up working frame
        h, w = frame.shape[:2]
        if w < MIN_WIDTH:
            scale_h = int(h * (MIN_WIDTH / w))
            proc = cv2.resize(frame, (MIN_WIDTH, scale_h))
        else:
            proc = frame

        # Run YOLO inference on this processed frame
        res = detector.predict(proc, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]

        # Calculate scaling factors to map from processed frame back to original
        sx = frame.shape[1] / proc.shape[1]
        sy = frame.shape[0] / proc.shape[0]

        # Pick the most likely ball detection in this frame
        best = pick_best_ball(res, roi=None, max_box=200)

        if best is not None:
            _, x1, y1, x2, y2 = best
            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            centers.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        # Draw ball path as a line connecting all tracked centre points
        for i in range(1, len(centers)):
            cv2.line(frame, centers[i - 1], centers[i], (0, 0, 255), 2)

        # Draw information bar above frame (scale used and YOLO config)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 28), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        cv2.putText(frame, f"YOLO({IMG_SIZE}) | scale={SCALE_M_PER_PX:.6f} m/px",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Write frame to output video if required, and show preview
        if writer is not None:
            writer.write(frame)
        if SHOW_PREVIEW_600:
            disp = cv2.resize(frame, (600, 600))
            cv2.imshow("Launch Monitor (AI)", disp)
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                break

    # Release video and output handles once finished
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # --- Post-detection analysis of trajectory (AI physics section) ---

    if len(centers) < 2:
        print("[RESULT] Not enough detections for analysis.")
        return

    # Finds frame where the ball launches (first significant movement)
    start_idx = find_launch_idx(centers, thresh_px=5.0)
    traj = centers[start_idx:] if start_idx < len(centers) else centers

    # Automatic or manual estimation of swing slow-motion scaling
    if KNOWN_SLOW_FACTOR is not None:
        slow_factor = float(KNOWN_SLOW_FACTOR)
        print(f"[TIME] Using known slow factor: {slow_factor:.6f}")
    else:
        s_est = estimate_slowmo_factor_from_gravity(traj, fps, SCALE_M_PER_PX,
                                                    vid_stride=VID_STRIDE, min_pts=7)
        if s_est is not None:
            slow_factor = s_est
            print(f"[TIME] Estimated slow factor: {slow_factor:.6f}")
        else:
            slow_factor = 1.0
            print("[TIME] Could not estimate slow factor, defaulting to 1.0")

    slow_factor = float(np.clip(slow_factor, 0.03, 1.2))

    if len(traj) < 3:
        print("[RESULT] Too few trajectory points for velocity fitting.")
        return

    # Use least squares regression (over up to 20 points) for velocity/angle estimation
    dt_video_step = VID_STRIDE / max(fps, 1e-6)
    dt_real_step  = dt_video_step * slow_factor

    K = min(20, len(traj))
    if K < 2:
        print("[RESULT] Not enough points for regression.")
        return

    t_real = np.arange(K, dtype=float) * dt_real_step
    xs = np.array([p[0] for p in traj[:K]], dtype=float)
    ys = np.array([p[1] for p in traj[:K]], dtype=float)

    A = np.c_[t_real, np.ones_like(t_real)]
    ax, bx = np.linalg.lstsq(A, xs, rcond=None)[0]
    ay, by = np.linalg.lstsq(A, ys, rcond=None)[0]

    # Convert motion rates from pixels per second to metres per second, flipping vertical direction
    vx = ax * SCALE_M_PER_PX
    vy = -ay * SCALE_M_PER_PX
    v0 = math.hypot(vx, vy)
    angle_rad = math.atan2(vy, vx)
    angle_deg = math.degrees(angle_rad)

    if v0 > 120:
        print("[WARN] Unrealistically high speed - check scale, ROI, or model.")

    # Projectile predictions (idealised flight, ignoring air drag)
    T_ideal = 2 * v0 * math.sin(angle_rad) / G if v0 > 0 else 0.0
    H_ideal = (v0**2) * (math.sin(angle_rad)**2) / (2 * G)
    D_ideal = (v0**2) * math.sin(2 * angle_rad) / G

    (x0, y0) = traj[0]
    (x1, y1) = traj[min(len(traj) - 1, K - 1)]
    dx_pix = float(x1 - x0)
    dy_pix = float(y1 - y0)
    dist_pix = float(np.hypot(dx_pix, dy_pix))
    dist_m = dist_pix * SCALE_M_PER_PX

    frames_spanned = (min(len(traj) - 1, K - 1)) * max(VID_STRIDE, 1)
    dt_video = frames_spanned / max(fps, 1e-6)
    dt_real  = dt_video * slow_factor

    # Print a concise summary of physics and detection results
    print("\n--- SUMMARY (AI-based launch metrics) ---")
    print(f"Video: {VIDEO_PATH}")
    print(f"Used {K} trajectory points starting at index {start_idx}")
    print(f"Slow factor (real/video): {slow_factor:.6f}")
    print(f"Δx = {dx_pix:.2f} px, Δy = {dy_pix:.2f} px, Distance = {dist_m:.3f} m")
    print(f"dt_video = {dt_video:.4f} s, dt_real = {dt_real:.4f} s")
    print(f"\nLaunch Angle: {angle_deg:.2f}°")
    print(f"Initial Velocity: {v0:.2f} m/s ({v0*3.6:.2f} km/h, {v0*2.237:.2f} mph)")
    print(f"Flight Time: {T_ideal:.2f} s")
    print(f"Apex Height: {H_ideal:.2f} m")
    print(f"Horizontal Distance: {D_ideal:.2f} m")

if __name__ == "__main__":
    main()
