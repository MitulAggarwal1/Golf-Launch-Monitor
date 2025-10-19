import cv2
import numpy as np
import math

# =========================
# USER SETTINGS
# =========================
VIDEO_PATH = r"C:\Users\mitul\OneDrive\GolfApp\Tiger golf swing 2.mp4"

# Scale in metres per pixel (calibrated for your video)
SCALE_M_PER_PX = 0.003893

# Actual swing duration in seconds (impact to finish)
REAL_SWING_DURATION = 1.5

# Gravitational acceleration in m/s^2
G = 9.81

# Contour filtering thresholds
MIN_CONTOUR_AREA = 90
MAX_CONTOUR_AREA = 100
ASPECT_RATIO_TOL = 0.2  # close to square bounding boxes


# =========================
# HELPERS
# =========================
def dump_metadata(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_video = frame_count / fps if fps > 0 else float("nan")
    print(f"[META] FPS: {fps:.3f}")
    print(f"[META] Frames: {frame_count}")
    print(f"[META] Duration (video): {duration_video:.3f} s")
    return fps, frame_count, duration_video


def find_launch_idx(points, thresh_px=5.0):
    """Find first index where movement exceeds threshold (in px)"""
    if len(points) < 2:
        return 0
    x0, y0 = points[0]
    for i in range(1, len(points)):
        dx = points[i][0] - x0
        dy = points[i][1] - y0
        if (dx * dx + dy * dy) ** 0.5 > thresh_px:
            return i
    return 0


# =========================
# MAIN PROGRAM
# =========================
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        return

    fps, frame_count, duration_video = dump_metadata(cap)

    # Calculate slow-motion correction factor (real duration / video duration)
    slow_factor = REAL_SWING_DURATION / duration_video if duration_video > 0 else 1.0
    print(f"[TIME] REAL_SWING_DURATION={REAL_SWING_DURATION:.3f}s  slow_factor={slow_factor:.4f}")

    backSub = cv2.createBackgroundSubtractorMOG2()

    ret, frame = cap.read()
    if not ret:
        print("Can't read first frame.")
        cap.release()
        return

    frame = cv2.resize(frame, (600, 600))
    print("[INFO] Select ROI, press ENTER/SPACE to confirm, 'c' to cancel.")
    roi = cv2.selectROI("ROI selector", frame, False, False)
    cv2.destroyWindow("ROI selector")

    centers = []

    print("[INFO] Press 'q' to stop tracking early (otherwise runs full video).")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (600, 600))
        fgMask = backSub.apply(frame)
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                ar = float(w) / h if h != 0 else 0
                if 1 - ASPECT_RATIO_TOL < ar < 1 + ASPECT_RATIO_TOL:
                    if roi[0] < x < roi[0] + roi[2] and roi[1] < y < roi[1] + roi[3]:
                        if 0 <= x <= frame.shape[1] and 0 <= y <= frame.shape[0]:
                            cx, cy = int(x + w / 2), int(y + h / 2)
                            centers.append((cx, cy))
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw path connecting center points
        for i in range(1, len(centers)):
            cv2.line(frame, centers[i - 1], centers[i], (0, 0, 255), 2)

        # Overlay info band
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 28), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        cv2.putText(frame, f"scale={SCALE_M_PER_PX:.6f} m/px  slow={slow_factor:.3f}x",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Analysis of tracked points (two-point approach)
    if len(centers) < 2:
        print("[RESULT] Insufficient points tracked for analysis.")
        return

    start_idx = find_launch_idx(centers)
    end_idx = len(centers) - 1

    if end_idx <= start_idx:
        start_idx = 0
        end_idx = 1 if len(centers) > 1 else 0

    (x0, y0) = centers[start_idx]
    (x1, y1) = centers[end_idx]

    dx_pix = float(x1 - x0)
    dy_pix = float(y1 - y0)
    dist_pix = float(np.hypot(dx_pix, dy_pix))
    dist_m = dist_pix * SCALE_M_PER_PX

    frames_spanned = end_idx - start_idx
    dt_video = frames_spanned / fps
    dt_real = dt_video * slow_factor
    if dt_real <= 0:
        print("[WARN] Non-positive real time; check metadata and slow_factor.")
        dt_real = 1e-6

    dx_m = dx_pix * SCALE_M_PER_PX
    dy_m = -dy_pix * SCALE_M_PER_PX  # Flip Y to positive up
    vx = dx_m / dt_real
    vy = dy_m / dt_real
    v0 = math.hypot(vx, vy)
    angle_rad = math.atan2(vy, vx)
    angle_deg = math.degrees(angle_rad)

    T_ideal = 2 * v0 * math.sin(angle_rad) / G if v0 > 0 else 0.0
    H_ideal = (v0**2) * (math.sin(angle_rad)**2) / (2 * G)
    D_ideal = (v0**2) * math.sin(2 * angle_rad) / G

    print("\n--- TWO-POINT SUMMARY ---")
    print(f"Start idx: {start_idx}, End idx: {end_idx}, Frames: {frames_spanned}")
    print(f"Δx = {dx_pix:.2f} px, Δy = {dy_pix:.2f} px, Distance = {dist_pix:.2f} px")
    print(f"Distance = {dist_m:.3f} m (Scale: {SCALE_M_PER_PX:.6f} m/px)")
    print(f"dt_video = {dt_video:.4f} s, dt_real = {dt_real:.4f} s (slow_factor={slow_factor:.4f})")
    print(f"Estimated Launch Angle: {angle_deg:.2f}°")
    print(f"Initial Velocity: {v0:.2f} m/s | {v0*3.6:.2f} km/h | {v0*2.237:.2f} mph")
    print(f"Flight Time (ideal): {T_ideal:.2f} s")
    print(f"Apex Height (ideal): {H_ideal:.2f} m")
    print(f"Horizontal Distance (ideal): {D_ideal:.2f} m")


if __name__ == "__main__":
    main()
