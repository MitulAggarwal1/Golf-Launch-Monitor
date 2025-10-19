import os
import time
import cv2
from ultralytics import YOLO

# =========================
# USER SETTINGS
# =========================
VIDEO_PATH = r"C:\Users\mitul\OneDrive\GolfApp\Tiger golf swing 2.mp4"

# Prefer OpenVINO model directory (if exported), otherwise fallback to PyTorch weights file
OPENVINO_DIR = r"C:\Users\mitul\runs\detect\train2\weights\best_openvino_model"
BEST_PT_PATH = r"C:\Users\mitul\runs\detect\train2\weights\best.pt"

# Output path for the processed annotated video
OUT_PATH = r".\ball_annotated.mp4"

# Detection and display parameters for inference
IMG_SIZE = 960       # Image size used by YOLO for inference; more pixels = slower but more accurate
MIN_WIDTH = 960      # Minimum width of video frames resized for detection consistency
CONF_THRES = 0.20    # Confidence threshold for filtering YOLO detections
IOU_THRES = 0.60     # Intersection over Union threshold for non-maximum suppression
VID_STRIDE = 1       # Frame skipping interval during processing; 1 means every frame
SHOW_PREVIEW_600 = True  # Whether to show live 600x600 preview window during video processing
DRAW_CONFIDENCE = True   # Whether to display confidence score on detected bounding boxes

def load_model():
    """Load the OpenVINO model if available, else fall back to PyTorch model."""
    if os.path.isdir(OPENVINO_DIR):
        print(f"[MODEL] Loading OpenVINO model: {OPENVINO_DIR}")
        return YOLO(OPENVINO_DIR)
    elif os.path.isfile(BEST_PT_PATH):
        print(f"[MODEL] OpenVINO not found. Loading PyTorch model: {BEST_PT_PATH}")
        return YOLO(BEST_PT_PATH)
    else:
        raise FileNotFoundError(
            f"Could not find model.\nTried OpenVINO directory:\n  {OPENVINO_DIR}\n"
            f"and PyTorch weights file:\n  {BEST_PT_PATH}"
        )

def main():
    # Load YOLO detection model
    model = load_model()

    # Open the input video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f"Could not open video: {VIDEO_PATH}"

    # Get the original properties of the video to preserve them in output
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[META] fps={orig_fps:.2f} size={orig_w}x{orig_h}")

    # Initialize a video writer with original dimensions and adjusted FPS to save annotated results
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = orig_fps / max(VID_STRIDE, 1)
    writer = cv2.VideoWriter(OUT_PATH, fourcc, out_fps, (orig_w, orig_h))
    print(f"[OUT ] Writing to: {os.path.abspath(OUT_PATH)} @ {out_fps:.2f} FPS")

    t0 = time.time()
    processed_frames = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # Skip frames if the stride is greater than 1 to speed up processing
        if (frame_idx - 1) % max(VID_STRIDE, 1) != 0:
            continue

        # Upscale smaller frames to minimum width for consistent detection quality
        h, w = frame.shape[:2]
        if w < MIN_WIDTH:
            scale_h = int(h * (MIN_WIDTH / w))
            proc = cv2.resize(frame, (MIN_WIDTH, scale_h))
        else:
            proc = frame

        # Run YOLO object detection inference on the current frame
        res = model.predict(
            proc,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=False
        )[0]

        # Scale the detected bounding boxes back to original frame size
        sx = frame.shape[1] / proc.shape[1]
        sy = frame.shape[0] / proc.shape[0]

        # If detections exist, draw bounding boxes and optionally confidence scores
        if res.boxes is not None:
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(boxes_xyxy, confs):
                x1, x2 = int(x1 * sx), int(x2 * sx)
                y1, y2 = int(y1 * sy), int(y2 * sy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                if DRAW_CONFIDENCE:
                    cv2.putText(
                        frame, f"{c:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1
                    )

        # Write the annotated frame to the output video
        writer.write(frame)
        processed_frames += 1

        # Display optional live preview window with current processing FPS
        if SHOW_PREVIEW_600:
            disp = cv2.resize(frame, (600, 600))
            elapsed = time.time() - t0
            fps_proc = processed_frames / max(elapsed, 1e-6)
            cv2.putText(
                disp, f"proc FPS: {fps_proc:.1f}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.imshow("Ball detector (preview)", disp)
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                break

    # Clean up resources after processing
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Saved annotated video to: {os.path.abspath(OUT_PATH)}")


# Execute main program
if __name__ == "__main__":
    main()
