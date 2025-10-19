import cv2
import numpy as np
import math

# Load the video file
cap = cv2.VideoCapture(r"C:\Users\mitul\OneDrive\GolfApp\Tiger golf swing (1).mp4")

# Get frames per second from the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Set scale (metres per pixel) - adjust according to your video
scale = 0.005

# Gravity constant (unused but declared)
g = -9.81

# Create a background subtractor for motion detection
backSub = cv2.createBackgroundSubtractorMOG2()

# Read initial frame and check validity
ret, frame = cap.read()
if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    exit()

# Resize initial frame for consistent processing
frame = cv2.resize(frame, (600, 600))

# User selects region of interest (ROI)
roi = cv2.selectROI(frame, False)

# Set contour size bounds to filter detections
min_contour_area = 90
max_contour_area = 100

# Aspect ratio tolerance to find near-square shapes
aspect_ratio_tolerance = 0.2

# List to keep centres of bounding boxes
center_points = []

while True:
    # Grab new frame from video
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for consistency
    frame = cv2.resize(frame, (600, 600))

    # Compute foreground mask using background subtraction
    fgMask = backSub.apply(frame)

    # Detect contours in the foreground mask
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter contours by area size
        area = cv2.contourArea(contour)
        if min_contour_area < area < max_contour_area:
            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate aspect ratio of bounding box
            aspect_ratio = float(w) / h

            # Confirm bounding box is approximately square
            if (1 - aspect_ratio_tolerance) < aspect_ratio < (1 + aspect_ratio_tolerance):

                # Check if bounding box lies fully inside ROI and frame limits
                if (roi[0] < x < roi[0] + roi[2] and
                    roi[1] < y < roi[1] + roi[3] and
                    0 <= x <= frame.shape[1] and
                    0 <= y <= frame.shape[0]):

                    # Draw green rectangle around detected object
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Calculate and store centre of bounding box
                    center_point = (int(x + w / 2), int(y + h / 2))
                    center_points.append(center_point)

    # Draw the path connecting centre points
    for i in range(1, len(center_points)):
        cv2.line(frame, center_points[i - 1], center_points[i], (0, 0, 255), 2)

    # Show the annotated frame (mask display disabled)
    cv2.imshow('Frame', frame)

    # Exit loop on 'q' or ESC key press
    keyboard = cv2.waitKey(30)
    if keyboard == ord('q') or keyboard == 27:
        break

# Once ball is tracked, calculate initial velocity using tracked centres
if len(center_points) > 1:
    for i in range(1, len(center_points)):
        dx = center_points[i][0] - center_points[0][0]
        dy = center_points[i][1] - center_points[0][1]
        if np.sqrt(dx**2 + dy**2) > 5:  # Threshold for movement
            break

    dt = i / fps
    v0 = scale * np.sqrt(dx**2 + dy**2) / dt
    v0_mps = v0 * fps

    print(f"The estimated initial velocity is {v0_mps} metres per second.")

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
