import cv2
import numpy as np
import math

# Load the video
cap = cv2.VideoCapture(r"C:\Users\mitul\OneDrive\GolfApp\Tiger golf swing (1).mp4")

# Get how many frames per second the video has
fps = cap.get(cv2.CAP_PROP_FPS)

# Set the scale in metres per pixel (change based on your setup)
scale = 0.005

# Convert gravitational acceleration for frame-based units (unused here)
g = -9.81

# Create a background subtractor for detecting moving objects
backSub = cv2.createBackgroundSubtractorMOG2()

# Note: Skipped initial frame read and ROI selection here on purpose

# Define minimum and maximum contour area for filtering
min_contour_area = 90
max_contour_area = 100

# Tolerance for near-square contours
aspect_ratio_tolerance = 0.2

# List to store the centres of detected bounding boxes
center_points = []

while True:
    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for uniform processing
    frame = cv2.resize(frame, (600, 600))

    # Generate the foreground mask via background subtraction
    fgMask = backSub.apply(frame)

    # Find contours from the foreground mask
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Only keep contours within size limits
        if min_contour_area < cv2.contourArea(contour) < max_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            # Check for roughly square bounding boxes
            if (1 - aspect_ratio_tolerance) < aspect_ratio < (1 + aspect_ratio_tolerance):

                # Skipping ROI and frame boundary checks

                # Draw box and store centre point
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                center_points.append((int(x + w/2), int(y + h/2)))

    # Draw trajectory of the detected object
    for i in range(1, len(center_points)):
        cv2.line(frame, center_points[i - 1], center_points[i], (0, 0, 255), 2)

    # Show the processed frame
    cv2.imshow('Frame', frame)
    # Commented out fgMask display for clarity
    # cv2.imshow('FG Mask', fgMask)

    # Exit on 'q' or ESC
    keyboard = cv2.waitKey(1)
    if keyboard == ord('q') or keyboard == 27:
        break

cap.release()
cv2.destroyAllWindows()
