import cv2
import numpy as np

# Load the video file
cap = cv2.VideoCapture(r"C:\Users\mitul\OneDrive\GolfApp\Tiger golf swing (1).mp4")

# Create a background subtractor for detecting motion
backSub = cv2.createBackgroundSubtractorMOG2()

# Read the first frame and handle failure gracefully
ret, frame = cap.read()
if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    exit()

# Resize the initial frame to a fixed size
frame = cv2.resize(frame, (600, 600))

# Manually select the region of interest (ROI)
roi = cv2.selectROI(frame, False)

# Define contour area thresholds for filtering detections
min_contour_area = 80   # Minimum contour area
max_contour_area = 110  # Maximum contour area

# Define tolerance for aspect ratio to detect approximately square contours
aspect_ratio_tolerance = 0.2

while True:
    # Capture a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize each frame for consistency
    frame = cv2.resize(frame, (600, 600))

    # Apply the background subtractor for foreground mask
    fgMask = backSub.apply(frame)

    # Find contours on the foreground mask
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter contours by area
        if min_contour_area < cv2.contourArea(contour) < max_contour_area:
            # Get bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate aspect ratio of bounding box
            aspect_ratio = float(w) / h

            # Check if bounding box is approximately square
            if (1 - aspect_ratio_tolerance) < aspect_ratio < (1 + aspect_ratio_tolerance):

                # Verify bounding box lies completely within the ROI
                if roi[0] < x < roi[0] + roi[2] and roi[1] < y < roi[1] + roi[3] and \
                   roi[0] < x + w < roi[0] + roi[2] and roi[1] < y + h < roi[1] + roi[3]:

                    # Draw bounding rectangle on the original frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show frame and foreground mask
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    # Exit on 'q' or ESC key press
    keyboard = cv2.waitKey(30)
    if keyboard == ord('q') or keyboard == 27:
        break

# Release video and close all windows
cap.release()
cv2.destroyAllWindows()
