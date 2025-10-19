import cv2
import numpy as np

# Load the video file
cap = cv2.VideoCapture(r"C:\Users\mitul\OneDrive\GolfApp\Tiger golf swing (1).mp4")

# Create a background subtractor for motion detection
backSub = cv2.createBackgroundSubtractorMOG2()

# Read the first frame to initialise
ret, frame = cap.read()
if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    exit()

# Resize the first frame for consistent processing
frame = cv2.resize(frame, (600, 600))

# Allow user to select the region of interest (ROI)
roi = cv2.selectROI(frame, False)

while True:
    # Read next frame from video
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to maintain size consistency
    frame = cv2.resize(frame, (600, 600))

    # Apply background subtraction to isolate moving objects (foreground mask)
    fgMask = backSub.apply(frame)

    # Find contours in the foreground mask representing motion regions
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get bounding rectangle for each moving object
        x, y, w, h = cv2.boundingRect(contour)

        # Only consider contours fully inside the ROI
        if roi[0] < x < roi[0] + roi[2] and roi[1] < y < roi[1] + roi[3] and \
           roi[0] < x + w < roi[0] + roi[2] and roi[1] < y + h < roi[1] + roi[3]:
            # Draw bounding box around motion within ROI
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the annotated frame and foreground mask for visual feedback
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    # Exit loop if 'q' or ESC is pressed
    keyboard = cv2.waitKey(30)
    if keyboard == ord('q') or keyboard == 27:
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
