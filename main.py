import cv2
import numpy as np

# Load video file
cap = cv2.VideoCapture(r"C:\Users\mitul\OneDrive\GolfApp\Tiger golf swing (1).mp4")

# Create background subtractor for motion detection
backSub = cv2.createBackgroundSubtractorMOG2()

# Read the first frame and check for success
ret, frame = cap.read()
if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    exit()

# Resize frame for uniform processing
frame = cv2.resize(frame, (600, 600))

# Select region of interest (ROI) manually
roi = cv2.selectROI(frame, False)

# Set min and max contour area for filtering
min_contour_area = 80  # Adjust per your needs
max_contour_area = 110

# Tolerance for aspect ratio to detect near-square shapes
aspect_ratio_tolerance = 0.2

# Initial list for storing centre points of detections
center_points = []

while True:
    # Read next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize current frame
    frame = cv2.resize(frame, (600, 600))

    # Get foreground mask via background subtraction
    fgMask = backSub.apply(frame)

    # Detect contours from the foreground mask
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter contours by area
        if min_contour_area < cv2.contourArea(contour) < max_contour_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate aspect ratio (width divided by height)
            aspect_ratio = float(w) / h

            # Check for approximately square bounding box
            if (1 - aspect_ratio_tolerance) < aspect_ratio < (1 + aspect_ratio_tolerance):
                # Confirm bounding box lies fully inside the ROI
                if (roi[0] < x < roi[0] + roi[2] and
                    roi[1] < y < roi[1] + roi[3] and
                    roi[0] < x + w < roi[0] + roi[2] and
                    roi[1] < y + h < roi[1] + roi[3]):

                    # Draw rectangle around detected object
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Compute and save centre point of bounding box
                    center_point = (int(x + w / 2), int(y + h / 2))
                    center_points.append(center_point)

    # Draw path of detected object's centre points
    for i in range(1, len(center_points)):
        cv2.line(frame, center_points[i - 1], center_points[i], (0, 0, 255), 2)

    # Display annotated frame and foreground mask
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    # Exit loop on 'q' or ESC key press
    keyboard = cv2.waitKey(30)
    if keyboard == ord('q') or keyboard == 27:
        break

# Release video resource and close windows
cap.release()
cv2.destroyAllWindows()
