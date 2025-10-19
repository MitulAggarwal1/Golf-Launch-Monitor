import cv2
import numpy as np
import math

# Load the video
cap = cv2.VideoCapture(r"C:\Users\mitul\OneDrive\GolfApp\Tiger golf swing (1).mp4")

# Get the frames per second of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Assume the scale of the video in metres per pixel
scale = 0.005  # Adjust this based on your setup

# Convert acceleration due to gravity to metres per frame squared
g = 9.81 / fps**2

# Create a background subtractor for foreground detection
backSub = cv2.createBackgroundSubtractorMOG2()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    exit()

# Resize initial frame for consistent processing
frame = cv2.resize(frame, (600, 600))

# Define the region of interest (ROI) manually
roi = cv2.selectROI(frame, False)

# Define minimum and maximum contour area to filter noise
min_contour_area = 90
max_contour_area = 100

# Define the tolerance for detecting roughly square contours
aspect_ratio_tolerance = 0.2

# Initialize list to store centre points of bounding boxes
center_points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize each frame for processing
    frame = cv2.resize(frame, (600, 600))

    # Apply background subtraction to get the foreground mask
    fgMask = backSub.apply(frame)

    # Find contours on the foreground mask
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area < max_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            # Only consider contours which are approximately square
            if (1 - aspect_ratio_tolerance) < aspect_ratio < (1 + aspect_ratio_tolerance):

                # Check if bounding box is fully inside the ROI and within frame boundaries
                if (roi[0] < x < roi[0] + roi[2] and
                    roi[1] < y < roi[1] + roi[3] and
                    0 <= x <= frame.shape[1] and
                    0 <= y <= frame.shape[0]):

                    # Draw bounding rectangle on the frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Store the centre point of the bounding box
                    center_points.append((int(x + w / 2), int(y + h / 2)))

    # Draw the path linking the detected centre points
    for i in range(1, len(center_points)):
        cv2.line(frame, center_points[i - 1], center_points[i], (0, 0, 255), 2)

    # Display the current frame
    cv2.imshow('Frame', frame)
    #cv2.imshow('FG Mask', fgMask)  # Uncomment to see foreground mask

    # Exit if 'q' or ESC keys are pressed
    keyboard = cv2.waitKey(1)
    if keyboard == ord('q') or keyboard == 27:
        break

if len(center_points) > 1:
    # Determine first position where ball moves appreciably
    for i in range(1, len(center_points)):
        dx = center_points[i][0] - center_points[0][0]
        dy = center_points[i][1] - center_points[0][1]
        if np.sqrt(dx**2 + dy**2) > 5:  # Movement threshold, adjust as needed
            break

    # Calculate time difference in seconds
    dt = i / fps

    # Compute initial velocity in metres per frame
    v0 = scale * np.sqrt(dx**2 + dy**2) / dt

    # Convert velocity to metres per second
    v0_mps = v0 * fps

    # Convert velocity to km/h and mph
    v0_kmph = v0_mps * 3.6
    v0_mph = v0_mps * 2.237

    print(f"Speed: {v0} m/frame, {v0_mps} m/s, {v0_kmph} km/h, {v0_mph} mph")

    # Calculate the launch angle in radians and degrees
    angle_radians = math.atan2(dy, dx)
    angle_degrees = math.degrees(angle_radians)

    print(f"Angle: {angle_radians} radians, {angle_degrees} degrees")

    # Calculate flight time, apex height, and horizontal distance (metres)
    t = 2 * v0_mps * math.sin(angle_radians) / g
    h = v0_mps**2 * math.sin(angle_radians)**2 / (2 * g)
    d = v0_mps**2 * math.sin(2 * angle_radians) / g

    print(f"Estimated launch angle: {-angle_degrees} degrees")
    print(f"Estimated initial velocity: {v0_kmph} km/h")
    print(f"Estimated initial velocity: {v0_mph} mph")
    print(f"Estimated flight time: {t} seconds")
    print(f"Estimated apex height: {-h} metres")
    print(f"Estimated horizontal distance travelled: {d} metres")

cap.release()
cv2.destroyAllWindows()
