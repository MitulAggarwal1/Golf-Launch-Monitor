import cv2
import numpy as np
import math
from pathlib import Path

# Prompt user for golf club selection
club = input("Please select club: ")

# Prompt for the path to the swing video file
video_path = input("Please provide the path to your " + club + " swing video: ")

# Remove any quotation marks from the path string
video_path = video_path.strip('"')

# Convert the string path to a Path object for better path handling
video_path = Path(video_path)

# Load the video using OpenCV, converting Path object to string
cap = cv2.VideoCapture(str(video_path))

# Obtain frames per second of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Scale conversion: metres per pixel (adjust based on video calibration)
scale = 0.005

# Gravity constant (negative for frame wise calculation)
g = -9.81

# Create background subtractor to extract moving regions
backSub = cv2.createBackgroundSubtractorMOG2()

# Read the first frame and confirm video is accessible
ret, frame = cap.read()
if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    exit()

# Resize frame to a standard manageable size
frame = cv2.resize(frame, (600, 600))

# Predefined ROI for the golf swing related to your setup (x,y,width,height)
roi = (267, 455, 328, 137)

# Set the contour size limits for filtering noise efficiently
min_contour_area = 90
max_contour_area = 100

# Aspect ratio tolerance to detect near-square bounding boxes
aspect_ratio_tolerance = 0.2

# Initialise list to record centre points of detected bounding boxes
center_points = []

while True:
    ret, frame = cap.read()
    if not ret:  # Stop loop if no frames are left
        break

    # Resize the current frame consistently
    frame = cv2.resize(frame, (600, 600))

    # Apply background subtraction to isolate moving foreground
    fgMask = backSub.apply(frame)

    # Detect contours in the foreground mask
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        # Filter contours by the area range
        if min_contour_area < area < max_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            # Confirm bounding box is approximately square
            if (1 - aspect_ratio_tolerance) < aspect_ratio < (1 + aspect_ratio_tolerance):

                # Check if bounding box lies completely inside ROI and frame boundaries
                if (roi[0] < x < roi[0] + roi[2] and
                    roi[1] < y < roi[1] + roi[3] and
                    0 <= x <= frame.shape[1] and
                    0 <= y <= frame.shape[0]):

                    # Draw bounding rectangle around the detected object
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Calculate and store centre point coordinates
                    center_point = (int(x + w / 2), int(y + h / 2))
                    center_points.append(center_point)

    # Draw connecting lines between centre points to visualise the path
    for i in range(1, len(center_points)):
        cv2.line(frame, center_points[i - 1], center_points[i], (0, 0, 255), 2)

    # Display the annotated frame
    cv2.imshow('Frame', frame)
    # Uncomment next line to view foreground mask for debugging
    # cv2.imshow('FG Mask', fgMask)

    # Exit loop on pressing 'q' or ESC key
    keyboard = cv2.waitKey(1)
    if keyboard == ord('q') or keyboard == 27:
        break

# Calculation of initial velocity and flight parameters
if len(center_points) > 1:
    for i in range(1, len(center_points)):
        dx = center_points[i][0] - center_points[0][0]
        dy = center_points[i][1] - center_points[0][1]
        # Detect significant movement of ball
        if np.sqrt(dx**2 + dy**2) > 5:
            break

    # Compute time difference based on frame count and fps
    dt = i / fps

    # Calculate velocity in metres/frame scaled by scale factor
    v0 = scale * np.sqrt(dx**2 + dy**2) / dt

    # Convert velocity to metres per second
    v0_mps = v0 * fps

    # Convert velocities to km/h and mph for common understanding
    v0_kmph = v0_mps * 3.6
    v0_mph = v0_mps * 2.237

    # Calculate launch angle in radians and then degrees
    angle_radians = math.atan2(dy, dx)
    angle_degrees = math.degrees(angle_radians)

    # Calculate trajectory parameters: flight time, apex height, horizontal distance
    t = 2 * v0_mps * math.sin(angle_radians) / g
    h = v0_mps**2 * math.sin(angle_radians)**2 / (2 * g)
    d = v0_mps**2 * math.sin(2 * angle_radians) / g

    # Print results with signs adjusted for clarity
    print(f"The estimated launch angle is {-angle_degrees} degrees")
    print(f"The estimated initial velocity is {v0_kmph} km/h or {v0_mph} mph.")
    print(f"The estimated flight time is {t} seconds.")
    print(f"The estimated apex of the curve is {-h} meters.")
    print(f"The estimated horizontal distance travelled is {d} meters.")

# Release video resources and close display windows
cap.release()
cv2.destroyAllWindows()
