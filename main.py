import cv2
import numpy as np
import math

# Load the video file
cap = cv2.VideoCapture(r"C:\Users\mitul\OneDrive\GolfApp\Tiger golf swing (1).mp4")

# Retrieve frames per second
fps = cap.get(cv2.CAP_PROP_FPS)

# Set the scale in metres per pixel (adjust to your video)
scale = 0.005

# Convert acceleration due to gravity from m/s² to m/frame²
g = 9.81 / fps**2

# Create background subtractor for motion detection
backSub = cv2.createBackgroundSubtractorMOG2()

# Read the first frame and verify success
ret, frame = cap.read()
if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    exit()

# Resize the frame for uniform processing
frame = cv2.resize(frame, (600, 600))

# Manually select the region of interest (ROI)
roi = cv2.selectROI(frame, False)

# Set minimum and maximum contour areas for filtering
min_contour_area = 90
max_contour_area = 100

# Define aspect ratio tolerance for near-square objects
aspect_ratio_tolerance = 0.2

# Initialise a list to hold bounding box centre points
center_points = []

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for consistency
    frame = cv2.resize(frame, (600, 600))

    # Generate the foreground mask
    fgMask = backSub.apply(frame)

    # Find contours within the foreground mask
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter contours by area size
        if min_contour_area < cv2.contourArea(contour) < max_contour_area:
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate the bounding box's aspect ratio
            aspect_ratio = float(w) / h

            # Check if bounding box is approximately square within tolerance
            if (1 - aspect_ratio_tolerance) < aspect_ratio < (1 + aspect_ratio_tolerance):

                # Check if bounding box lies fully inside the ROI and frame dimensions
                if (roi[0] < x < roi[0] + roi[2] and
                    roi[1] < y < roi[1] + roi[3] and
                    0 <= x <= frame.shape[1] and
                    0 <= y <= frame.shape[0]):

                    # Draw a green rectangle around the contour
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Compute and store the centre of the bounding box
                    center_point = (int(x + w / 2), int(y + h / 2))
                    center_points.append(center_point)

    # Draw the trajectory/path by connecting centre points
    for i in range(1, len(center_points)):
        cv2.line(frame, center_points[i - 1], center_points[i], (0, 0, 255), 2)

    # Show the annotated frame
    cv2.imshow('Frame', frame)
    #cv2.imshow('FG Mask', fgMask)  # Optional mask display

    # Exit loop on pressing 'q' or ESC
    keyboard = cv2.waitKey(30)
    if keyboard == ord('q') or keyboard == 27:
        break

# Calculate initial velocity and launch angle once ball motion starts
if len(center_points) > 1:
    for i in range(1, len(center_points)):
        dx = center_points[i][0] - center_points[0][0]
        dy = center_points[i][1] - center_points[0][1]
        if np.sqrt(dx**2 + dy**2) > 5:  # Movement threshold
            break

    # Calculate time interval in seconds
    dt = i / fps

    # Calculate initial velocity in metres per frame
    v0 = scale * np.sqrt(dx**2 + dy**2) / dt

    # Convert velocity to metres per second
    v0_mps = v0 * fps

    # Calculate launch angle in radians (coordinate system: negative dy for upright)
    angle_radians = math.atan2(-dy, dx)

    # Convert angle to degrees
    angle_degrees = math.degrees(angle_radians)

    print(f"The estimated launch angle is {angle_degrees} degrees")
    print(f"The estimated initial velocity is {v0_mps} metres per second.")

# Clean up resources    
cap.release()
cv2.destroyAllWindows()
