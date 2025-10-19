import cv2
import numpy as np
import math

# Load the video
cap = cv2.VideoCapture(r"C:\Users\mitul\OneDrive\GolfApp\Tiger golf swing (1).mp4")

# Get the frames per second
fps = cap.get(cv2.CAP_PROP_FPS)

# Assume the scale (in metres per pixel)
scale = 0.005  # adjust this value based on your video

# Convert g from m/s² to m/frame²
g = 9.81 / fps**2

# Read frames until the club has moved noticeably
for _ in range(65):  # adjust this value if needed
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (600, 600))

# Now manually select the region of interest (ROI)
roi = cv2.selectROI(frame, False)
cv2.destroyAllWindows()

# Set up two different trackers for reliability
tracker1 = cv2.legacy.TrackerMedianFlow_create()
tracker1.init(frame, tuple(roi))

tracker2 = cv2.legacy.TrackerMOSSE_create()
tracker2.init(frame, tuple(roi))

# Create an empty list to store the positions of the ball
positions1 = []
positions2 = []

# Make an empty image to draw the ball’s path as it moves
path_img = np.zeros_like(frame)

while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (600, 600))

    # Update the trackers on the current frame
    ret1, roi1 = tracker1.update(frame)
    ret2, roi2 = tracker2.update(frame)

    # Decide which tracker output to use
    if ret1 and ret2:
        # Use the average if both trackers succeed
        roi = ((roi1[0] + roi2[0]) / 2, (roi1[1] + roi2[1]) / 2, (roi1[2] + roi2[2]) / 2, (roi1[3] + roi2[3]) / 2)
    elif ret1:
        roi = roi1
    elif ret2:
        roi = roi2
    else:
        # Skip this frame if both fail to track
        continue

    # Draw the ROI on the frame
    p1 = (int(roi[0]), int(roi[1]))
    p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
    
    # Record the centre point of the ROI
    centre = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    positions1.append(centre)

    # Draw the path of the ball as it moves
    if len(positions1) > 1:
        cv2.line(path_img, tuple(map(int, positions1[-2])), tuple(map(int, positions1[-1])), (0, 255, 0), 2)
        frame = cv2.add(frame, path_img)
    else:
        # Indicate when tracking is lost
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Add a short delay if needed for clarity (optional)
    # cv2.waitKey(100)

    # Exit if ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Estimate the initial speed once the ball leaves the club
if len(positions1) > 1:
    # Identify when the ball has moved a noticeable distance
    for i in range(1, len(positions1)):
        dx = positions1[i][0] - positions1[0][0]
        dy = positions1[i][1] - positions1[0][1]
        if np.sqrt(dx**2 + dy**2) > 5:  # adjust if needed
            break

    # Calculate the initial velocity in metres per frame
    dt = i / fps
    v0 = scale * np.sqrt(dx**2 + dy**2) / dt

    # Convert velocity to metres per second
    v0_mps = v0 * fps

    print(f"The estimated initial velocity is {v0_mps} metres per second.")

cv2.destroyAllWindows()
cap.release()
