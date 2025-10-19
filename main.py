import cv2
import numpy as np
import math

# Load the video
cap = cv2.VideoCapture(r"C:\Users\mitul\OneDrive\GolfApp\Tiger golf swing (1).mp4")
ret, frame = cap.read()
frame = cv2.resize(frame, (600, 600))  # Resize the first frame after reading it

# Get the frames per second
fps = cap.get(cv2.CAP_PROP_FPS)

# Assume the scale (in metres per pixel)
scale = 0.005  # adjust this value based on your video

# Convert g from m/s² to m/frame² (gravity not used as negative now)
g = -9.81

# Read frames until the club has moved
for _ in range(65):  # adjust as necessary
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (600, 600))  # Fixed: pass a tuple to resize

# Select the region of interest (ROI) manually
roi = cv2.selectROI(frame, False)
cv2.destroyAllWindows()

# Initialize two CSRT trackers with the first frame and ROI for better reliability
tracker1 = cv2.legacy.TrackerCSRT_create()
tracker1.init(frame, tuple(roi))

tracker2 = cv2.legacy.TrackerCSRT_create()
tracker2.init(frame, tuple(roi))

# Create empty lists to store the positions of the ball
positions1 = []
positions2 = []

# Create an empty image to draw the path of the ball
path_img = np.zeros_like(frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (600, 600))
    # Apply Gaussian blur for noise reduction
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Update trackers
    ret1, roi1 = tracker1.update(frame)
    ret2, roi2 = tracker2.update(frame)

    if ret1 and ret2:
        # Average ROI if both trackers are successful
        roi = ((roi1[0] + roi2[0]) / 2, (roi1[1] + roi2[1]) / 2,
               (roi1[2] + roi2[2]) / 2, (roi1[3] + roi2[3]) / 2)
    elif ret1:
        roi = roi1
    elif ret2:
        roi = roi2
    else:
        # Skip frame if tracking fails
        continue

    # Check if ROI is within the frame boundaries to avoid errors
    if (roi[0] < 0 or roi[1] < 0 or
        roi[0] + roi[2] > frame.shape[1] or
        roi[1] + roi[3] > frame.shape[0]):
        print("The ball has exited the frame.")
        break

    # Draw the ROI rectangle on the frame
    p1 = (int(roi[0]), int(roi[1]))
    p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

    # Store the centre position of the ROI
    centre = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    positions1.append(centre)

    # Draw the path of the ball
    if len(positions1) > 1:
        cv2.line(path_img, tuple(map(int, positions1[-2])), tuple(map(int, positions1[-1])), (0, 255, 0), 2)
        frame = cv2.add(frame, path_img)
    else:
        # Indicate tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Calculate the initial velocity when the ball is hit
if len(positions1) > 1:
    for i in range(1, len(positions1)):
        dx = positions1[i][0] - positions1[0][0]
        dy = positions1[i][1] - positions1[0][1]
        if np.sqrt(dx**2 + dy**2) > 5:  # Movement threshold
            break
    dt = i / fps
    v0 = scale * np.sqrt(dx**2 + dy**2) / dt
    v0_mps = v0 * fps
    print(f"The estimated initial velocity is {v0_mps} metres per second.")

cv2.destroyAllWindows()
cap.release()
