import cv2
import numpy as np

# Load the video I want to analyze
cap = cv2.VideoCapture(r"C:\Users\mitul\OneDrive\GolfApp\Tiger golf swing (1).mp4")

# Set up parameters for Lucas-Kanade optical flow (not directly used yet, just kept for reference)
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Read the first frame from the video
ret, old_frame = cap.read()

# Resize the first frame to a manageable resolution
old_frame = cv2.resize(old_frame, (600, 600))

# Convert the first frame to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Create a blank mask which I’ll use for drawing later
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop if the video has ended or failed to read a frame

    # Resize frame for consistency
    frame = cv2.resize(frame, (600, 600))

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define a color range to isolate bright/white regions (like clothing or the ball)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 25, 255])
    mask = cv2.inRange(frame, lower_white, upper_white)

    # Apply simple morphological operations to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    # Find contours for all the white regions detected in the frame
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pick the largest contour (most likely the main object in view)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Draw a circle only if it’s large enough to be meaningful
        if radius > 10:
            cv2.circle(frame, center, radius, (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow('frame', frame)

    # Stop the loop if the ESC key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release resources once done
cv2.destroyAllWindows()
cap.release()
