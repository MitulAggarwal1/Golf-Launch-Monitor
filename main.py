import cv2
import numpy as np

# Load the video file
cap = cv2.VideoCapture(r"C:\Users\mitul\OneDrive\GolfApp\Tiger golf swing (1).mp4")

# Create a background subtractor object for motion detection
backSub = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:  # Stop if video ends or frame not read
        break

    # Resize the frame for consistent processing
    frame = cv2.resize(frame, (600, 600))

    # Apply background subtraction to extract moving objects
    fgMask = backSub.apply(frame)

    # Find contours in the foreground mask, representing moving objects
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw a green rectangle around detected moving region
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the original frame with annotations
    cv2.imshow('Frame', frame)
    # Display the foreground mask for debugging
    cv2.imshow('FG Mask', fgMask)

    # Break the loop if 'q' or ESC key is pressed
    keyboard = cv2.waitKey(30)
    if keyboard == ord('q') or keyboard == 27:
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
