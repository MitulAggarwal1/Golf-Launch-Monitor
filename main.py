import cv2

# Load the video
cap = cv2.VideoCapture(r"C:\Users\mitul\OneDrive\GolfApp\Tiger golf swing (1).mp4")

# Take first frame and find corners in it
ret, frame = cap.read()

# Resize the first frame
frame = cv2.resize(frame, (600, 600))

# Select the region of interest (ROI) where I want to start tracking (the ball or club)
roi = cv2.selectROI(frame, False)

# Set up a KCF tracker (Kernelized Correlation Filter) – designed for efficient single-object tracking
tracker = cv2.TrackerKCF_create()

# Initialise the tracker using the first frame and the manually selected ROI
ret = tracker.init(frame, roi)

while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (600, 600))

    # Update the tracker position on the current frame
    ret, roi = tracker.update(frame)

    # Draw the ROI on the frame if tracking was successful
    if ret:
        (x, y, w, h) = tuple(map(int, roi))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Exit if ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()
