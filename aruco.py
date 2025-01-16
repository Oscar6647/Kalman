import cv2
import cv2.aruco as aruco

# Load the predefined dictionary (you can change the dictionary for different marker sets)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Open a video file or start the webcam (use 0 for webcam)
cap = cv2.VideoCapture(0)  # Replace with "path_to_video.mp4" for a video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale as ArUco detection works better on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the grayscale image
    corners, ids, rejected = detector(gray, dictionary, parameters=parameters)

    # Draw detected markers and their IDs on the original frame
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Display each marker's ID
        for i in range(len(ids)):
            c = corners[i][0]
            # Draw marker's ID near the top-left corner of the marker
            cv2.putText(frame, f"ID: {ids[i][0]}", (int(c[0][0]), int(c[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detected markers
    cv2.imshow("ArUco Marker Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
