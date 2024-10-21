import numpy as np
import cv2 as cv

def nothing(x):
    #removes the pesky error of 5 args required
    pass
cap = cv.VideoCapture('videoCarrito.mp4')
#create frame and slidebars to rapid adjust
cv.namedWindow("TrackBar")
cv.createTrackbar("L-H","TrackBar",0,360,nothing)
cv.createTrackbar("L-S","TrackBar",0,360,nothing)
cv.createTrackbar("L-V","TrackBar",0,360,nothing)
cv.createTrackbar("U-H","TrackBar",0,360,nothing)
cv.createTrackbar("U-S","TrackBar",255,255,nothing)
cv.createTrackbar("U-V","TrackBar",255,255,nothing)
cv.createTrackbar("MIN_AREA","TrackBar",1000,1000,nothing)
cv.createTrackbar("MAX_AREA","TrackBar",2628,5000,nothing)#closest value to correct?

# minimum & max area of detection initilization
MIN_AREA = 1000  
MAX_AREA = 2500 

# Initialize Kalman Filter
kalman = cv.KalmanFilter(4, 2)
# State vector [x, y, vx, vy]
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
# Transition matrix (constant velocity model)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
# Process noise covariance
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
# Load Haar cascade for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Initial prediction
last_prediction = np.zeros((2, 1), np.float32)
movement_threshold = 10 
last_detected_center = None

#play video
while cap.isOpened():
    ret, frame = cap.read()

    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    l_h = cv.getTrackbarPos("L-H","TrackBar")
    l_s = cv.getTrackbarPos("L-S","TrackBar") 
    l_v = cv.getTrackbarPos("L-V","TrackBar") 
    u_h = cv.getTrackbarPos("U-H","TrackBar") 
    u_s = cv.getTrackbarPos("U-S","TrackBar") 
    u_v = cv.getTrackbarPos("U-V","TrackBar")
    MIN_AREA = cv.getTrackbarPos("MIN_AREA","TrackBar")
    MAX_AREA = cv.getTrackbarPos("MAX_AREA","TrackBar") 
    


    lower_red = np.array([l_h,49,27])
    upper_red = np.array ([u_h,u_s,u_v])
 #LS=49 CLOSEST CORRECT VALUE FOR FILTRATION 
 #LV= 27 CLOSEST CORRECT VALUE FOR FILTRATION
    mask = cv.inRange(hsv,lower_red,upper_red)

    # Find contours in the mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Detect faces in the frame
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    # Store face rectangles for overlap checking
    face_rects = []
    for (x, y, w, h) in faces:
        face_rects.append((x, y, x + w, y + h))


    # Variable to store the detected contour center
    detected_center = None

    # Iterate through contours and filter based on size
    for contour in contours:
        area = cv.contourArea(contour)
        
        # Only process contours within the desired area range
        if MIN_AREA < area < MAX_AREA:
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detected_center = np.array([[cx], [cy]], np.float32)

            # Check if the contour overlaps with any detected face
                for (fx1, fy1, fx2, fy2) in face_rects:
                    if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
                        detected_center = None  # Reset detected center if overlap occurs
                        break

            # Draw the contour if no overlap with face
            if detected_center is not None:
                cv.drawContours(frame, [contour], -1, (0, 255, 0), 2)

                # Get the bounding box for the contour
                x, y, w, h = cv.boundingRect(contour)
                
                # Draw a rectangle around the contour
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Kalman filter prediction step
        prediction = kalman.predict()
        predicted_center = (int(prediction[0]), int(prediction[1]))

        # Kalman filter update step
        if detected_center is not None:
            # Calculate the distance between the detected center and the last prediction
            distance = np.linalg.norm(detected_center - last_prediction)
            # Update the Kalman filter only if the movement is significant
            if distance > movement_threshold:
                kalman.correct(detected_center)
                last_prediction = detected_center
                last_detected_center = detected_center
            else:
                detected_center = last_detected_center 
        else:
            # If no detection, continue using the last prediction
            detected_center = last_prediction

        # Draw the predicted position (circle or other shape to indicate prediction)
        cv.circle(frame, predicted_center, 5, (0, 0, 255), -1)

 
    # Indicador que el video termino
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    #display
    cv.imshow('frame',frame)
    cv.imshow('MASK',mask)
    if cv.waitKey(300) == ord('q'):
        break
 
cap.release()
cv.destroyAllWindows()