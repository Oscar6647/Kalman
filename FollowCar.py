import numpy as np
import cv2 as cv

def nothing(x):
    #removes the pesky error of 5 args required
    pass
cap = cv.VideoCapture('videoCarrito.mp4')
#create frame and slidebars to rapid adjust
cv.namedWindow("TrackBar")
cv.createTrackbar("L-H","TrackBar",20,179,nothing)
cv.createTrackbar("L-S","TrackBar",100,255,nothing)
cv.createTrackbar("L-V","TrackBar",100,255,nothing)
cv.createTrackbar("U-H","TrackBar",30,179,nothing)
cv.createTrackbar("U-S","TrackBar",255,255,nothing)
cv.createTrackbar("U-V","TrackBar",255,255,nothing)
cv.createTrackbar("MIN_AREA","TrackBar",680,1000,nothing)
cv.createTrackbar("MAX_AREA","TrackBar",1301,5000,nothing)#closest value to correct?

# minimum & max area of detection initilization
MIN_AREA = 1000  
MAX_AREA = 2500 

# Initialize Kalman Filter
kalman = cv.KalmanFilter(4, 2)

# State Transition Matrix (does the change for every diference in time?)

dt=300

# State vector [x, y, vx, vy]
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
# Transition matrix (constant velocity model)
kalman.transitionMatrix = np.array([[1, 0, dt, 0],
                                    [0, 1, 0, dt],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
# Process noise covariance
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# Measurement noise coverance
kalman.measurementNoiseCov = np.eye(2,dtype=np.float32)*1e-1

# Initial error covariance
kalman.errorCovPost = np.eye(4, dtype=np.float32)

# Initial state (start at 0 position and velocity)
kalman.statePost = np.array([0, 0, 0, 0], dtype=np.float32)


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

    lower_red = np.array([l_h,l_s,l_v])
    upper_red = np.array ([u_h,u_s,u_v])
    mask = cv.inRange(hsv,lower_red,upper_red)

    # Find contours in the mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

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
                # Predict the next state
                predicted_state = kalman.predict()

                # Measurement (detected position from the color)
                measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                
                # Correct the Kalman filter with the measurement
                kalman.correct(measurement)

                # Get the corrected state (x, y, vx, vy)
                corrected_state = kalman.statePost
                x_estimated, y_estimated = int(corrected_state[0]), int(corrected_state[1])
                # Draw the actual measured position
                cv.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # Green dot (measured)
                
                # Draw the Kalman filter predicted position
                cv.circle(frame, (x_estimated, y_estimated), 5, (255, 0, 0), -1)  # Blue dot (predicted)


                # Get the bounding box for the contour
                x, y, w, h = cv.boundingRect(contour)
                
                # Draw a rectangle around the contour
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Kalman filter prediction step
 
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