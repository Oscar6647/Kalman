import numpy as np
import cv2 as cv

def nothing(x):
    #removes the pesky error of 5 args required
    pass
#inicialize the video
cap = cv.VideoCapture('Celular.mp4')

#create frame and slidebars to rapid adjust
cv.namedWindow("TrackBar")
cv.createTrackbar("L-H","TrackBar",0,180,nothing)
cv.createTrackbar("L-S","TrackBar",0,255,nothing)
cv.createTrackbar("L-V","TrackBar",0,255,nothing)
cv.createTrackbar("U-H","TrackBar",180,180,nothing)
cv.createTrackbar("U-S","TrackBar",255,255,nothing)
cv.createTrackbar("U-V","TrackBar",255,255,nothing)
cv.createTrackbar("MIN_AREA","TrackBar",0,1000,nothing)
cv.createTrackbar("NAX_AREA","TrackBar",0,5000,nothing)


MIN_AREA = 1000  # You can adjust this value
MAX_AREA = 2500  # You can adjust this value

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
 

    lower_red = np.array([l_h,49,27])
    upper_red = np.array ([u_h,u_s,u_v])
 #LS=49
 #LV= 27
    mask = cv.inRange(hsv,lower_red,upper_red)

    # Find contours in the mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Iterate through contours and filter based on size
    for contour in contours:
        area = cv.contourArea(contour)
        
        # Only process contours within the desired area range (medium-sized)
        if MIN_AREA < area < MAX_AREA:
            # Draw the medium-sized contour
            cv.drawContours(frame, [contour], -1, (0, 255, 0), 2)

            # Get the bounding box for the medium-sized contour
            x, y, w, h = cv.boundingRect(contour)
            
            # Draw a rectangle around the medium-sized contour
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

 
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