import cv2
import numpy as np
from facial_landmark import FacialLandmarks


fl = FacialLandmarks()

cap = cv2.VideoCapture("person_walking.mp4")


while True:
    ret, frame = cap.read()
    if ret ==True:
        frame = cv2.resize(frame, None, fx = 0.3, fy = 0.3)
        height, width,_ = frame.shape
        frame_copy = frame.copy()

        landmarks = fl.get_facial_landmark(frame)
        print(landmarks)
        if landmarks is not None:
            convexhull = cv2.convexHull(landmarks)

            Mask = np.zeros((height, width), frame.dtype)
            cv2.fillPoly(Mask, [convexhull], 255)

            frame_copy = cv2.blur(frame_copy,(27,27))
            face_extracted = cv2.bitwise_and(frame_copy,frame_copy, mask=Mask)

            ### Extract Background
            background_mask = cv2.bitwise_not(Mask)
            background = cv2.bitwise_and(frame,frame, mask = background_mask)


            ### Final result
            result = cv2.add(background, face_extracted)
            cv2.imshow('result', result)
        else:
            cv2.imshow('result',frame)
        key = cv2.waitKey(10)
    else:
        break

cap.release()
cv2.destroyAllWindows()