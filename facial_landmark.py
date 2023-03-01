import mediapipe as mp
import cv2
import numpy as np


class FacialLandmarks:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh()

    def get_facial_landmark(self, frame):
        height, width,_ = frame.shape
        frame_rgb =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)

        face_landmarks = []
        try:
            for facial_landmarks in result.multi_face_landmarks:
                for i in range(0,468):
                    pt = facial_landmarks.landmark[i]
                    x = int(pt.x * width)
                    y = int(pt.y * height)
                    face_landmarks.append([x,y])

            return np.array(face_landmarks, np.int32)

        except:
            pass
                


