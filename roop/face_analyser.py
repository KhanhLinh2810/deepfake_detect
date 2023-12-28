import dlib
import os

from typing import Optional, List
from roop.typing import Frame, Face

# Khởi tạo detector
detector = dlib.cnn_face_detection_model_v1('../models/mmod_human_face_detector.dat')

def get_faces(frame: Frame) -> Optional[List[Face]]:
    faces = detector(frame)

    return faces