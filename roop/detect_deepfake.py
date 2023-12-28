import dlib
import numpy as np

import roop.globals


from keras.models import load_model


from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video
from roop.face_analyser import get_faces
from roop.typing import Frame


model_path = "../models/model_checkpoint.h5"
model_detect_deepfake = load_model(model_path)

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['http://dlib.net/files/mmod_human_face_detector.dat.bz2'])
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.source_path) and not is_video(roop.globals.source_path):
        # update_status('Select an image or video for source path.', NAME)
        return False
    if is_image(roop.globals.source_path) and not get_faces(dlib.load_rgb_image(roop.globals.source_path)):
        # update_status('No face in source path detected.', NAME)
        print("No face in source path detected.")
    return True



def check_deepfake(frame: Frame) -> int:
    prediction = model_detect_deepfake.predict(np.expand_dims(Frame, axis=0))
    if prediction < 0.5:
        return 0
    else:
        return 1


def process_image(source_path: str) -> bool:
    source_frame = dlib.load_rgb_image(roop.globals.source_path)
    faces = get_faces(source_frame)
    is_deepfake_image = False

    for face in faces :
        left = face.rect.left()
        top = face.rect.top()
        right = face.rect.right()
        bottom = face.rect.bottom()

        tmp_image_face = source_frame[top:bottom, left:right]

        if (check_deepfake(tmp_image_face)==1):
            is_deepfake_image = True
            break
    return is_deepfake_image
