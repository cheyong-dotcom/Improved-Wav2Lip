import os
import dlib
import cv2
import glob
import tqdm
import numpy as np
from multiprocessing import Pool
import json


predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.isfile(predictor_path):
	print(
		"You can download a trained facial shape predictor from:\n"
        "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
	)
	exit()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def get_landmarks(img_dir):
    img_path_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
    lms_dict = {}
    for img_path in img_path_list:
        img_id = str(os.path.basename(img_path).split(".")[0])
        try:
            img = dlib.load_rgb_image(img_path)
            dets = detector(img, 1)
            shape = predictor(img, dets[0])
            landmark_points = [(shape.part(i).x, shape.part(i).y) for i in range(0, shape.num_parts)]
            lms_dict[img_id] = landmark_points
        except Exception as e:
            lms_dict[img_id] = None
            with open("error.txt", "a") as f:
                f.write(img_path + "\n")

    json_path = os.path.join(img_dir, "face_landmarks.json")
    with open(json_path, 'w') as json_file: 
        json.dump(lms_dict, json_file)
    print(f"{img_dir}已完成")
    return



if __name__ == "__main__":
    data_root = r"E:\code_dev\xunlianji\HQ_Processed_ALL"
    subdirectories = []
    for root, dirs, files in os.walk(data_root):
        level = root.count(os.sep) - data_root.count(os.sep)
        if level == 1:
            for dir in dirs:
                subdirectory_path = os.path.join(root, dir)
                subdirectories.append(subdirectory_path)
    num_workers = 6
    with Pool(num_workers) as p:
        p.starmap(get_landmarks, [(subdir,) for subdir in subdirectories])

        




     

