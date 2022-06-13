import os
import random

import time
from datetime import datetime

import cv2
import dlib
import face_recognition
from PIL import Image

import torch
import numpy as np
import matplotlib.pyplot as plt

from clustering import calc

print('dlib.DLIB_USE_CUDA : ', dlib.DLIB_USE_CUDA)
print('cv2.__version__ : ', cv2.__version__)


class FaceExtractor():
    def __init__(self, video_path, result_dir, test_img_path="/opt/ml/person-clustering/test.jpg", seed=79):
        # seed everything
        self._seed_everything(seed)
        # Input person image and activate gpu
        self._activate_gpu(test_img_path)
        
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        assert self.video.isOpened(), f"Could not Open : {video_path}"
        self.running = False

        self.result_dir = result_dir
        self._is_exist(self.result_dir)


    def run(self, 
            face_cnt=350,
            frame_batch_size=16,
            face_cloth_weights=[1.0, 1.0],
            use_scene_transition=False,
            capture_interval_sec=3,
            stop_sec=0,
            skip_sec=0,
            resizing_resolution=1000,
            resizing_ratio=None
        ):

        video = cv2.VideoCapture(self.video_path)
        fingerprints = dict()

        frames = []
        frame_idx = 0
        fps = self.video.get(cv2.CAP_PROP_FPS)

        last_down_frame = None
        last_org_frame = None  
        start_frame_idx = 0
        start_down_frame = None
        start_org_frame = None
        min_scene_frames = 15
        timelines = []
        down_scale_factor = 8
        transition_threshold = 100

        capture_interval = capture_interval_sec * int(round(fps)) # n초 간격 프레임 캡쳐

        print("====== Start Extracting... ======")
        self.running = True
        total_start = time.time()
        while self.running:
            ret, frame = video.read()
            if frame is None:
                break

            seconds = int(round(frame_idx / fps, 3))
            # print(f"Running in {seconds} sec in video...")
            if seconds > stop_sec > 0:
                break
            if seconds < skip_sec:
                frame_idx += 1
                continue

            ###### Frame extraction ######
            if use_scene_transition:
                cur_down_frame = frame[::down_scale_factor, ::down_scale_factor, :]
                        
                if last_down_frame is None:
                    last_down_frame = cur_down_frame
                    last_org_frame = frame
                    start_frame_idx = frame_idx
                    start_down_frame = cur_down_frame
                    start_org_frame = frame
                    frame_idx += 1
                    continue
                            
                num_pixels = cur_down_frame.shape[0] * cur_down_frame.shape[1]
                rgb_distance = np.abs(cur_down_frame - last_down_frame) / float(num_pixels)
                rgb_distance = rgb_distance.sum() / 3.0
                        
                if rgb_distance > transition_threshold and frame_idx - start_frame_idx > min_scene_frames:
                    # print("({}~{})".format(start_frame_idx, frame_idx-1))  
                    # resize frame
                    if resizing_ratio and frame.shape[1] >= resizing_resolution:
                        start_org_frame = cv2.resize(start_org_frame, None, fx=resizing_ratio, fy=resizing_ratio)
                        last_org_frame = cv2.resize(last_org_frame, None, fx=resizing_ratio, fy=resizing_ratio)      
                    frames.append(start_org_frame)
                    frames.append(last_org_frame)
                            
                    start_frame_idx = frame_idx
                    start_down_frame = cur_down_frame
                    start_org_frame = frame
                        
                last_down_frame = cur_down_frame
                last_org_frame = frame
            else:
                if frame_idx % capture_interval == 0:
                    # resize frame
                    if resizing_ratio and frame.shape[1] >= resizing_resolution:
                        frame = cv2.resize(frame, None, fx=resizing_ratio, fy=resizing_ratio)
                    frames.append(frame)
                    
            ###### detect_faces ######
            if len(frames) < frame_batch_size:
                frame_idx += 1
                continue        
            else:
                frame_fingerprints = self.detect_faces(frames, frame_batch_size, face_cloth_weights)
                if frame_fingerprints:
                    fingerprints.update(frame_fingerprints)
                    print('# of face images: ', len(fingerprints))
                    print('# of frames: ', frame_idx)
                    print()   
                frames = []
            
            ###### finish loop ######
            if len(fingerprints) >= face_cnt:
                break
            
            frame_idx += 1


        print("====== Finish Extracting... ======")

        self.running = False
        video.release()
        print()

        total_end = time.time()
        print('Inference time: ',total_end-total_start)
        print("Captured frames : ", frame_idx)
        print("Total face images : ", len(fingerprints))

        return fingerprints
        

    def detect_faces(self, frames, batch_size, face_cloth_weights):
        # face locations
        batch_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0, batch_size=batch_size)
        frames = [frames[i] for i in range(len(frames)) if 1 <= len(batch_face_locations[i]) <= 4]
        batch_face_locations = [x for x in batch_face_locations if (1 <= len(x) <= 4)]

        if len(batch_face_locations) == 0:
            return None

        faces = []
        now = datetime.now()
        str_ms = now.strftime('%Y%m%d_%H%M%S.%f')[:-3] + '-'

        cloth_encoding_model = calc.get_model() # resnet

        fingerprints = dict()
        face_encodings_batch = []
        upper_body_images_batch = []
        clothes_batch = []
        
        # face encodings
        for frame_number_in_batch, face_locations in enumerate(batch_face_locations):
            face_encodings = []
            for face_location in face_locations:
                top, right, bottom, left = face_location
                resized_frame = cv2.resize(frames[frame_number_in_batch][top:bottom,left:right], dsize=(224,224))
                resized_encodings = face_recognition.face_encodings(resized_frame,[(0,223,223,0)], model='small')[0] # list 안에 인물 수만큼 numpy array
                face_encodings.append(resized_encodings)
            # crop face image
            upper_body_images, cloth_images = self._get_face_and_cloth_image(frames[frame_number_in_batch], face_locations) # list 형태로 반환
            # cloth preprocessing
            preprocessed_cloth_images = self._preprocess(cloth_images, (224,224))
            
            # save batch
            face_encodings_batch.extend(face_encodings)
            upper_body_images_batch.extend(upper_body_images)
            clothes_batch.extend(preprocessed_cloth_images)

        # cloth encodings
        cloth_encodings = calc.fingerprint(clothes_batch, cloth_encoding_model, device = torch.device(device='cuda'))

        # calculate fingerprints (feature vectors)
        face_weight, cloth_weight = face_cloth_weights
        for i in range(len(face_encodings_batch)):
            # normalize
            normalized_face_encoding = face_encodings_batch[i] / np.linalg.norm(face_encodings_batch[i])
            normalized_cloth_encoding = cloth_encodings[i] / np.linalg.norm(cloth_encodings[i])
            # concat features [face | cloth]
            encoding = np.concatenate((normalized_face_encoding*face_weight, normalized_cloth_encoding*cloth_weight), axis=0) # 128-d + 128-d
            # save image
            filename = str_ms + str(i) + ".png"
            filepath = os.path.join(self.result_dir, filename)
            cv2.imwrite(filepath, upper_body_images_batch[i])
            # print('image saved path: ', filepath)
            # save fingerprint
            fingerprints[filepath] = encoding

        return fingerprints

    def print_video_infos(self):
        length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video.get(cv2.CAP_PROP_FPS)

        print('====== Video Infos ======')
        print("length :", length)
        print("width :", width)
        print("height :", height)
        print("fps :", fps)

    def _get_face_and_cloth_image(self, frame, boxes):
        '''
        param:
            frame: 프레임 이미지
            box: 좌표값 list [(top, right, bottom, left), ...]
        return:
            padded_faces: 얼굴 이미지 list [(numpy array), ...]
            padded_clothes: 옷 이미지 list [(numpy array), ...]
        '''

        padded_faces = []
        padded_clothes = []
        img_height, img_width = frame.shape[:2]

        for box in boxes:
            (box_top, box_right, box_bottom, box_left) = box # 딱 얼굴 이미지
            box_width = box_right - box_left
            box_height = box_bottom - box_top
            # padding
            crop_top = max(box_top - box_height, 0)
            pad_top = -min(box_top - box_height, 0)
            crop_bottom = min(box_bottom + box_height, img_height - 1)
            pad_bottom = max(box_bottom + box_height - img_height, 0)
            crop_left = max(box_left - box_width, 0)
            pad_left = -min(box_left - box_width, 0)
            crop_right = min(box_right + box_width, img_width - 1)
            pad_right = max(box_right + box_width - img_width, 0)
            # cropping
            face_image = frame[crop_top:crop_bottom, crop_left:crop_right]
            cloth_image = frame[box_bottom+int(box_height*0.2):crop_bottom, crop_left:crop_right]
            # return
            if (pad_top == 0 and pad_bottom == 0):
                if (pad_left == 0 and pad_right == 0):
                    padded_faces.append(face_image)
                    padded_clothes.append(cloth_image)
                    continue
            padded_face = cv2.copyMakeBorder(face_image, pad_top, pad_bottom,
                                            pad_left, pad_right, cv2.BORDER_CONSTANT)
            padded_cloth = cv2.copyMakeBorder(cloth_image, pad_top, pad_bottom,
                                            pad_left, pad_right, cv2.BORDER_CONSTANT)
            padded_faces.append(padded_face)
            padded_clothes.append(padded_cloth)

        return padded_faces, padded_clothes

    def _preprocess(self, images, size):
        try:
            imgs = []
            for image in images:
                img = Image.fromarray(image).convert('RGB').resize(size, resample=3)
                # arr = np.asarray(img).astype(int)
                imgs.append(img)
            return imgs # arr
        except OSError as ex:
            print(f"skipping file...: {ex}")
            return None


    def _seed_everything(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore


    def _activate_gpu(self, test_img_path):
        image = face_recognition.load_image_file(test_img_path)
        face_locations = face_recognition.face_locations(image, model='cnn')
        if len(face_locations) > 0:
            print('Using GPU')
        else:
            print('***Not using GPU***')

    def _is_exist(self, dir_path):
        try:    
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except OSError:
            print('Error: Creaing directory. ' + dir_path)