import os
from os.path import join
import argparse
import subprocess
import cv2
from tqdm import tqdm
import numpy as np
import logging
import torch
from test_tools.common import detect_all, grab_all_frames
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.warp_for_xray import (
    estimiate_batch_transform,
    transform_landmarks,
    std_points_256,
)
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.utils import get_crop_box
import datetime
# from FaceForensics.face_detection_save import get_boundingbox

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device=torch.device('cuda')
#Date
now = datetime.datetime.now()
logger = logging.getLogger("main") #Logger 선언
stream_handler = logging.StreamHandler() # Logger output 방법 선언
formatter = logging.Formatter('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

#### 각 manipulation method 별 root path ####
# DATASET_PATHS = {
#     ''
#     'FaceShifter' : 'manipulated_sequences/FaceShifter',
#     # 'original': 'original_sequences/actors',
#     # 'Deepfakesdetection': 'manipulated_sequences/DeepFakeDetection',
#     # 'FaceShifter': 'manipulated_sequences/FaceShifter',
#     'original': 'original_sequences/youtube',
#     'Deepfakes': 'manipulated_sequences/Deepfakes',
#     'Face2Face': 'manipulated_sequences/Face2Face',
#     'FaceSwap': 'manipulated_sequences/FaceSwap',
#     'NeuralTextures': 'manipulated_sequences/NeuralTextures',
#     # 'Celeb-real':'Celeb-real',
#     # 'Celeb-fake':'Celeb-synthesis',
    
#     # 'youtube-real':'YouTube-real'
#     #,'test' : 'test' # 직접 찍은 동영상으로 해보기
# }

# COMPRESSION = ['c0', 'c23', 'c40', 'raw']
crop_align_func = FasterCropAlignXRay(256)
max_frame= 10000

### video path를 받으면 crop된 face를 저장하는 함수 ###
def crop_face_from_video(video_path,cache_path,crop_path,clip_size):
    # mp4 파일이 아니면 return
    if 'mp4' not in video_path : return
    # video name
    video_name = video_path.split('/')[-1].replace('.mp4','')
    # 만약 crop image path에 crop된 이미지가 110개 이상이면 return
    # 중복 방지용 코드 삭제해도 무관
    if os.path.exists(crop_path):
        if len(os.listdir(crop_path))>110:
            logger.info(f'{video_name} already exists')
            return
        
    ##########################################
    # detect_res : list, 전체 frame, whole frame
    # detect_res [] : list, len = 사람 수로 예상 the number of detected face in a frame
    # detect_res [] [] : tuple, length = 3 
    # detect_res [] [] 의 각 요소는 각각 box, lm5 : landmark (5,2) , score 
    ##########################################
    # all_lm68 : list, 전체 frame, whole frame
    # all_lm68 : list, len = 사람 수로 예상, the number of detected face in a frame
    # all_lm68 : np.array : landmark 68개 (68,2)
    ##########################################
    # frames : each frame's np.array
    
    
    # cache_file : cache file path
    # landmark와 box를 저장하는 cache file
    cache_file = f"{cache_path}.pth"
   
    if os.path.exists(cache_file):
        # cache file이 존재하면 load하고 frame만 불러옴
        detect_res, all_lm68 = torch.load(cache_file)
        frames = grab_all_frames(video_path, max_size=max_frame, cvt=True)
        logger.info("detection result loaded from cache")
    else:
        # cache file이 존재하지 않으면 detect_all 함수를 통해 detect_res, all_lm68, frames를 불러옴
        # detection_all 함수는 retina_face를 이용해서 box와 landmark를 찾는 함수
        detect_res, all_lm68, frames = detect_all(
                    video_path, return_frames=True, max_size=10000
                )
        torch.save((detect_res, all_lm68), cache_file)
    try:
        shape = frames[0].shape[:2]
    except IndexError: # if there is no frame in the video, error list에 저장
        f = open("./indexerror.txt", 'a')
        f.write("{}\n".format(video_path))
        f.close()
        return
    
    # 모든 detect_res
    all_detect_res = []

    assert len(all_lm68) == len(detect_res)
    # in each frame, save the detected face's bounding box, landmark(5, 68), score as a tuple and save it in a list
    for faces, faces_lm68 in zip(detect_res, all_lm68):
            new_faces = []
            for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
                new_face = (box, lm5, face_lm68, score)
                new_faces.append(new_face)
            all_detect_res.append(new_faces)
    detect_res = all_detect_res    
    # SORT tracking
    # tracks : list, len = 사람 수로 예상, the number of detected face in a frame
    # tracks [] : list, len = 프레임 수, the number of frames
    # tracks [] [] : tuple, length = 4, 각각 box, lm5 : landmark (5,2) , lm68 : landmark (68,2), score
    tracks = multiple_tracking(detect_res)
    # tuples : list, len = 사람 수로 예상, the number of detected face in a frame
    # tuples [] : tuple, length = 2, 각각 0, 프레임 수 the number of frames
    tuples = [(0, len(detect_res))] * len(tracks)
    # if there is no face detected, find the longest face in the video
    if len(tracks) == 0:
        tuples, tracks = find_longest(detect_res)
    data_storage = {}
    frame_boxes = {}
    super_clips = []
    frame_res = {}
    super_clips_start_end = []
    # super_clips : tracking된 face들을 의미하는 것으로 보임
    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)): # each track(=face)
        
        # if detect_res's length is not equal to track's length, raise error 
        assert len(detect_res[start:end]) == len(track)
        
        super_clips.append(len(track))
        super_clips_start_end.append((start, end))
        for face, frame_idx, j in zip(track, range(start, end), range(len(track))): # frame에서 각각의 face
            box,lm5,lm68 = face[:3] # box, lm5, lm68
            big_box = get_crop_box(shape, box, scale=0.5) # get crop box

            top_left = big_box[:2][None, :] # top left point

            new_lm5 = lm5 - top_left 
            new_lm68 = lm68 - top_left

            new_box = (box.reshape(2, 2) - top_left).reshape(-1)

            info = (new_box, new_lm5, new_lm68, big_box) # face info


            x1, y1, x2, y2 = big_box
            cropped = frames[frame_idx][y1:y2, x1:x2]
            # cropped = cv2.resize(cropped, (512, 512))
            # face들을 tracking한 박스들로 crop함
            # landmark들도 box에 맞게 변환
            # data_storage에 저장 i는 face id, j는 frame을 의미
            base_key = f"{track_i}_{j}_" # i : face, j : frame
            data_storage[base_key + "img"] = cropped
            data_storage[base_key + "ldm"] = info
            data_storage[base_key + "idx"] = frame_idx
            frame_boxes[frame_idx] = np.rint(box).astype(np.int64)
    # 총 crop된 face들과 그 face들의 frame 수를 알려줌
    logger.info(f"{crop_path}  :  sampling clips from super clips {super_clips}")
    clips_for_video = []
    clip_size = clip_size
    pad_length = clip_size - 1
    
    # 각 face id 별로 clip을 만듦
    # 아래의 영어 표기로는 8clip을 의미하지만 정확하겐 clip size 만큼 함
    for super_clip_idx, super_clip_size in enumerate(super_clips): # cut the super clip into clips, overlap 7frames, 8frames per clip 
        inner_index = list(range(super_clip_size))
        
        if super_clip_size < clip_size: # if there is not enough frames to make a clip, pad the frames   
            # to do : how to operate the padding
            # 정확하게 이 코드가 어떻게 동작하는지 모르겠지만 
            # 대략적으로 frame들을 clipsize로 나눌때 부족하면 
            # clip size만큼의 frame이 되도록 padding을 함
            if super_clip_size < clip_size//2 : continue 
            post_module = inner_index[1:-1][::-1] + inner_index

            l_post = len(post_module)
            post_module = post_module * (pad_length // l_post + 1)
            post_module = post_module[:pad_length]
            assert len(post_module) == pad_length

            pre_module = inner_index + inner_index[1:-1][::-1]
            l_pre = len(post_module)
            pre_module = pre_module * (pad_length // l_pre + 1)
            pre_module = pre_module[-pad_length:]
            assert len(pre_module) == pad_length

            inner_index = pre_module + inner_index + post_module

        super_clip_size = len(inner_index)

        frame_range = [
            inner_index[i : i + clip_size] for i in range(super_clip_size) if i + clip_size <= super_clip_size
        ]
        for indices in frame_range:
            clip = [(super_clip_idx, t) for t in indices]
            clips_for_video.append(clip)
            
    # landmarks, images = crop_align_func(landmarks, images) # i : face, j : frame
    for clip in clips_for_video: 
        # 각 자른 clip에 대해서 진행
        images = [data_storage[f"{i}_{j}_img"] for i, j in clip] # call cropped face images from data_storage, i : face, j : frame
        landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip] # call landmarks from data_storage, i : face, j : frame
        # landmark를 기준으로 crop align func을 진행 
        # 해당 함수가 clip에 있는 얼굴들의 landmark 평균을 기준으로 박스를 설정하고
        # 박스를 기준으로 crop align을 진행
        # 다르게 말하면, landmark 평균을 기준으로 박스의 geometry를 설정하고
        # 박스의 geometry는 고정한체로 얼굴이 움직이는 걸 찍었다 생각하면 됨
        # 다시 또 말하면, 카메라를 고정하고 사람이 움직이는 것을 찍은것처럼 
        # PPT 참조
        landmarks, images = crop_align_func(landmarks, images) # align the face images by landmarks in the clip
        i, j = clip[-1]
        k = super_clips[i]%clip_size 
        
        ##########################################################################
        # 코드 변경시 이 함수에서는 이부분만 변경할 것을 권고 !!!!!!!!!!!!!!!!!!!!!!!!
        # 특히, cv2.imwrite함수만 변경할 것을 추천
        ##########################################################################
        if (j+1)%clip_size==0: # if last frame number of the clip is multiple of clip_size, save all images in the clip
            # it means save face alignments in 8 frames in the video so that they don't overlap
            for f, (i,j) in enumerate(clip) : 
                cv2.imwrite(join(crop_path, f'{i:02}_{j:04}.png'), cv2.cvtColor(images[f], cv2.COLOR_BGR2RGB))   
        if j == super_clips[i]-1: # if the clip have last frame image, save all images in the clip
            if k!=0 : # if the clip is not multiple of clip_size, save the last k images in the clip
                # k is the number of frames that are not overlapped
                for l in range(clip_size-k,clip_size):
                    ci,cj = clip[l]
                    cv2.imwrite(join(crop_path, f'{ci:02}_{cj:04}.png'), cv2.cvtColor(images[l], cv2.COLOR_BGR2RGB)) # clip means face alignment images in 8 frames, non-overlap
 
# 각 manipulation method 별로 video를 불러오고 crop_face_from_video 함수를 실행하는 함수
def extract_method_videos(data_path,cache_path,save_path,clip_size):
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    # video path, crop image 저장 위치 설정
    # videos_path = join(data_path,  type_path)
    # caches_path = join(cache_path,type_path)
    # crop_images_path = join(save_path,type_path,'crop_images')
    # os.makedirs(caches_path, exist_ok=True)
    # logger.info("videos_path : {}".format(videos_path))
    
    import glob
    # under the data_path, there are many videos in the subfolders
    import glob
    # under the data_path, there are many videos in the subfolders
    videos_path = glob.glob(join(data_path,'**/*.mp4'), recursive=True)
    type_path = data_path.replace('/workspace/datasets/deepfakes/DFD_video/','')
    save_path = os.path.join(save_path,type_path.replace('videos','crop_images'))
    print(f"type_path : {type_path}")
    print(f"save_path : {save_path}")
    for video_path in tqdm(videos_path):
        hierarchy = video_path.split('/')[-1].split('.')[0]
        cache_img_path = os.path.join(cache_path,type_path,hierarchy)
        os.makedirs(os.path.dirname(cache_img_path), exist_ok=True)
        crop_img_path = os.path.join(save_path,hierarchy)
        os.makedirs(crop_img_path, exist_ok=True)
        crop_face_from_video(video_path,cache_img_path,crop_img_path,clip_size)
        # break #inferece 1개



if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--data_path','-p', type=str, default='/workspace/datasets/deepfakes/DFD_video/DeepFakeDetection/raw/videos')
    p.add_argument('--save_path','-s', type=str, default='/workspace/datasets/deepfakes/DFD')
    p.add_argument('--cachepath', '-c', type=str, 
                   default='/workspace/FTCN_preprocessing/caches')
    p.add_argument('--clipsize','-l',type=int,default=32)
    args = p.parse_args()
    data_path = args.data_path
    cache_path = args.cachepath
    clip_size = args.clipsize
    save_path = args.save_path
    
    extract_method_videos(data_path,cache_path,save_path,clip_size)
    
###################### reference ###################### 

# this code reference from FTCN Official Code in git hub
# link is  https://github.com/yinglinzheng/FTCN

# - Zheng, Y., Bao, J., Chen, D., Zeng, M., & Wen, F. (2021). Exploring Temporal Coherence for More General Video Face Forgery Detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 15044–15054).
