import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from PIL import Image
import bisect


#######################
### for Triplet GRU ###
#######################

def get_files_from_split(split):
    """ "
    Get filenames for real and fake samples

    Parameters
    ----------
    split : pandas.DataFrame
        DataFrame containing filenames
    """
    files_1 = split[0].astype(str).str.cat(split[1].astype(str), sep="_")
    files_2 = split[1].astype(str).str.cat(split[0].astype(str), sep="_")
    files_real = pd.concat([split[0].astype(str), split[1].astype(str)]).to_list()
    files_fake = pd.concat([files_1, files_2]).to_list()
    return files_real, files_fake

########################################################################################

ffpproot='your/dataset/path'

######################
### for evaluation ###
######################
class FF_Dataset_eval(Dataset):
    def __init__(
        self,
        split='train',
        dtype='total',
        ds_types=['Origin', 'Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures'],
    ):
        ### FIXME ### 
        self.ffpproot = os.path.join(ffpproot, 'clipc23')
        
        self.split = split
        self.ds_types = ds_types
        self.__dataset_set = {0: [], 1: []}
        
        self.__dataset_keys = [0, 1]
        
        self.videos_per_type = {}
        
        self.paths = []
        
        self.clips_per_video = []
        
    
        if self.split == 'train':
            train_split = pd.read_json('/path/to/split/metadata/train.json', dtype=False)
            self.files_real, self.files_fake = get_files_from_split(train_split)
        elif self.split == 'val':
            val_split = pd.read_json('/path/to/split/metadata/val.json', dtype=False)
            self.files_real, self.files_fake = get_files_from_split(val_split)
        elif self.split == 'test':
            test_split = pd.read_json('/path/to/split/metadata/test.json', dtype=False)
            self.files_real, self.files_fake = get_files_from_split(test_split)
        else:
            raise Exception("invalid dataset error!")
        
    
        for ds_type in self.ds_types:
            video_paths = os.path.join(self.ffpproot, ds_type)
            
            if ds_type == 'Origin':
                videos = sorted(self.files_real)
            else:
                videos = sorted(self.files_fake)
            
            self.videos_per_type[ds_type] = len(videos)
            
            for video in videos:
                video_path = os.path.join(video_paths, video)
                
                clip_list = os.listdir(video_path)
                
                self.paths.append([video_path, ds_type])
                
                # Only use in test 
                self.clips_per_video.append(len(clip_list))
                
                for clip in clip_list:
                    path = os.path.join(video_path, clip)
                
                    if ds_type == 'Origin':
                        self.__dataset_set[0].append(path)
                    else:
                        self.__dataset_set[1].append(path)
        
        clip_lengths = torch.as_tensor(self.clips_per_video)
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()
    
    def __len__(self):
        return self.cumulative_sizes[-1]
        
    def get_Triplet(self):
        return

    
    def get_clip(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
            
        item = self.paths[video_idx]
        
        path = item[0]
        label = 0 if item[1] == 'Origin' else 1
        
        clip_path = os.path.join(path, f"clipnum_{clip_idx}.p")
        
        return clip_path, label, video_idx

########################################
### for train Supervised contrastive ###
########################################
class FF_Dataset_con(Dataset):
    def __init__(
        self,
        split='train',
        dtype='style',
        ds_types=['Origin', 'Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures'],
    ):
        ### FIXME ### 
        self.ffpproot = os.path.join(ffpproot, 'clipc23')
        self.ffpproot_r = os.path.join(ffpproot, 'clipc23_r')
        
        self.split = split
        self.ds_types = ds_types
        self.__dataset_set = {0: [], 1: []}
        self.__dataset_set_r = {0: [], 1: []}
        
        self.__dataset_keys = [0, 1]
        
        self.videos_per_type = {}
        
        self.paths = []
        self.paths_r = []
        
        self.clips_per_video = []
        
    
        if self.split == 'train':
            train_split = pd.read_json('/path/to/split/metadata/train.json', dtype=False)
            self.files_real, self.files_fake = get_files_from_split(train_split)
        elif self.split == 'val':
            val_split = pd.read_json('/path/to/split/metadata/val.json', dtype=False)
            self.files_real, self.files_fake = get_files_from_split(val_split)
        elif self.split == 'test':
            test_split = pd.read_json('/path/to/split/metadata/test.json', dtype=False)
            self.files_real, self.files_fake = get_files_from_split(test_split)
        else:
            raise Exception("invalid dataset error!")
        
    
        for ds_type in self.ds_types:
            video_paths = os.path.join(self.ffpproot, ds_type)
            video_paths_r = os.path.join(self.ffpproot_r, ds_type)
            
            if ds_type == 'Origin':
                videos = sorted(self.files_real)
            else:
                videos = sorted(self.files_fake)
            
            self.videos_per_type[ds_type] = len(videos)
            
            for video in videos:
                video_path = os.path.join(video_paths, video)
                video_path_r = os.path.join(video_paths_r, video)
                
                clip_list = os.listdir(video_path)
                clip_list_r = os.listdir(video_path_r)
                
                self.paths.append([video_path, ds_type])
                self.paths_r.append([video_path_r, ds_type])
                
                # Only use in test 
                self.clips_per_video.append(len(clip_list))
                
                for clip, clip_r in zip(clip_list, clip_list_r):
                    path = os.path.join(video_path, clip)
                    path_r = os.path.join(video_path_r, clip_r)
                    if ds_type == 'Origin':
                        self.__dataset_set[0].append(path)
                        self.__dataset_set_r[0].append(path_r)
                    else:
                        self.__dataset_set[1].append(path)
                        self.__dataset_set_r[1].append(path_r)
        
        clip_lengths = torch.as_tensor(self.clips_per_video)
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()
    
    def __len__(self):
        return self.cumulative_sizes[-1]
        
    def get_Triplet(self):
        dataset = self.__dataset_set
        dataset_r = self.__dataset_set_r
        keys = self.__dataset_keys
        
        pos_idx = 0
        neg_idx = 0
        pos_anchor_clip_idx = 0
        pos_clip_idx = 0
        neg_clip_idx = 0
        
        pos_idx = random.choice(keys)
        neg_idx = 0 if pos_idx == 1 else 1
        
        pos_anchor_clip_idx = random.randint(0, len(dataset[pos_idx]) - 1)
        pos_clip_idx = random.randint(0, len(dataset_r[pos_idx]) - 1)
        neg_clip_idx = random.randint(0, len(dataset[neg_idx]) - 1)
        neg_clip_idx_r = random.randint(0, len(dataset_r[neg_idx]) -1)
        
        pos_anchor_clip = dataset[pos_idx][pos_anchor_clip_idx]
        pos_clip = dataset_r[pos_idx][pos_clip_idx]
        
        neg_type = random.choice(keys)
        if neg_type == 0:
            neg_clip = dataset[neg_idx][neg_clip_idx]
        elif neg_type == 1:
            neg_clip = dataset_r[neg_idx][neg_clip_idx_r]
        else:
            raise Exception("unexpected neg type!")
        
        labels = [pos_idx, pos_idx, neg_idx]
        
        return pos_anchor_clip, pos_clip, neg_clip, labels
    
    def get_clip(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
            
        item = self.paths[video_idx]
        
        path = item[0]
        label = 0 if item[1] == 'Origin' else 1
        
        clip_path = os.path.join(path, f"clipnum_{clip_idx}.p")
        
        return clip_path, label, video_idx

