import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import pickle
from PIL import Image

import random
import numpy as np


def get_diff(array):
    array_ = array[1:, :]
    _array = array[:-1, :]
    return array_ - _array

class TripletLoader(Dataset):
    def __init__(self, triplets, dtype='total'):
        self.triplets = triplets
        self.dtype = dtype
        
    def __getitem__(self, index):
        clip1_pth, clip2_pth, clip3_pth, labels = self.triplets[index]
        
        with open(clip1_pth, 'rb') as file:
            latents1 = self.cutlatents(pickle.load(file))
            clip1 = latents1
        with open(clip2_pth, 'rb') as file:
            latents2 = self.cutlatents(pickle.load(file))
            clip2 = latents2
        with open(clip3_pth, 'rb') as file:
            latents3 = self.cutlatents(pickle.load(file))
            clip3 = latents3
        return clip1, clip2, clip3, labels
    
    def __len__(self):
        return len(self.triplets)
    
    def cutlatents(self, latents):
        if self.dtype == 'total':
            return latents
        elif self.dtype == 'coarse':
            return latents[:, :3*512]
        elif self.dtype == 'middle':
            return latents[:, 3*512:7*512]
        elif self.dtype == 'fine':
            return latents[:, 7*512:]
        elif self.dtype == 'middle+':
            return latents[:, 3*512:]
        elif self.dtype == 'coarse+':
            return latents[:, :7*512]
        else:
            raise Exception('invalid dtype!')



class BaseLoader(Dataset):
    def __init__(self, testset):
        self.testset = testset
    
    def __getitem__(self, index):
        clip_path, label, video_idx = self.testset[index]
        
        with open(clip_path, 'rb') as file:
            clip = get_diff(pickle.load(file))
        
        return clip, label, video_idx
    
    def __len__(self):
        return len(self.testset)
        


def get_loader(dset_obj, args):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    
    if dset_obj.split == 'train':
        loader = TripletLoader
        _triplets = []
        
        ### supcon ###
        for _ in range(args.num_train_samples):
            pos_anchor_clip, pos_clip, neg_clip, labels = dset_obj.get_Triplet()
            _triplets.append([pos_anchor_clip, pos_clip, neg_clip, labels])
                
        _data_loader = DataLoader(
            loader(_triplets, args.dtype),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_works, drop_last=True
        )
    
    
    elif dset_obj.split == 'val':
        loader = TripletLoader
        _triplets = []
        
        for _ in range(args.num_test_samples):  
            pos_anchor_clip, pos_clip, neg_clip, labels = dset_obj.get_Triplet()
            _triplets.append([pos_anchor_clip, pos_clip, neg_clip, labels])

        _data_loader = DataLoader(
            loader(_triplets, args.dtype),
            batch_size=args.batch_size, shuffle=True, **kwargs
        )
        
    elif dset_obj.split == 'test':
        loader = BaseLoader
        _testset = []
        for i in range(len(dset_obj)):
            clip_path, label, video_idx = dset_obj.get_clip(i)
            _testset.append([clip_path, label, video_idx])
        
        _data_loader = DataLoader(
            loader(_testset),
            batch_size=args.batch_size, shuffle=False, **kwargs
        )
        
    else:
        raise Exception("dset_obj.split must in [train, val, test]")
    
    return _data_loader
