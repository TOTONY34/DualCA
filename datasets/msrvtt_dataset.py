import os
import cv2
import sys
import torch
import random
import itertools
import numpy as np
import pandas as pd
import ujson as json
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture

from modules.tokenization_clip import tokenize

class MSRVTTDataset(Dataset):
    def __init__(self, config: Config, split_type = 'train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        self.db_file = 'data/MSRVTT/MSRVTT_data.json'
        self.test_csv = 'data/MSRVTT/MSRVTT_JSFUSION_test.csv'

        self.flow_score_path = getattr(config, "flow_score_path", "data/MSRVTT/msrvtt_flow_scores_robust.json")
        self.short_caption_path = getattr(config, "short_caption_path", "data/MSRVTT/msrvtt_short_captions.json")
        
        self.flow_scores = {}
        if os.path.exists(self.flow_score_path):
            print(f"Loading flow scores from {self.flow_score_path}...")
            self.flow_scores = load_json(self.flow_score_path)
        else:
            print(f"Warning: {self.flow_score_path} not found. Flow scores will be zeros.")

        self.short_captions = {}
        if os.path.exists(self.short_caption_path):
            print(f"Loading short captions from {self.short_caption_path}...")
            self.short_captions = load_json(self.short_caption_path)
        else:
            print(f"Warning: {self.short_caption_path} not found.")
        
        if config.msrvtt_train_file == '7k':
            train_csv = 'data/MSRVTT/MSRVTT_train.7k.csv'
        else:
            train_csv = 'data/MSRVTT/MSRVTT_train.9k.csv'

        self.db = load_json(self.db_file)
        if split_type == 'train':
            train_df = pd.read_csv(train_csv)
            self.train_vids = train_df['video_id'].unique()
            self._compute_vid2caption()
            self._construct_all_train_pairs()
        else:
            self.test_df = pd.read_csv(self.test_csv)

            
    def __getitem__(self, index):
        video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
        
        imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                         self.config.num_frames, 
                                                         self.config.video_sample_type)

        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)
        cur_flow_scores = np.zeros(self.config.num_frames, dtype=np.float32)
        if video_id in self.flow_scores:
            scores = self.flow_scores[video_id]
            if len(scores) >= self.config.num_frames:
                cur_flow_scores = np.array(scores[:self.config.num_frames], dtype=np.float32)
            else:
                cur_flow_scores[:len(scores)] = np.array(scores, dtype=np.float32)
        
        short_text_str = caption 
        if self.split_type == 'train' and video_id in self.short_captions:
            candidates = self.short_captions[video_id]
            if len(candidates) > 0:
                short_text_str = random.choice(candidates)
                
                short_text_str = short_text_str.strip('"').strip("'").strip()
                if "Subject:" in short_text_str:
                    short_text_str = short_text_str.replace("Subject:", "").replace("Verb:", "").replace("Object:", "")
                short_text_str = " ".join(short_text_str.split())

        short_text_tensor = tokenize(short_text_str, context_length=77, truncate=True).squeeze(0)

        return {
            'video_id': video_id,
            'video': imgs,
            'text': caption,
            'short_text': short_text_tensor, 
            'flow_scores': torch.from_numpy(cur_flow_scores),
        }

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.test_df)

    def _get_vidpath_and_caption_by_index(self, index):
        if self.split_type == 'train':
            vid, caption = self.all_train_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
        else:
            vid = self.test_df.iloc[index].video_id
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            caption = self.test_df.iloc[index].sentence
        return video_path, caption, vid
    
    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        if self.split_type == 'train':
            for vid in self.train_vids:
                for caption in self.vid2caption[vid]:
                    self.all_train_pairs.append([vid, caption])
            
    def _compute_vid2caption(self):
        self.vid2caption = defaultdict(list)
        for annotation in self.db['sentences']:
            caption = annotation['caption']
            vid = annotation['video_id']
            self.vid2caption[vid].append(caption)