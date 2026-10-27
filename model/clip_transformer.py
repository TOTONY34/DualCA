import torch
import torch.nn as nn
from config.base_config import Config
from modules.mgtr import MGTR 
from modules.transformer import Transformer

class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config
        
        if self.config.huggingface:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch)

        self.mgtr = MGTR(config)
        self.xpool_transformer = Transformer(config)     
        self.vocab_size = 49408 
        self.text_decoder = nn.Linear(config.embed_dim, self.vocab_size)
        nn.init.normal_(self.text_decoder.weight, std=0.02)
        nn.init.constant_(self.text_decoder.bias, 0)

    def forward(self, data, return_all_frames=False):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        flow_scores = data.get('flow_scores')
        short_text_tokens = data.get('short_text') 

        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        
        if self.config.huggingface:
            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)            
            short_text_features = text_features 
            text_features = text_features.unsqueeze(1) 
        else:
            text_out = self.clip.encode_text(text_data)
            
            if isinstance(text_out, tuple):
                text_seq, text_global = text_out
                text_features = text_global
                
                short_text_features = text_global
            else:
                text_features = text_out
                short_text_features = text_features

            video_features = self.clip.encode_image(video_data)
            
            if short_text_tokens is not None:
                if short_text_tokens.device != video_data.device:
                    short_text_tokens = short_text_tokens.to(video_data.device)
                
                short_out = self.clip.encode_text(short_text_tokens)
                if isinstance(short_out, tuple):
                    short_text_features = short_out[1] 
                else:
                    short_text_features = short_out
            
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        if flow_scores is not None:
            flow_scores = flow_scores.to(video_data.device)
        else:
            flow_scores = torch.zeros(batch_size, self.config.num_frames).to(video_data.device)

        video_fused, alpha = self.mgtr(video_features, short_text_features, flow_scores)
        video_mean = video_features.mean(dim=1)
        
        logits = self.text_decoder(video_mean)

        video_features_pooled = self.xpool_transformer(text_features, video_fused)
            
        if return_all_frames:
            return text_features, video_features, video_features_pooled, short_text_features, alpha, logits

        return text_features, video_features_pooled, alpha, logits