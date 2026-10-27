import torch
import torch.nn as nn
from config.base_config import Config

class MGTR(nn.Module):
    """
    MGTR (Motion-Guided Temporal Refinement):
    A lightweight, plug-and-play module that refines video features using:
    1. Temporal Adapter (Conv1D)
    2. Motion Enhancement (Flow-guided)
    3. Adaptive Gating (Short-text guided)
    
    It outputs a single fused feature sequence to be used by any downstream pooling method (e.g., XPool).
    """
    def __init__(self, config: Config):
        super(MGTR, self).__init__()
        
        self.embed_dim = getattr(config, 'embed_dim', 512)
        
        # 1. Temporal Adapter
        self.dw_conv = nn.Conv1d(self.embed_dim, self.embed_dim, 3, padding=1, groups=self.embed_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(self.embed_dim)

        # 2. Motion Enhancement
        self.motion_proj = nn.Linear(1, self.embed_dim)
        self.motion_sigmoid = nn.Sigmoid()
        
        # 3. Adaptive Gating
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'dw_conv' in name and 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'motion_proj' in name or 'gate_mlp' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, video_embeds, short_text_embeds, flow_scores):
        """
        Args:
            video_embeds: [B_v, T, D]
            short_text_embeds: [B_t, D] (Used for gating)
            flow_scores: [B_v, T]
            
        Output:
            video_fused: [B_v, T, D] (Refined feature sequence ready for interaction)
            alpha_val: [B_t, 1] (Monitoring value)
        """
        
        # --- A. Context Branch (Temporal Adapter) ---
        residual = video_embeds
        x = video_embeds.transpose(1, 2) 
        x = self.dw_conv(x)
        x = x.transpose(1, 2)
        feat_context = self.norm(residual + self.act(x)) 
        
        # --- B. Motion Branch (Flow Enhancement) ---
        if flow_scores is not None:
            flow_input = flow_scores.unsqueeze(-1)
            motion_gate = self.motion_sigmoid(self.motion_proj(flow_input))
            # Inject motion cues: Enhance features where flow is high
            feat_motion = feat_context + (feat_context * motion_gate)
        else:
            feat_motion = feat_context

        # --- C. Adaptive Gating (Early Fusion) ---
        if short_text_embeds is not None:
            alpha_val = self.gate_mlp(short_text_embeds) # [B_t, 1]
            alpha = alpha_val.unsqueeze(1) # [B, 1, 1] to match [B, T, D]
        else:
            alpha_val = torch.tensor(0.5, device=video_embeds.device)
            alpha = 0.5 
            
        video_fused = alpha * feat_motion + (1 - alpha) * feat_context
        
        return video_fused, alpha_val