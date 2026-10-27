import torch.nn as nn
import torch
import torch.nn.functional as F
from config.base_config import Config
import torch.distributed as dist

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims, logit_scale):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """
        logit_scale = logit_scale.exp()
        logits = sims * logit_scale
        
        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0


class SALSLoss(CLIPLoss):
    def __init__(self, cls_loss_weight=0.1):
        super().__init__()
        self.cls_loss_weight = cls_loss_weight

    def reweight_targets(self, cap_fq, num_samples, targets, pos_weights=None):
        cap_fq += targets.sum(dim=0, keepdim=True) / targets.shape[0]
        num_samples += 1

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cap_fq, op=dist.ReduceOp.AVG)
            dist.all_reduce(num_samples, op=dist.ReduceOp.AVG)
            
        world_size = 1 
        if dist.is_available() and dist.is_initialized():
             world_size = dist.get_world_size()

        all_batch_size = world_size * targets.shape[0]
        weight_factor = torch.log((num_samples + 1.0/all_batch_size) / (cap_fq + 1.0/all_batch_size))
        
        if pos_weights is not None:
            pos_weights = pos_weights.to(dtype=weight_factor.dtype, device=weight_factor.device)
            weight_factor = weight_factor * pos_weights

        targets = targets * weight_factor.to(dtype=targets.dtype)
        
        return targets

    def calc_class_loss(self, logits, targets):
        norm_targets = F.normalize(targets, p=1, dim=1)
        loss = -(F.log_softmax(logits, dim=1) * norm_targets).sum(dim=1).mean()
        return loss

    def forward(self, sims, logit_scale, 
                cls_logits=None, text_labels=None, 
                cap_fq=None, num_samples=None,
                pos_weights=None):
        clip_loss = super().forward(sims, logit_scale)
        
        if cls_logits is not None and text_labels is not None:
            B, Vocab_Size = cls_logits.shape
            targets = torch.zeros(B, Vocab_Size, dtype=torch.float32).to(cls_logits.device)
            targets.scatter_(dim=1, index=text_labels, value=1.0)
            
            if cap_fq is not None and num_samples is not None:
                targets = self.reweight_targets(cap_fq, num_samples, targets, pos_weights=pos_weights)
            
            class_loss = self.calc_class_loss(cls_logits, targets)
            return clip_loss + self.cls_loss_weight * class_loss
        
        return clip_loss


class LossFactory:
    @staticmethod
    def get_loss(config: Config):
        if config.loss == 'clip':
            return SALSLoss() 
        else:
            raise NotImplemented
