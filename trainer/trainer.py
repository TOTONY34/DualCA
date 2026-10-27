from config.base_config import Config
import numpy as np
import torch
import os
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id
from tqdm import tqdm

try:
    from modules.tokenization_clip import tokenize
except ImportError:
    try:
        import clip
        tokenize = clip.tokenize
    except ImportError:
        tokenize = None

class Trainer(BaseTrainer):
    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader, 
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None):
        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer 
        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0
        self.vocab_size = 49408
        self.cap_fq = torch.zeros([1, self.vocab_size], dtype=torch.float64)
        self.num_samples = torch.zeros([1, 1], dtype=torch.float64)
        self.pos_weight_path = "data/MSRVTT/pos_weights_new_light1.pt"
        self.pos_weights = None
        if os.path.exists(self.pos_weight_path):
            print(f"Loading POS weights from {self.pos_weight_path}...")
            self.pos_weights = torch.load(self.pos_weight_path, map_location='cpu')
        else:
            print(f"Warning: POS weights not found at {self.pos_weight_path}")

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]
        if self.cap_fq.device != self.device:
            self.cap_fq = self.cap_fq.to(self.device)
            self.num_samples = self.num_samples.to(self.device)
        if self.pos_weights is not None and self.pos_weights.device != self.device:
            self.pos_weights = self.pos_weights.to(self.device)

        for batch_idx, data in enumerate(self.train_data_loader):
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
            
            data['video'] = data['video'].to(self.device)
            if 'flow_scores' in data:
                data['flow_scores'] = data['flow_scores'].to(self.device)
            if 'idf_scores' in data:
                data['idf_scores'] = data['idf_scores'].to(self.device)
            res = self.model(data)
            if len(res) == 4:
                 text_embeds, video_embeds_pooled, alpha_val, cls_logits = res
            else:
                 text_embeds, video_embeds_pooled, alpha_val = res
                 cls_logits = None
            
            output = sim_matrix_training(text_embeds, video_embeds_pooled, self.pooling_type)
            text_labels = data['text']
            if isinstance(text_labels, dict):
                text_labels = text_labels['input_ids']

            loss = self.loss(output, self.model.clip.logit_scale,
                             cls_logits=cls_logits,
                             text_labels=text_labels,
                             cap_fq=self.cap_fq,
                             num_samples=self.num_samples,
                             pos_weights=self.pos_weights)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

            self.global_step += 1
            if self.writer is not None:
                self.writer.add_scalar('train/loss_train', loss.detach().item(), self.global_step)
                if isinstance(alpha_val, torch.Tensor):
                    self.writer.add_scalar('train/alpha_mean', alpha_val.mean().item(), self.global_step)

            total_loss += loss.detach().item()

            if batch_idx % self.log_step == 0:
                alpha_str = f"{alpha_val.mean().item():.4f}" if isinstance(alpha_val, torch.Tensor) else str(alpha_val)
                print('Train Epoch: {} dl: {}/{} Loss: {:.6f} Alpha: {}'.format(
                    epoch, batch_idx, num_steps-1, loss.detach().item(), alpha_str))

            if batch_idx in eval_steps:
                val_res = self._valid_epoch_step(epoch, batch_idx, num_steps-1)
                self.model.train()
                if val_res['R1-window'] > self.best_window:
                    self.best_window = val_res['R1-window']
                    self._save_checkpoint(epoch, save_best=True)
                if val_res['R1'] > self.best:
                    self.best = val_res['R1']
                print(" Current Best Window Average R@1 is {}".format(self.best_window))
                print(" Current Best R@1 is {}\n\n".format(self.best))

        return {'loss_train':  total_loss / num_steps}

    def _valid_epoch_step(self, epoch, step, num_steps):
        self.model.eval()
        total_val_loss = 0.0
        text_embed_arr = []
        vid_embed_arr = []
        short_text_embed_arr = []
        flow_score_arr = []
        all_vid_ids = []
        
        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.device)
                else:
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

                data['video'] = data['video'].to(self.device)
                if 'flow_scores' in data:
                    data['flow_scores'] = data['flow_scores'].to(self.device)
                    flow_score_arr.append(data['flow_scores'].cpu())
                if 'idf_scores' in data:
                    data['idf_scores'] = data['idf_scores'].to(self.device)
                res = self.model(data, return_all_frames=True)
                if len(res) == 6:
                    text_embed, vid_embed, _, short_text_embed, _, _ = res
                elif len(res) == 5:
                    text_embed, vid_embed, _, short_text_embed, _ = res
                elif len(res) == 4:
                    text_embed, vid_embed, _, short_text_embed = res
                else:
                    text_embed, vid_embed, _ = res
                    short_text_embed = text_embed 

                text_embed_arr.append(text_embed.cpu())
                vid_embed_arr.append(vid_embed.cpu())
                short_text_embed_arr.append(short_text_embed.cpu())
                # 1. Feature Fusion
                video_fused, _ = self.model.mgtr(vid_embed, short_text_embed, data.get('flow_scores'))
                # 2. XPool Interaction
                batch_pooled = self.model.xpool_transformer(text_embed, video_fused)
                
                sims_batch = sim_matrix_training(text_embed, batch_pooled, self.pooling_type)
                curr_loss = self.loss(sims_batch, self.model.clip.logit_scale)
                total_val_loss += curr_loss.item()

                for v_id in data['video_id']:
                    all_vid_ids.append(v_id)
            text_embeds = torch.cat(text_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)
            short_text_embeds = torch.cat(short_text_embed_arr)
            
            if len(flow_score_arr) > 0:
                all_flow_scores = torch.cat(flow_score_arr)
            else:
                all_flow_scores = torch.zeros(vid_embeds.shape[0], vid_embeds.shape[1])

            vid_embeds_per_video_id = {}
            flow_scores_per_video_id = {}
            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id:
                    vid_embeds_per_video_id[v_id] = vid_embeds[idx]
                    flow_scores_per_video_id[v_id] = all_flow_scores[idx]
            
            vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])
            flow_scores = torch.stack([flow_scores_per_video_id[v_id] for v_id in vid_embeds_per_video_id])
            self.model.mgtr.cpu()
            self.model.xpool_transformer.cpu()

            # Feature Fusion (MGTR) on CPU
            video_fused_all, _ = self.model.mgtr(vid_embeds, short_text_embeds, flow_scores)
            # X-Pool Interaction on CPU
            vid_embeds_pooled = self.model.xpool_transformer(text_embeds, video_fused_all)
            
            self.model.mgtr.to(self.device)
            self.model.xpool_transformer.to(self.device)
            
            # Metric Calculation (On CPU)
            vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)
            text_embeds_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            text_embeds_norm = text_embeds_norm.unsqueeze(0) 
            sims = (vid_embeds_pooled * text_embeds_norm).sum(dim=-1)
            sims = sims.t().cpu()

            total_val_loss = total_val_loss / len(self.valid_data_loader)
            metrics = self.metrics
            res = metrics(sims)
            
            for m in res:
                self.window_metric[m].append(res[m])
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            print(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                  f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                  f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                  f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                  f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                  f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
                  f"Loss: {total_val_loss}")
            
            res['loss_val'] =  total_val_loss
            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f'val/{m}', res[m], self.global_step)

            return res