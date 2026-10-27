import os
import argparse
from config.base_config import Config
from modules.basic_utils import mkdirp, deletedir


class AllConfig(Config):
    def __init__(self):
        super().__init__()

    def parse_args(self):
        description = 'Text-to-Video Retrieval'
        parser = argparse.ArgumentParser(description=description)
        
        parser.add_argument('--dataset_name', type=str, default='MSRVTT', help="Dataset name")
        parser.add_argument('--videos_dir', type=str, default='data/MSRVTT/vids', help="Location of videos")
        parser.add_argument('--msrvtt_train_file', type=str, default='9k')
        parser.add_argument('--num_frames', type=int, default=12)
        parser.add_argument('--video_sample_type', default='uniform', help="'rand'/'uniform'")
        parser.add_argument('--input_res', type=int, default=224)

        parser.add_argument('--exp_name', type=str, required=True, help="Name of the current experiment")
        parser.add_argument('--output_dir', type=str, default='./outputs')
        parser.add_argument('--save_every', type=int, default=1, help="Save model every n epochs")
        parser.add_argument('--log_step', type=int, default=10, help="Print training log every n steps")
        parser.add_argument('--evals_per_epoch', type=int, default=10, help="Number of times to evaluate per epoch")
        parser.add_argument('--load_epoch', type=int, help="Epoch to load from exp_name, or -1 to load model_best.pth")
        parser.add_argument('--eval_window_size', type=int, default=5, help="Size of window to average metrics")
        parser.add_argument('--metric', type=str, default='t2v', help="'t2v'/'v2t'")

        parser.add_argument('--huggingface', action='store_true', default=False)
        parser.add_argument('--arch', type=str, default='clip_transformer')
        parser.add_argument('--clip_arch', type=str, default='ViT-B/32', help="CLIP arch. only when not using huggingface")
        parser.add_argument('--embed_dim', type=int, default=512, help="Dimensionality of the model embedding")

        parser.add_argument('--loss', type=str, default='clip')
        parser.add_argument('--clip_lr', type=float, default=1e-6, help='Learning rate used for CLIP params')
        parser.add_argument('--noclip_lr', type=float, default=1e-5, help='Learning rate used for new params')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_epochs', type=int, default=5)
        parser.add_argument('--weight_decay', type=float, default=0.2, help='Weight decay')
        parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Warmup proportion for learning rate schedule')

        parser.add_argument('--pooling_type', type=str)
        parser.add_argument('--k', type=int, default=-1, help='K value for topk pooling')
        parser.add_argument('--attention_temperature', type=float, default=0.01, help='Temperature for softmax (used in attention pooling only)')
        parser.add_argument('--num_mha_heads', type=int, default=1, help='Number of parallel heads in multi-headed attention')
        parser.add_argument('--transformer_dropout', type=float, default=0.3, help='Dropout prob. in the transformer pooling')

        parser.add_argument('--motion_temperature', type=float, default=0.1, help='Temperature for motion gated pooling (tau_m)')
        parser.add_argument('--flow_score_path', type=str, default='data/MSRVTT/msrvtt_flow_scores_robust.json', help='Path to precomputed optical flow scores')
        parser.add_argument('--short_caption_path', type=str, default='data/MSRVTT/msrvtt_short_captions.json', help='Path to precomputed short captions')
        parser.add_argument('--vatex_data_dir', type=str, default='data/VATEX', help='Directory for VATEX annotations')
        parser.add_argument('--vatex_manifest_path', type=str, default='data/VATEX/manifest.json', help='Path to VATEX manifest')
        parser.add_argument('--vatex_train_list', type=str, default='data/VATEX/train_list.txt', help='Path to VATEX train split list')
        parser.add_argument('--vatex_val_list', type=str, default='data/VATEX/val_list.txt', help='Path to VATEX val split list')
        parser.add_argument('--vatex_test_list', type=str, default='data/VATEX/test_list.txt', help='Path to VATEX test split list')
        parser.add_argument('--allow_missing_preprocessed', action='store_true', default=False)
        parser.add_argument('--disable_mgtr', action='store_true', default=False)

        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--seed', type=int, default=24, help='Random seed')
        parser.add_argument('--no_tensorboard', action='store_true', default=False)
        parser.add_argument('--tb_log_dir', type=str, default='logs')

        args = parser.parse_args()

        args.model_path = os.path.join(args.output_dir, args.exp_name)
        args.tb_log_dir = os.path.join(args.tb_log_dir, args.exp_name)

        mkdirp(args.model_path)
        deletedir(args.tb_log_dir)
        mkdirp(args.tb_log_dir)

        return args