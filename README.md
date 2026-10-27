<div align="center">
<h1>
<b>
Dual-Cue Alignment for Text-Video Retrieval
</b>
</h1>
<h4>
<b>
Tuo Tang, Ning Han
</b>
</h4>
</div>

## Introduction
This repository contains the official PyTorch implementation of our **MMM 2027** paper **"Dual-Cue Alignment for Text-Video Retrieval"**.

**DualCA** introduces an evidence-preserving stage prior to query-conditioned video pooling (**X-Pool**):
- **Motion-Guided Temporal Refinement (MGTR)**: Integrates lightweight depthwise temporal convolution with offline RAFT-derived optical flow scores and query-adaptive residual gating to expose frame-to-frame state transitions while preserving pretrained CLIP appearance semantics.
- **Saliency-Aware Lexical Supervision (SALS)**: Guides the visual encoder during training via a visual-to-vocabulary objective weighted by online batch-level token rarity (IDF) and part-of-speech (POS) priors, and is completely removed at inference.

## Dependencies
Our model is developed and evaluated under **Python 3.8** with the following core dependencies:
- PyTorch 1.8.1
- Torchvision 0.9.1
- Transformers 4.6.1 (up to 4.30.2)
- OpenCV 4.5.3

Install all required packages via:
```bash
pip install -r requirements.txt
```

## Datasets and Precomputed Cues
We evaluate DualCA on four text-video retrieval benchmarks: **MSR-VTT** (9K train / 1K-A test), **MSVD**, **LSMDC**, and **VATEX**.
- For raw video downloading and standard split preparation on MSR-VTT, MSVD, and LSMDC, please refer to [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) and [X-Pool](https://github.com/layer6ai-labs/xpool).
- For **VATEX**, English captions from the official v1.0 annotations are matched to available video files via the dataset manifest (`datasets/vatex_dataset.py`).

### Precomputed Auxiliary Files (`data/MSRVTT/`)
To support training and evaluation without running dense optical flow extraction online, the repository provides the precomputed files for MSR-VTT under `data/MSRVTT/`:
- `msrvtt_flow_scores_robust.json`: 12 offline RAFT motion scalar scores per video (after spatial median centering and 0.999 quantile clipping).
- `msrvtt_short_captions.json`: Concise training text descriptions used by the adaptive fusion gate in MGTR.
- `pos_weights_new_light1.pt`: Corpus-derived part-of-speech prior weights ($K=49,408$) for SALS. You can also regenerate POS weights using `python tools/compute_pos_weights.py`.

## Training
Use the commands below to train **DualCA** on each benchmark. By default, evaluation metrics are reported for text-to-video retrieval (`t2v`). To evaluate video-to-text retrieval (`v2t`), append `--metric=v2t`.

Replace `{exp_name}` with your experiment name and `{videos_dir}` with the directory containing the dataset videos.

| Dataset | Training Command |
| :---: | :--- |
| **MSR-VTT (9K)** | `python train.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --noclip_lr=1e-5 --transformer_dropout=0.3 --dataset_name=MSRVTT --msrvtt_train_file=9k --flow_score_path=data/MSRVTT/msrvtt_flow_scores_robust.json --short_caption_path=data/MSRVTT/msrvtt_short_captions.json` |
| **MSVD** | `python train.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --noclip_lr=1e-5 --transformer_dropout=0.4 --dataset_name=MSVD` |
| **LSMDC** | `python train.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --noclip_lr=1e-5 --transformer_dropout=0.3 --dataset_name=LSMDC` |
| **VATEX** | `python train.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --noclip_lr=1e-5 --transformer_dropout=0.3 --dataset_name=VATEX` |

## Evaluation
Trained checkpoints (`model_best.pth`) are saved under `outputs/{exp_name}/`. Use the commands below to evaluate a saved checkpoint on the test split:

| Dataset | Backbone | Evaluation Command |
| :---: | :---: | :--- |
| **MSR-VTT (1K-A)** | CLIP ViT-B/32 | `python test.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --load_epoch=-1 --dataset_name=MSRVTT --msrvtt_train_file=9k --flow_score_path=data/MSRVTT/msrvtt_flow_scores_robust.json` |
| **MSR-VTT (1K-A)** | CLIP ViT-B/16 | `python test.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --load_epoch=-1 --dataset_name=MSRVTT --msrvtt_train_file=9k --clip_arch=ViT-B/16 --flow_score_path=data/MSRVTT/msrvtt_flow_scores_robust.json` |
| **MSVD** | CLIP ViT-B/32 | `python test.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --load_epoch=-1 --dataset_name=MSVD` |
| **LSMDC** | CLIP ViT-B/32 | `python test.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --load_epoch=-1 --dataset_name=LSMDC` |
| **VATEX** | CLIP ViT-B/32 | `python test.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --load_epoch=-1 --dataset_name=VATEX` |

## Citation
If you find this repository useful in your research, please cite our paper:

```bibtex
@inproceedings{tang2027dualca,
  title={Dual-Cue Alignment for Text-Video Retrieval},
  author={Tang, Tuo and Han, Ning},
  booktitle={International Conference on Multimedia Modeling (MMM)},
  year={2027}
}
```

## Acknowledgements
Our implementation is built upon [X-Pool](https://github.com/layer6ai-labs/xpool) and [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip). We thank the authors for releasing their codebases.
