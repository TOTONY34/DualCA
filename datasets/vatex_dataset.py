import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config.base_config import Config
from datasets.video_capture import VideoCapture


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class VATEXDataset(Dataset):
    """Filtered HGR VATEX retrieval split with aligned MGTR scores.

    Every English caption is an independent query. Official VATEX IDs are kept
    as canonical IDs; the audited manifest maps them to ``videoXXX.mp4`` files.
    Each returned sample also contains the 12 PTS-aligned RAFT scalar scores and
    a short-text string. With no short-caption JSON, the original caption is used
    exactly as in the trusted minimal adaptations on earlier datasets.
    """

    _SPLIT_TO_LIST_ATTR = {
        'train': 'vatex_train_list',
        'val': 'vatex_val_list',
        'test': 'vatex_test_list',
    }
    _SPLIT_TO_ANNOTATION = {
        'train': 'vatex_training_v1.0.json',
        'val': 'vatex_validation_v1.0.json',
        'test': 'vatex_validation_v1.0.json',
    }
    _SPLIT_TO_MANIFEST_SOURCE = {
        'train': 'train',
        'val': 'validation',
        'test': 'validation',
    }

    def __init__(self, config: Config, split_type='train', img_transforms=None):
        if split_type not in self._SPLIT_TO_LIST_ATTR:
            raise ValueError(
                'VATEX split_type must be train/val/test, got {!r}'.format(
                    split_type
                )
            )

        self.config = config
        self.split_type = split_type
        self.img_transforms = img_transforms
        self.videos_dir = Path(config.videos_dir).expanduser().resolve()
        self.data_dir = Path(config.vatex_data_dir).expanduser().resolve()
        self.manifest_path = self._resolve_path(config.vatex_manifest_path)
        self.list_path = self._resolve_path(
            getattr(config, self._SPLIT_TO_LIST_ATTR[split_type])
        )
        self.annotation_path = (
            self.data_dir / self._SPLIT_TO_ANNOTATION[split_type]
        )
        self.flow_path = self._resolve_path(config.flow_score_path)
        self.short_caption_path = (
            self._resolve_path(config.short_caption_path)
            if config.short_caption_path
            else None
        )
        self.strict_preprocessed = not config.allow_missing_preprocessed

        self._require_file(self.manifest_path, 'VATEX manifest')
        self._require_file(self.list_path, '{} split list'.format(split_type))
        self._require_file(
            self.annotation_path, '{} annotation'.format(split_type)
        )
        if not config.disable_mgtr or self.strict_preprocessed:
            self._require_file(self.flow_path, 'VATEX MGTR flow-score JSON')

        self.video_ids = self._read_unique_lines(self.list_path)
        if not self.video_ids:
            raise RuntimeError('Empty VATEX split list: {}'.format(self.list_path))

        self.video_paths = self._load_and_validate_manifest()
        self.vid2captions = self._load_selected_english_captions()
        self.flow_scores = self._load_flow_scores()
        self.short_captions = self._load_short_captions()
        self.all_pairs = self._construct_pairs()

        distribution = Counter(
            len(self.vid2captions[video_id]) for video_id in self.video_ids
        )
        print(
            '[VATEX:{}] unique_videos={} caption_video_pairs={} '
            'caption_count_distribution={} flow_entries={} list={}'.format(
                self.split_type,
                len(self.video_ids),
                len(self.all_pairs),
                dict(sorted(distribution.items())),
                len(self.flow_scores),
                self.list_path,
            )
        )

    @staticmethod
    def _require_file(path, label):
        if not path.is_file():
            raise FileNotFoundError('{} not found: {}'.format(label, path))

    @staticmethod
    def _load_json(path):
        with path.open('r', encoding='utf-8') as handle:
            return json.load(handle)

    @staticmethod
    def _read_unique_lines(path):
        values = []
        seen = set()
        with path.open('r', encoding='utf-8-sig') as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                value = raw_line.strip()
                if not value:
                    continue
                if value in seen:
                    raise RuntimeError(
                        'Duplicate ID in {} at line {}: {}'.format(
                            path, line_number, value
                        )
                    )
                seen.add(value)
                values.append(value)
        return values

    @staticmethod
    def _resolve_path(path_value):
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    def _load_and_validate_manifest(self):
        manifest = self._load_json(self.manifest_path)
        if not isinstance(manifest, dict) or not isinstance(
            manifest.get('videos'), dict
        ):
            raise RuntimeError(
                'Manifest must contain a top-level videos dictionary: {}'.format(
                    self.manifest_path
                )
            )

        records = manifest['videos']
        expected_source = self._SPLIT_TO_MANIFEST_SOURCE[self.split_type]
        selected = {}
        missing = []
        wrong_source = []
        missing_files = []

        for video_id in self.video_ids:
            record = records.get(video_id)
            if not isinstance(record, dict):
                missing.append(video_id)
                continue
            if record.get('split') != expected_source:
                wrong_source.append(
                    (video_id, record.get('split'), expected_source)
                )
            path = self._resolve_physical_path(record)
            if not path.is_file():
                missing_files.append((video_id, str(path)))
                continue
            selected[video_id] = str(path)

        if missing:
            raise RuntimeError(
                '{} IDs missing from manifest; examples={}'.format(
                    len(missing), missing[:5]
                )
            )
        if wrong_source:
            raise RuntimeError(
                '{} IDs have wrong manifest source split; examples={}'.format(
                    len(wrong_source), wrong_source[:5]
                )
            )
        if missing_files:
            raise FileNotFoundError(
                '{} manifest-resolved files missing; examples={}'.format(
                    len(missing_files), missing_files[:5]
                )
            )
        return selected

    def _resolve_physical_path(self, record):
        # Prefer the command-line videos_dir so a copied manifest cannot silently
        # redirect a formal run back to the uncompressed source directory.
        filename = record.get('filename')
        if isinstance(filename, str) and filename:
            candidate = self.videos_dir / filename
            if candidate.is_file():
                return candidate.resolve()

        local_stem = record.get('local_stem')
        if isinstance(local_stem, str) and local_stem:
            candidate = self.videos_dir / (local_stem + '.mp4')
            if candidate.is_file():
                return candidate.resolve()

        absolute_path = record.get('absolute_path')
        if isinstance(absolute_path, str) and absolute_path:
            candidate = Path(absolute_path).expanduser()
            if candidate.is_file():
                return candidate.resolve()

        relative_path = record.get('relative_path')
        if isinstance(relative_path, str) and relative_path:
            candidate = self.data_dir / relative_path
            if candidate.is_file():
                return candidate.resolve()

        if isinstance(filename, str) and filename:
            return (self.videos_dir / filename).resolve()
        raise RuntimeError('Manifest record has no usable path: {}'.format(record))

    def _load_selected_english_captions(self):
        annotations = self._load_json(self.annotation_path)
        if not isinstance(annotations, list):
            raise RuntimeError(
                'VATEX annotation top level must be a list: {}'.format(
                    self.annotation_path
                )
            )

        selected_ids = set(self.video_ids)
        output = {}
        for index, entry in enumerate(annotations):
            if not isinstance(entry, dict):
                raise RuntimeError(
                    'Annotation {} is not a dictionary'.format(index)
                )
            video_id = entry.get('videoID')
            if video_id not in selected_ids:
                continue
            if video_id in output:
                raise RuntimeError(
                    'Duplicate annotation videoID: {}'.format(video_id)
                )
            raw_captions = entry.get('enCap')
            if not isinstance(raw_captions, list) or not raw_captions:
                raise RuntimeError(
                    'Missing English captions for {}'.format(video_id)
                )
            captions = []
            for caption in raw_captions:
                if not isinstance(caption, str) or not caption.strip():
                    raise RuntimeError(
                        'Invalid English caption for {}'.format(video_id)
                    )
                captions.append(' '.join(caption.split()))
            output[video_id] = captions

        missing = [video_id for video_id in self.video_ids if video_id not in output]
        if missing:
            raise RuntimeError(
                '{} split IDs missing captions; examples={}'.format(
                    len(missing), missing[:5]
                )
            )
        return output

    def _load_flow_scores(self):
        if not self.flow_path.is_file():
            if self.strict_preprocessed:
                raise FileNotFoundError(
                    'Flow score file not found: {}'.format(self.flow_path)
                )
            return {}

        payload = self._load_json(self.flow_path)
        if not isinstance(payload, dict):
            raise RuntimeError('Flow-score JSON must be a dictionary')

        selected = {}
        missing = []
        invalid = []
        for video_id in self.video_ids:
            values = payload.get(video_id)
            if values is None:
                missing.append(video_id)
                continue
            if not isinstance(values, list) or len(values) != self.config.num_frames:
                invalid.append((video_id, 'length', len(values) if isinstance(values, list) else None))
                continue
            try:
                array = np.asarray(values, dtype=np.float32)
            except (TypeError, ValueError):
                invalid.append((video_id, 'non-numeric', None))
                continue
            if array.shape != (self.config.num_frames,) or not np.isfinite(array).all():
                invalid.append((video_id, 'non-finite/shape', array.shape))
                continue
            selected[video_id] = array

        if (missing or invalid) and self.strict_preprocessed:
            raise RuntimeError(
                'VATEX flow coverage/shape check failed: missing={} invalid={} '
                'missing_examples={} invalid_examples={}'.format(
                    len(missing), len(invalid), missing[:5], invalid[:5]
                )
            )
        return selected

    def _load_short_captions(self):
        if self.short_caption_path is None:
            return {}
        self._require_file(self.short_caption_path, 'short-caption JSON')
        payload = self._load_json(self.short_caption_path)
        if not isinstance(payload, dict):
            raise RuntimeError('Short-caption JSON must be a dictionary')
        return payload

    def _construct_pairs(self):
        pairs = []
        for video_id in self.video_ids:
            for caption in self.vid2captions[video_id]:
                pairs.append((video_id, caption))
        return pairs

    def _get_short_text(self, video_id, caption):
        candidates = self.short_captions.get(video_id)
        if candidates is None:
            return caption
        if isinstance(candidates, str):
            candidates = [candidates]
        if not isinstance(candidates, (list, tuple)) or not candidates:
            return caption
        value = random.choice(candidates) if self.split_type == 'train' else candidates[0]
        if not isinstance(value, str):
            return caption
        value = value.strip('"').strip("'").strip()
        if 'Subject:' in value:
            value = (
                value.replace('Subject:', '')
                .replace('Verb:', '')
                .replace('Object:', '')
            )
        value = ' '.join(value.split())
        return value if value else caption

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, index):
        video_id, caption = self.all_pairs[index]
        frames, _ = VideoCapture.load_frames_from_video(
            self.video_paths[video_id],
            self.config.num_frames,
            self.config.video_sample_type,
        )
        if self.img_transforms is not None:
            frames = self.img_transforms(frames)

        flow = self.flow_scores.get(video_id)
        if flow is None:
            flow = np.zeros(self.config.num_frames, dtype=np.float32)

        return {
            'video_id': video_id,
            'video': frames,
            'text': caption,
            'short_text': self._get_short_text(video_id, caption),
            'flow_scores': torch.from_numpy(flow.copy()),
        }
