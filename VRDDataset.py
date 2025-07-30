"""
Visual Relation Detection (VRD) style dataset loader
===================================================

This file rewrites / refactors the original Visual Genome (VG) dataset class you
provided into a *generic* loader for a **Visual Relation Detection** style dataset
with the directory & JSON organization you just showed:

    vrd/
      sg_train_images/               # training images (jpg / png)
      sg_test_images/                # test / val images
      sg_train_annotations.json      # list[image_record]
      sg_test_annotations.json       # same structure

Each *image_record* (sample) in the annotations JSON is a dictionary with at least:

    {
        "photo_id": str | int,               # unique image id (string ok)
        "filename": str,                    # image filename relative to split folder
        "height": int,
        "width": int,
        "objects": [                        # list of objects
            {
                "names": [str, ...],        # one or more candidate names (we take first by default)
                "bbox": {"x": float, "y": float, "w": float, "h": float},  # top-left + size (pixel)
                "attributes": [
                    {"attribute": str, "text": [ ... ]},  # optional attribute records
                    ...
                ]
            }, ...
        ],
        "relationships": [
            {
                "objects": [sub_idx, obj_idx],
                "relationship": str,       # predicate phrase (may contain spaces)
                "text": [ ... ]            # optional textual description tokens/phrases
            }, ...
        ]
    }

Differences from VG original code
---------------------------------
1. **No HDF5**: We ingest plain JSON; everything is already per-image.
2. **Dynamic vocab building**: Object & predicate vocabularies are derived from the
   training annotation file (unless you provide cached vocab JSON to keep index stability).
3. **Simpler transforms path**: We can plug in a `transforms` callable identical to the
   torchvision DETR style (img, target) -> (img, target).
4. **No base/novel filtering logic** unless toggled; you can extend via hooks.
5. **Supports multiple synonyms per object** (the `names` list). We choose the first
   or can merge all (configurable).
6. **Edge duplication filtering** optional.
7. **Attribute extraction** optional.
8. **Caption / text prompt generation** (optional) using enumerated object & predicate vocab.

Public API
----------
Classes / functions provided:

- `VRDConfig`: Dataclass holding configuration flags.
- `VRDDataset`: `torch.utils.data.Dataset` subclass.
- `build_vrd(...)`: helper building train / val datasets (and automatically building /
  saving vocab on train split).

Returned `target` dict fields (for a sample):

    target = {
        'image_id': str/int,
        'boxes': FloatTensor [N,4]  (x1,y1,x2,y2)
        'labels': LongTensor [N]    (object class indices)
        'edges':  LongTensor [R,3]  (sub_ind, obj_ind, predicate_class_index)
        'orig_size': LongTensor [2] (h, w)
        'iscrowd': zeros [N]
        # Optional extras when enabled:
        'object_names': List[str] length N (canonical names)
        'predicates': List[str] length num_predicates (global list) or None
        'relations_text': List[Tuple[sub_name, predicate, obj_name]]
        'attributes': List[List[str]] (attributes per object) if available
        'caption': str  (concatenated object vocab for OVD style) if enabled
        'rel_caption': str (concatenated predicate vocab) if enabled
    }

Edge Cases handled:
- Empty relationship list -> allowed at eval time; at train time optionally re-sample.
- Duplicate relationships (same subject, object, predicate) can be collapsed.
- Bounding box coordinate conversion (provided as x,y,w,h -> x1,y1,x2,y2).

Usage Example
-------------
```
from vrd_dataset import build_vrd
train_set = build_vrd(
    root='path/to/vrd',
    split='train',
    config=VRDConfig(flip_augmentation=True, build_text_prompts=True)
)
img, target = train_set[0]
```

You may serialize the generated vocabulary for stable indices across runs:
```
train_set.save_vocab('vrd_vocab.json')
# Later
vrd_vocab = json.load(open('vrd_vocab.json'))
val_set = build_vrd(root, 'test', config, vocab=vrd_vocab)
```

TODO markers indicate optional extension points you can fill per project.

"""
from __future__ import annotations
import os
import json
import copy
from pickle import FALSE
import random
from dataclasses import dataclass, field
from sre_parse import Tokenizer
from typing import Callable, Dict, List, Optional, Tuple, Any, Sequence

from numpy._typing import _UnknownType
import torch
from torch.utils.data import Dataset
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------
@dataclass
class VRDConfig:
    max_objects: int = 100                 # cap objects per image (random subsample if exceeded)
    filter_duplicate_rels: bool = True     # remove duplicate triples (same subj,obj,pred)
    require_overlap: bool = False          # if True, keep only overlapping boxes for rels (needs IoU function)
    flip_augmentation: bool = False        # random horizontal flip during training
    training: bool = True                  # set False for eval split
    build_text_prompts: bool = False       # build concatenated vocab captions (OVD style)
    use_all_names: bool = False            # if True, merge all 'names' synonyms into caption (still one class id)
    min_relations: int = 0                 # if >0, images with fewer rels are skipped (train only)
    resample_on_empty_rels: bool = True    # during training, recursively sample another idx if no rels
    seed: int = 42                         # RNG seed for reproducibility of subsampling
    # Future extensions
    lowercase: bool = True
    # optional externally supplied vocab
    '''
    __background__	Default object	Represents background or ignored objects
    [UNK]	Default predicate	Represents unknown or rare relations
    '''
    object_bg_token: str = "__background__"
    predicate_bg_token: str = "[UNK]"      # keep compatibility with original code's special token
    # caption formatting
    caption_joiner: str = '. '
    caption_period: bool = True

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def _canonical_name(names: Sequence[str], lowercase: bool = True) -> str:
    if not names:
        return 'object'
    name = names[0]
    return name.lower() if lowercase else name

##x1 , y1 : bala chap
##x2 , y2 : payin rast
def _bbox_xywh_to_xyxy(bbox: Dict[str, float]) -> Tuple[float, float, float, float]:
    # Input keys: x,y,w,h  (top-left + size)
    x1 = float(bbox['x'])
    y1 = float(bbox['y'])
    x2 = x1 + float(bbox['w'])
    y2 = y1 + float(bbox['h'])
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Vocabulary Builder
# ---------------------------------------------------------------------------
class VRDVocabulary:
    def __init__(self, object_bg: str = '__background__', predicate_bg: str = '[UNK]'):
        self.object_bg = object_bg # Added this line
        self.predicate_bg = predicate_bg # Added this line
        self.obj2id: Dict[str, int] = {self.object_bg: 0} # Changed to self.object_bg
        self.id2obj: List[str] = [self.object_bg]         # Changed to self.object_bg
        self.pred2id: Dict[str, int] = {self.predicate_bg: 0} # Changed to self.predicate_bg
        self.id2pred: List[str] = [self.predicate_bg] # Changed to self.predicate_bg
    '''
    def __init__(self, object_bg: str = '__background__', predicate_bg: str = '[UNK]'):
        self.object_bg = object_bg
        self.predicate_bg = predicate_bg
        self.obj2id: Dict[str, int] = {object_bg: 0} # maps object name to ID
        self.id2obj: List[str] = [object_bg]         # reverse: maps ID to name
        self.pred2id: Dict[str, int] = {predicate_bg: 0}
        self.id2pred: List[str] = [predicate_bg]
    '''
    def add_object(self, name: str) -> int:
        if name not in self.obj2id:
            self.obj2id[name] = len(self.id2obj)
            self.id2obj.append(name)
        return self.obj2id[name]

    def add_predicate(self, name: str) -> int:
        if name not in self.pred2id:
            self.pred2id[name] = len(self.id2pred)
            self.id2pred.append(name)
        return self.pred2id[name]

    def object_count(self) -> int:
        return len(self.id2obj)

    def predicate_count(self) -> int:
        return len(self.id2pred)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'object_bg': self.object_bg,
            'predicate_bg': self.predicate_bg,
            'objects': self.id2obj,
            'predicates': self.id2pred,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'VRDVocabulary':
        vocab = cls(object_bg=d.get('object_bg', '__background__'), predicate_bg=d.get('predicate_bg', '[UNK]'))
        vocab.id2obj = list(d['objects'])
        vocab.obj2id = {n: i for i, n in enumerate(vocab.id2obj)}
        vocab.id2pred = list(d['predicates'])
        vocab.pred2id = {n: i for i, n in enumerate(vocab.id2pred)}
        return vocab

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class VRDDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        annotation_file: str,
        image_subdir: Optional[str] = None,
        transforms: Optional[Callable] = None,
        config: Optional[VRDConfig] = None,
        vocab: Optional[VRDVocabulary] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Parameters
        ----------
        root : dataset root directory (contains the split image folders & annotation files)
        split : 'train', 'val', 'test', etc (used for augmentation decisions)
        annotation_file : path to JSON list with image records (structure as described above)
        image_subdir : override subdirectory containing images (if None, inferred from split)
        transforms : callable(img, target) -> (img, target)
        config : VRDConfig (default if None)
        vocab : prebuilt VRDVocabulary (if None and training, will be built; if None and NOT training, raises)
        """
        self.root = root
        self.split = split
        self.tokenizer = tokenizer
        self.config = config or VRDConfig(training=(split == 'train'))
        #The sequence of random numbers is the same every time the code runs
        random.seed(self.config.seed)
        self.transforms = transforms

        if image_subdir is None:
            # default naming convention like: sg_train_images, sg_test_images
            image_subdir = f'sg_{split}_images'
        self.image_dir = os.path.join(root, image_subdir)

        with open(annotation_file, 'r') as f:
            self.records: List[Dict[str, Any]] = json.load(f)

        # Build or load vocabulary
        if vocab is None:
            if not self.config.training:
                raise ValueError("Vocabulary must be supplied for non-training split to keep indices stable.")
            vocab = VRDVocabulary(self.config.object_bg_token, self.config.predicate_bg_token)
            self._build_vocab(vocab)
        self.vocab = vocab

        # Pre-extract per-image tensorizable info to accelerate __getitem__
        self.index: List[Dict[str, Any]] = []  # stores processed object list & relationships
        self._prepare_index()

    # --------------------------- Vocabulary Building -----------------------
    def _build_vocab(self, vocab: VRDVocabulary):
        for rec in self.records:
            # objects
            for obj in rec.get('objects', []):
                cname = _canonical_name(obj.get('names', []), lowercase=self.config.lowercase)
                vocab.add_object(cname)
            # predicates
            for rel in rec.get('relationships', []):
                pname = rel.get('relationship', '').strip().lower() if self.config.lowercase else rel.get('relationship', '')
                if pname:
                    vocab.add_predicate(pname)

    # --------------------------- Index Preparation ------------------------
    def _prepare_index(self):
        cfg = self.config
        for rec in self.records:
            objs = rec.get('objects', [])
            rels = rec.get('relationships', [])

            # build object list
            obj_names: List[str] = []
            attributes: List[List[str]] = []
            boxes_xyxy: List[Tuple[float, float, float, float]] = []
            for o in objs:
                cname = _canonical_name(o.get('names', []), lowercase=cfg.lowercase)
                obj_names.append(cname)
                b = _bbox_xywh_to_xyxy(o['bbox'])
                boxes_xyxy.append(b)
                attr_list = []
                for attr in o.get('attributes', []):
                    a = attr.get('attribute')
                    if a:
                        if cfg.lowercase:
                            a = a.lower()
                        attr_list.append(a)
                attributes.append(attr_list)

            # convert relationship list to triple indices
            edges: List[Tuple[int, int, int]] = []
            for r in rels:
                pair = r.get('objects', [])
                if len(pair) != 2:
                    continue
                s, oidx = pair
                if s >= len(obj_names) or oidx >= len(obj_names):
                    continue  # skip malformed
                pred_name = r.get('relationship', '')
                if cfg.lowercase:
                    pred_name = pred_name.lower()
                if pred_name not in self.vocab.pred2id:
                    # In eval splits, we may encounter unseen predicate -> map to background token
                    pred_id = self.vocab.pred2id[self.vocab.predicate_bg]
                else:
                    pred_id = self.vocab.pred2id[pred_name]
                edges.append((s, oidx, pred_id))

            # optional min relations filter (train only)
            if cfg.training and cfg.min_relations > 0 and len(edges) < cfg.min_relations:
                continue

            self.index.append({
                'record': rec,
                'object_names': obj_names,
                'attributes': attributes,
                'boxes': boxes_xyxy,
                'edges': edges,
            })

    # --------------------------- Public Helpers ---------------------------
    def save_vocab(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.vocab.to_dict(), f, indent=2)

    # --------------------------- Dataset API ------------------------------
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        sample = self.index[idx]
        rec = sample['record']
        h = rec['height']
        w = rec['width']

        # load image
        filename = rec['filename']
        img_path = os.path.join(self.image_dir, filename)
        img = Image.open(img_path).convert('RGB')

        boxes = torch.tensor(sample['boxes'], dtype=torch.float32)  # [N,4] x1,y1,x2,y2
        names = sample['object_names']     
        #labels = torch.tensor([self.vocab.obj2id[n] for n in names], dtype=torch.int64)
        labels = torch.tensor([self.vocab.obj2id.get(n, self.vocab.obj2id[self.vocab.object_bg]) for n in names], dtype=torch.int64)
        edges = torch.tensor(sample['edges'], dtype=torch.int64) if len(sample['edges']) else torch.zeros((0,3), dtype=torch.int64)

        # Cap objects if necessary (subsample + remap edges)
        cfg = self.config
        if boxes.shape[0] > cfg.max_objects:
            keep_idx = torch.randperm(boxes.shape[0])[:cfg.max_objects]
            keep_idx_sorted, _ = torch.sort(keep_idx)
            old2new = {int(o): i for i, o in enumerate(keep_idx_sorted.tolist())}
            boxes = boxes[keep_idx_sorted]
            labels = labels[keep_idx_sorted]
            names = [names[i] for i in keep_idx_sorted.tolist()]
            # filter & remap edges
            new_edges_list = []
            for (s, o, p) in edges.tolist():
                if s in old2new and o in old2new:
                    new_edges_list.append((old2new[s], old2new[o], p))
            edges = torch.tensor(new_edges_list, dtype=torch.int64) if new_edges_list else torch.zeros((0,3), dtype=torch.int64)

        # Deduplicate relationships if required
        if cfg.filter_duplicate_rels and edges.shape[0] > 0:
            uniq = {}
            for i, (s,o,p) in enumerate(edges.tolist()):
                uniq.setdefault((s,o,p), i)
            if len(uniq) != edges.shape[0]:
                edges = torch.tensor(list(uniq.keys()), dtype=torch.int64)

        # Horizontal flip augmentation
        if cfg.training and cfg.flip_augmentation and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w_img = img.width
            # flip boxes
            x1 = boxes[:,0].clone()
            x2 = boxes[:,2].clone()
            boxes[:,0] = w_img - x2
            boxes[:,2] = w_img - x1

        target: Dict[str, Any] = {
            'image_id': rec.get('photo_id', idx),
            'orig_size': torch.tensor([h, w], dtype=torch.int64),
            'boxes': boxes,
            'labels': labels,
            'edges': edges,
            'iscrowd': torch.zeros((boxes.shape[0],), dtype=torch.int64),
            'object_names': names,
            'attributes': sample['attributes'],  # list[list[str]]
        }

        # Relation text triples for potential captioning / language heads
        if edges.shape[0] > 0:
            triples = []
            for (s,o,p) in edges.tolist():
                triples.append((names[s], self.vocab.id2pred[p], names[o]))
            target['relations_text'] = triples
        else:
            target['relations_text'] = []

        # Optional captions
        if self.config.build_text_prompts:
            # Object caption
            obj_vocab_seq = self.vocab.id2obj[1:]  # skip background
            if self.config.lowercase:
                obj_vocab_seq = [o.lower() for o in obj_vocab_seq]
            caption = self.config.caption_joiner.join(obj_vocab_seq)
            if self.config.caption_period and not caption.endswith('.'):
                caption += '.'
            target['caption'] = caption
            # Predicate caption
            pred_vocab_seq = self.vocab.id2pred[1:]  # skip background
            pred_caption = self.config.caption_joiner.join(pred_vocab_seq)
            if self.config.caption_period and not pred_caption.endswith('.'):
                pred_caption += '.'
            target['rel_caption'] = pred_caption

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if cfg.training and cfg.resample_on_empty_rels and target['edges'].shape[0] == 0 and len(self) > 1:
            # resample a different index (avoid infinite loops by disabling recursion on second call)
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

        ###############################
        '''
        relations = target.get('relations_text', [])
        if  relations:
            

            triplet_strings = [f"{s} {r} {o}" for s, r, o in relations]
            full_caption = '. '.join(triplet_strings)

            j = random.randint(0, len(relations) - 1)
            masked_triplets = triplet_strings.copy()
            s, r, o = relations[j]
            masked_triplets[j] = f"{s} [$] {o}"
            masked_caption = '. '.join(masked_triplets)
            target['caption'] = full_caption
            target['replaced_caption'] = masked_caption

        ###############################
        '''
        relations = target.get('relations_text', [])
        if not relations:
            return "", "", 0

        triplet_strings = [f"{s} {r} {o}" for s, r, o in relations]
        full_caption_raw = '. '.join(triplet_strings)
        
        
        tokenized = self.tokenizer(full_caption_raw, return_tensors='pt', padding=False, truncation=False)
        input_ids = tokenized['input_ids'][0]

        if len(input_ids) <= 77:
            final_triplets = triplet_strings
        else:
           
            current_caption = ""
            final_triplets = []
            for triplet in triplet_strings:
                tentative = current_caption + (". " if current_caption else "") + triplet
                tokens = self.tokenizer(tentative, return_tensors='pt', padding=False, truncation=False)['input_ids'][0]
                if len(tokens) <= 77:
                    current_caption = tentative
                    final_triplets.append(triplet)
                else:
                    break

        
        full_caption = '. '.join(final_triplets)

       
        j = random.randint(0, len(final_triplets) - 1)
        s, r, o = final_triplets[j].split(' ', 2)
        masked_triplets = final_triplets.copy()
        masked_triplets[j] = f"{s} [$] {o}"
        masked_caption = '. '.join(masked_triplets)
        target['caption'] = full_caption
        target['replaced_caption'] = masked_caption
        #return full_caption, masked_caption, 1
        return img, target

# ---------------------------------------------------------------------------
# Builder helper
# ---------------------------------------------------------------------------

def build_vrd(
    root: str,
    split: str,
    config: Optional[VRDConfig] = None,
    transforms: Optional[Callable] = None,
    vocab: Optional[Dict[str, Any]] = None,
    annotation_filename: Optional[str] = None,
    tokenizer : Optional[Any] = None,
):
    """Factory building the VRDDataset.

    Parameters
    ----------
    root : dataset root (holds image folders and annotation json files)
    split : 'train' or 'test'/'val'
    config : VRDConfig
    transforms : torchvision-style transform
    vocab : Either None (will build if training) OR a dict produced by VRDVocabulary.to_dict
    annotation_filename : override default annotation filename
    """
    if annotation_filename is None:
        annotation_filename = f'sg_{split}_annotations.json'
    ann_path = os.path.join(root, annotation_filename)

    vocab_obj: Optional[VRDVocabulary] = None
    if vocab is not None:
        vocab_obj = VRDVocabulary.from_dict(vocab)

    ds = VRDDataset(
        root=root,
        split=split,
        annotation_file=ann_path,
        image_subdir=None,  # infer from split
        transforms=transforms,
        config=config,
        vocab=vocab_obj,
        tokenizer = tokenizer,
    )
    return ds

# ---------------------------------------------------------------------------
# Example main (debug)
# ---------------------------------------------------------------------------

def createDataset (root: str ,split: str,vocab: str, save_vocab: str , tokenizer):

    config = VRDConfig(build_text_prompts=False, flip_augmentation=False, training=(split=='train'))
    vocab_dict = None
    if vocab:
        with open(vocab, 'r') as f:
            vocab_dict = json.load(f)

    dataset = build_vrd(root, split, config=config, vocab=vocab_dict , tokenizer=tokenizer)
    print(f"Loaded {len(dataset)} {split} samples. Object classes: {dataset.vocab.object_count()} | Predicates: {dataset.vocab.predicate_count()}")

    if save_vocab and split=='train':
        dataset.save_vocab(save_vocab)

    return dataset


'''

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Path to VRD root directory')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--vocab', type=str, default='')
    parser.add_argument('--save_vocab', type=str, default='')
    args = parser.parse_args()

    config = VRDConfig(build_text_prompts=False, flip_augmentation=False, training=(args.split=='train'))

    vocab_dict = None
    if args.vocab:
        with open(args.vocab, 'r') as f:
            vocab_dict = json.load(f)

    dataset = build_vrd(args.root, args.split, config=config, vocab=vocab_dict)
    print(f"Loaded {len(dataset)} {args.split} samples. Object classes: {dataset.vocab.object_count()} | Predicates: {dataset.vocab.predicate_count()}")

    if args.save_vocab and args.split=='train':
        dataset.save_vocab(args.save_vocab)
        print(f"Saved vocab to {args.save_vocab}")

    # quick sanity sample
    img, target = dataset[0]
    print(target['image_id'], target['boxes'].shape, target['edges'].shape)
    print(target['caption'])
    print(target['replaced_caption'])
    #print(target['edges'])
    #print(target['boxes'])
    
    #print(target.get('relations_text')[:5])
    #print(target)
    #print(len(target.get('relations_text')))
    #print(target.get('rel_caption'))
    #print(target.get('caption')[:50])
'''
