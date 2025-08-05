'''
LinCIR
Copyright (c) 2023-present NAVER Corp.
CC BY-NC-4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
'''
import os
import functools
import glob
import random
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Literal
import PIL
import PIL.Image
import torch
from torch.utils.data import Dataset
import webdataset as wds
import spacy
import numpy as np
import datasets
from transformers import CLIPTextModelWithProjection


class captionVRDDataset(Dataset):
    def __init__(self, original_dataset, tokenizer, max_len=77):
        self.original_dataset = original_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = self.extract_samples()

    def extract_samples(self):
        samples = []
        for i in range(len(self.original_dataset)):
            img, target = self.original_dataset[i]
            
            samples.append({
                "caption": target['caption'] ,
                "replaced_caption": target['replaced_caption']
            })

        return samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        caption = sample['caption'].lower().strip()
        masked = sample['replaced_caption'].lower().strip()

        # tokenize original
        tokenized = self.tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
        # tokenize masked
        masked_tokenized = self.tokenizer(masked, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)

        mask_token_id = 49408 
        replacement_token_id = 259  
        replaced_tokens = masked_tokenized['input_ids'][0]
        replaced_tokens = torch.where(
            replaced_tokens == mask_token_id,
            torch.ones_like(replaced_tokens) * replacement_token_id,
            replaced_tokens
        )
        if len(tokenized['input_ids'][0]) > 77:
            print('moreeeee thannnnn 77')
        if len(masked_tokenized['input_ids'][0]) > 77:
            print('moreeeee thannnnn 77')
        '''
        return {
            "tokens": tokenized['input_ids'][0],
            "replaced_tokens": replaced_tokens,
            "indicator": torch.tensor(1)
        }
        '''
        return tokenized['input_ids'][0], replaced_tokens, torch.tensor(1)


def build_loader(args, tokenizer, original_dataset):
    dataset = captionVRDDataset(original_dataset, tokenizer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    return loader

def create_relation_to_tokens_test(VRDDataset_test , tokenizer):
    
    relation_to_tokens_test  = {}
    for imgTest, targetTest in VRDDataset_test:
        relations = targetTest.get('relations_text', [])
        for s, r, o in relations:
            if r not in relation_to_tokens_test :
                tokenized_relation = tokenizer(r, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
                # TODO: What is 0?
                '''
               tokenized_relation['input_ids'] =  tensor([[49406,   525, 49407]])
               tokenized_relation['input_ids'][0] = tensor([49406,   525, 49407])
               declining the dimention
                '''
               
                relation_to_tokens_test[r]=tokenized_relation['input_ids'][0]
    return  relation_to_tokens_test     