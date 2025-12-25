'''
LinCIR
Copyright (c) 2023-present NAVER Corp.
CC BY-NC-4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
'''
import clip
import torch
from clip.model import CLIP
from transformers import CLIPTextModelWithProjection
import torch.nn.functional as F
import json
import os

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    Copy-paste from https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_clip.py#L679-L693
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def encode_with_pseudo_tokens_HF(clip_model: CLIPTextModelWithProjection, text: torch.Tensor, pseudo_tokens: torch.Tensor,
                              num_tokens=1, return_last_states=False) -> torch.Tensor:
    x = clip_model.text_model.embeddings.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]
    '''
    pseudo_tokens :the output of phi
    
    '''
    x = torch.where(text.unsqueeze(-1) == 259,
                    pseudo_tokens.unsqueeze(1).type(clip_model.dtype),
                    x)
    x = x + clip_model.text_model.embeddings.position_embedding(clip_model.text_model.embeddings.position_ids)
    _causal_attention_mask = _make_causal_mask(text.shape, x.dtype, device=x.device)
    x = clip_model.text_model.encoder(inputs_embeds=x,
                                      attention_mask=None,
                                      causal_attention_mask=_causal_attention_mask,
                                      output_attentions=False,
                                      output_hidden_states=False,
                                      return_dict=False)
    x = x[0]
    x_last = clip_model.text_model.final_layer_norm(x)
    x = x_last[torch.arange(x_last.shape[0], device=x_last.device),
          text.to(dtype=torch.int, device=x_last.device).argmax(dim=-1),
          ]
    if hasattr(clip_model, 'text_projection'):
        x = clip_model.text_projection(x)

    if return_last_states:
        return x, x_last
    else:
        return x

def encode_textEncoder(tokenizer ,input_captions_r , accelerator,clip_model: CLIPTextModelWithProjection)-> torch.Tensor:
    input_ids = tokenizer(
                input_captions_r,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77
            )["input_ids"].to(accelerator.device)
    emb = clip_model.text_model.embeddings.token_embedding(input_ids).type(clip_model.dtype)#create vector 77*768
    emb = emb + clip_model.text_model.embeddings.position_embedding(clip_model.text_model.embeddings.position_ids) #torch.Size([1, 77, 768])
    #TODO add batch size
    _causal_attention_mask = _make_causal_mask(input_ids.shape, emb.dtype, device=emb.device)
    x = clip_model.text_model.encoder(inputs_embeds=emb,
                                      attention_mask=None,
                                      causal_attention_mask=_causal_attention_mask,
                                      output_attentions=False,
                                      output_hidden_states=False,
                                      return_dict=False) #(batch_size, seq_len, embed_dim) →(1, 77, 768)
    x = x[0] # torch.Size([1, 77, 768])
    x_last = clip_model.text_model.final_layer_norm(x) #torch.Size([1, 77, 768])
    x = x_last[torch.arange(x_last.shape[0], device=x_last.device),
            input_ids.to(dtype=torch.int, device=x_last.device).argmax(dim=-1),
            ] #torch.Size([1, 768])
    if hasattr(clip_model, 'text_projection'):
        x = clip_model.text_projection(x)
    cls = x #torch.Size([1, 768])
    return cls
def calculate_validation2(clip_model: CLIPTextModelWithProjection, text_encoder, tokenizer, phi_for_eval,VRDDataset_test , relation_to_tokens_test, save_path, args,accelerator):
    
    scores = []
    scores2 = []
    relation_embeddings = {}
    best_r = {}
    for r , tokens  in relation_to_tokens_test.items():
        input_captions_r = f"a photo demonstrating {r}"
        cls =encode_textEncoder(tokenizer ,input_captions_r , accelerator,clip_model)#torch.Size([1, 768])
        relation_embeddings[r] = cls #(batch_size, embed_dim) → (1, 768)
    results = []
    # Loop over test dataset
    for imgTest, targetTest in VRDDataset_test:
        relations = targetTest.get("relations_text", [])
        
        for s, r, o in relations:
            triplet_string = f"{s} {r} {o}"
            cls22 = encode_textEncoder(tokenizer ,triplet_string , accelerator,clip_model) #torch.Size([1, 768])
            
            
            # Phi prediction
            #estimated_token_embeddings = phi_for_eval(input_features).squeeze(0)  # shape [dim]
            estimated_token_embeddings = phi_for_eval(cls22) #estimated_token_embeddings , input_features=torch.Size([1, 768])
            #estimated_token_embeddings = torch.vstack(estimated_token_embeddings)
            estimated_token_embeddings = estimated_token_embeddings.to(accelerator.device)#[1, 768]
            input_captions = [
            f"a photo demonstrating $"]
            tokenized_input_captions = clip.tokenize(input_captions, context_length=77).to(accelerator.device)
            text_features = encode_with_pseudo_tokens_HF(clip_model, tokenized_input_captions, estimated_token_embeddings)
            cls2 = F.normalize(text_features)
            estimated_token_embeddings = cls2 #torch.Size([1, 768])
            
            #TODO create new function to calculate cls + add x_last 
            # Compute cosine similarity with all relation embeddings
           
            best_rel = None
            best_score = -1
            # TODO: Use vectorized operations
            ###########################
            rel_texts = list(relation_embeddings.keys())
            rel_embs = torch.stack([relation_embeddings[rel] for rel in rel_texts])  
            # rel_embs.shape
            # torch.Size([305, 1, 768])

            # estimated_token_embeddings.shape
            # torch.Size([1, 768])
            estimated = estimated_token_embeddings.unsqueeze(0).expand(rel_embs.size(0), 1, -1)
            # Compute cosine similarities / after f.cosine -> [305, 1] — one cosine value per (batch, pair)/ after squeeze-> [305]
            estimated = F.normalize(estimated)
            rel_embs = F.normalize(rel_embs)
            cos_sims = F.cosine_similarity(estimated, rel_embs, dim=2).squeeze(1) #torch.Size([305, 1, 768])

            # Get the best one
            best_idx = torch.argmax(cos_sims).item()
            best_score = cos_sims[best_idx].item()
            best_rel = rel_texts[best_idx]
            #results.append({
            #   "predicted": best_rel,
            #  "ground_truth": r,
            #})
            ###########################
            # Compare best match with ground truth r
            scores.append(1 if best_rel == r else 0)
            best_r[r] = best_rel

    # Compute accuracy
    accuracy = sum(scores) / len(scores) if len(scores) > 0 else 0.0

    # Save accuracy to JSON file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # Use a unique key for each call
    results[f"run_{len(results)+1}"] = accuracy

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    return accuracy