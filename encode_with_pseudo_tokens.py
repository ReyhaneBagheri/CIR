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

def extract_pseudo_tokens_with_phi(clip_model: CLIPTextModelWithProjection, text_encoder, tokenizer, phi_for_eval,VRDDataset_test , relation_to_tokens_test, save_path, args,accelerator):
    scores = []

    # Precompute relation embeddings (so we don't recompute for every image)
    
    relation_embeddings = {}
    
    for r , tokens  in relation_to_tokens_test.items():
        #####
        #x = clip_model.text_model.embeddings.token_embedding(text).type(clip_model.dtype) 
        #####
        input_ids = tokens.to(clip_model.text_model.embeddings.token_embedding.weight.device)
        emb = clip_model.text_model.embeddings.token_embedding(input_ids).type(clip_model.dtype)#create vector 77*768
        relation_embeddings[r] = emb[1] # example:start wearing end 0 0 0 ... 
        # with torch.no_grad():
        #     org_rel = text_encoder(input_ids=tokens.to(accelerator.device))
        #     rel_emb = org_rel.text_embeds.clone()
        #     if args.l2_normalize:
        #         rel_emb = F.normalize(rel_emb, dim=-1)
        # relation_embeddings[r] = rel_emb.squeeze(0)
    results = []
    # Loop over test dataset
    for imgTest, targetTest in VRDDataset_test:
        relations = targetTest.get("relations_text", [])
        
        for s, r, o in relations:
            triplet_string = f"{s} {r} {o}"
            tokenized_triplet = tokenizer(
                triplet_string,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77
            )["input_ids"].to(accelerator.device)#torch.Size([1, 77])

            # Encode triplet after the transformer and projection
            org = text_encoder(input_ids=tokenized_triplet)
            input_features = org.text_embeds.clone()
            if args.l2_normalize:
                input_features = F.normalize(input_features, dim=-1)#[1, 768]

            # Phi prediction
            #estimated_token_embeddings = phi_for_eval(input_features).squeeze(0)  # shape [dim]
            estimated_token_embeddings = phi_for_eval(input_features) #torch.Size([1, 768])
           
            if estimated_token_embeddings.dim() == 2 and estimated_token_embeddings.size(0) == 1:
                estimated_token_embeddings = estimated_token_embeddings.squeeze(0)#torch.Size([768])
            elif estimated_token_embeddings.dim() == 1:
                pass  # already in shape [dim]
            else:
                raise ValueError(f"Unexpected shape from phi: {estimated_token_embeddings.shape}")
            # Compute cosine similarity with all relation embeddings
           
            best_rel = None
            best_score = -1
            # TODO: Use vectorized operations
            ###########################
            # Stack all relation embeddings into one tensor [num_relations, hidden_dim]
            rel_texts = list(relation_embeddings.keys())
            rel_embs = torch.stack([relation_embeddings[rel] for rel in rel_texts])  # shape: [N, D]

            # Expand estimated embedding to match [N, D]
            #estimated = estimated_token_embeddings.unsqueeze(0).expand_as(rel_embs)  # shape: [N, D]
            estimated = estimated_token_embeddings.unsqueeze(0).expand(rel_embs.size(0), -1)  # shape: [N, D]
            # Compute cosine similarities
            cos_sims = F.cosine_similarity(estimated, rel_embs, dim=1)  # shape: [N]

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

    # Compute accuracy
    accuracy = sum(scores) / len(scores) if len(scores) > 0 else 0.0

    # Save accuracy to JSON file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # Use a unique key for each call (e.g., use dataset name or timestamp)
    results[f"run_{len(results)+1}"] = accuracy

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    return accuracy



def calculate_validation(clip_model: CLIPTextModelWithProjection, text_encoder, tokenizer, phi_for_eval,VRDDataset_test , relation_to_tokens_test, save_path, args,accelerator):
    
    scores = []
    scores2 = []
    relation_embeddings = {}
    
    for r , tokens  in relation_to_tokens_test.items():

        #input_ids = tokens.to(clip_model.text_model.embeddings.token_embedding.weight.device)
        input_ids = tokens.unsqueeze(0).to(
                    clip_model.text_model.embeddings.token_embedding.weight.device
                )#torch.Size([1, 77])
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
        relation_embeddings[r] = cls #(batch_size, embed_dim) → (1, 768)
    results = []
    # Loop over test dataset
    for imgTest, targetTest in VRDDataset_test:
        relations = targetTest.get("relations_text", [])
        
        for s, r, o in relations:
            triplet_string = f"{s} {r} {o}"
            tokenized_triplet = tokenizer(
                triplet_string,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77
            )["input_ids"].to(accelerator.device)#torch.Size([1, 77])

            # Encode triplet after the transformer and projection
            org = text_encoder(input_ids=tokenized_triplet) #torch.Size([1, 77])
            input_features = org.text_embeds.clone()
            if args.l2_normalize:
                input_features = F.normalize(input_features, dim=-1)#[1, 768]

            # Phi prediction
            #estimated_token_embeddings = phi_for_eval(input_features).squeeze(0)  # shape [dim]
            estimated_token_embeddings = phi_for_eval(input_features) #estimated_token_embeddings , input_features=torch.Size([1, 768])
            #estimated_token_embeddings = torch.vstack(estimated_token_embeddings)
            estimated_token_embeddings = estimated_token_embeddings.to(accelerator.device)#[1, 768]
            estimated_token_embeddings = estimated_token_embeddings.type(clip_model.dtype)
            estimated_token_embeddings = estimated_token_embeddings + clip_model.text_model.embeddings.position_embedding(clip_model.text_model.embeddings.position_ids) #torch.Size([1, 77, 768])
            _causal_attention_mask2 = _make_causal_mask(torch.Size([1, 77]), estimated_token_embeddings.dtype, device=estimated_token_embeddings.device)
            x2 = clip_model.text_model.encoder(inputs_embeds=estimated_token_embeddings,
                                        attention_mask=None,
                                        causal_attention_mask=_causal_attention_mask2,
                                        output_attentions=False,
                                        output_hidden_states=False,
                                        return_dict=False) #(batch_size, seq_len, embed_dim) →(1, 77, 768)
            x2 = x2[0] #torch.Size([1, 77, 768])
            x_last2 = clip_model.text_model.final_layer_norm(x2)#torch.Size([1, 77, 768])
            x2 = x_last2[torch.arange(x_last2.shape[0], device=x_last2.device),
                estimated_token_embeddings.to(dtype=torch.int, device=x_last2.device).argmax(dim=-1).argmax(dim=-1),
                ]
            if hasattr(clip_model, 'text_projection'):
                x2 = clip_model.text_projection(x2)
            cls2 = x2 #torch.Size([1, 768])
            estimated_token_embeddings = cls2
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
            estimated = estimated = estimated_token_embeddings.unsqueeze(0).expand(rel_embs.size(0), 1, -1)
            # Compute cosine similarities / after f.cosine -> [305, 1] — one cosine value per (batch, pair)/ after squeeze-> [305]
            cos_sims = F.cosine_similarity(estimated, rel_embs, dim=2).squeeze(1) 

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

    # Compute accuracy
    accuracy = sum(scores) / len(scores) if len(scores) > 0 else 0.0

    # Save accuracy to JSON file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # Use a unique key for each call (e.g., use dataset name or timestamp)
    results[f"run_{len(results)+1}"] = accuracy

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    return accuracy

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

    # Use a unique key for each call (e.g., use dataset name or timestamp)
    results[f"run_{len(results)+1}"] = accuracy

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    return accuracy