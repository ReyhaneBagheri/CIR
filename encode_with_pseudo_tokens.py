'''
LinCIR
Copyright (c) 2023-present NAVER Corp.
CC BY-NC-4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
'''
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
    
    relation_embeddings = {}
    
    for r , tokens  in relation_to_tokens_test.items():

        input_ids = tokens.to(clip_model.text_model.embeddings.token_embedding.weight.device)
        emb = clip_model.text_model.embeddings.token_embedding(input_ids).type(clip_model.dtype)#create vector 77*768
        emb = emb + clip_model.text_model.embeddings.position_embedding(clip_model.text_model.embeddings.position_ids)
        _causal_attention_mask = _make_causal_mask(input_ids.shape, emb.dtype, device=emb.device)
        x = clip_model.text_model.encoder(inputs_embeds=emb,
                                      attention_mask=None,
                                      causal_attention_mask=_causal_attention_mask,
                                      output_attentions=False,
                                      output_hidden_states=False,
                                      return_dict=False) #(batch_size, seq_len, embed_dim) →(1, 77, 768)
        cls = x[:, 0, :] 
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
            org = text_encoder(input_ids=tokenized_triplet)
            input_features = org.text_embeds.clone()
            if args.l2_normalize:
                input_features = F.normalize(input_features, dim=-1)#[1, 768]

            # Phi prediction
            #estimated_token_embeddings = phi_for_eval(input_features).squeeze(0)  # shape [dim]
            estimated_token_embeddings = phi_for_eval(input_features) #torch.Size([1, 768])
            estimated_token_embeddings = torch.vstack(estimated_token_embeddings)
            estimated_token_embeddings = estimated_token_embeddings.to(accelerator.device)
            
            estimated_token_embeddings = estimated_token_embeddings + clip_model.text_model.embeddings.position_embedding(clip_model.text_model.embeddings.position_ids)
            #_causal_attention_mask = _make_causal_mask(input_ids.shape, estimated_token_embeddings.dtype, device=estimated_token_embeddings.device)
            x = clip_model.text_model.encoder(inputs_embeds=estimated_token_embeddings,
                                        attention_mask=None,
                                        causal_attention_mask=None,#_causal_attention_mask,
                                        output_attentions=False,
                                        output_hidden_states=False,
                                        return_dict=False) #(batch_size, seq_len, embed_dim) →(1, 77, 768)
            estimated_token_embeddings = x[:, 0, :] 
            
            
            
            
            # if estimated_token_embeddings.dim() == 2 and estimated_token_embeddings.size(0) == 1:
            #     estimated_token_embeddings = estimated_token_embeddings.squeeze(0)#torch.Size([768])
            # elif estimated_token_embeddings.dim() == 1:
            #     pass  # already in shape [dim]
            # else:
            #     raise ValueError(f"Unexpected shape from phi: {estimated_token_embeddings.shape}")
            
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






def extract_pseudo_tokens_with_phi2(
    clip_model: CLIPTextModelWithProjection,
    text_encoder,
    tokenizer,
    phi_for_eval,
    VRDDataset_test,
    relation_to_tokens_test,
    save_path,
    args,
    accelerator,
):
    import os, json
    import torch
    import torch.nn.functional as F

    device = accelerator.device
    scores = []

    # --- Precompute relation embeddings in CLIP space (safe shapes) ---
    rel_texts = list(relation_to_tokens_test.keys())
    rel_embs_clip = []

    for r in rel_texts:
        tokens = relation_to_tokens_test[r].to(device)  # might be 1D ([seq_len]) or 2D ([1, seq_len])
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)  # [1, seq_len]

        # build attention_mask in case tokens are padded (assumes pad_token_id available)
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            attention_mask = (tokens != tokenizer.pad_token_id).long().to(device)  # [1, seq_len]
        else:
            attention_mask = torch.ones_like(tokens).to(device)

        with torch.no_grad():
            # Use the text encoder to get final CLIP text embedding (text_embeds)
            out = text_encoder(input_ids=tokens, attention_mask=attention_mask)
            rel_emb = out.text_embeds  # [1, D]
            if args.l2_normalize:
                rel_emb = F.normalize(rel_emb, dim=-1)
            rel_embs_clip.append(rel_emb.squeeze(0))  # [D]

    rel_embs_clip = torch.stack(rel_embs_clip)  # [N_rel, D]

    # --- Loop over dataset ---
    for imgTest, targetTest in VRDDataset_test:
        relations = targetTest.get("relations_text", [])

        for s, r, o in relations:
            triplet_string = f"{s} {r} {o}"
            tokenized_triplet = tokenizer(
                triplet_string,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            )["input_ids"].to(device)  # [1, seq_len]

            # Encode triplet (full text encoder => CLIP space)
            org = text_encoder(input_ids=tokenized_triplet)
            input_features = org.text_embeds.clone()  # [1, D]
            if args.l2_normalize:
                input_features = F.normalize(input_features, dim=-1)

            # Phi prediction (your phi may output token-level or pooled)
            estimated_token_embeddings = phi_for_eval(input_features)  # could be [1,D], [D], or [seq_len, hidden]
            # Normalize shapes
            if estimated_token_embeddings.dim() == 2 and estimated_token_embeddings.size(0) == 1:
                estimated_token_embeddings = estimated_token_embeddings.squeeze(0)  # [D]
            # If phi returns [seq_len, hidden] (token-level), leave as-is and handle below.

            # --- Map phi output into CLIP embedding space ---
            # Case A: phi output is pooled vector ([D] or [1,D]) and already in CLIP space:
            #   If phi already returns embeddings aligned to CLIP space, use directly.
            # Case B: phi output is token-level embeddings (seq_len x hidden_dim) in token space:
            #   We must pass them through CLIP text model as inputs_embeds to get pooled CLIP embedding.
            clip_emb_output = None
            with torch.no_grad():
                if estimated_token_embeddings.dim() == 1:
                    # single pooled vector [D] — assume it's already in same dim as rel_embs_clip
                    # If phi is in token space (vocab) this is NOT correct; but your earlier indicated phi outputs shape [768]
                    clip_emb_output = estimated_token_embeddings.to(device)
                    if args.l2_normalize:
                        clip_emb_output = F.normalize(clip_emb_output, dim=-1)
                elif estimated_token_embeddings.dim() == 2:
                    # token-level embeddings: [seq_len, hidden_dim]
                    # make batch dim: [1, seq_len, hidden_dim]
                    phi_input_embeds = estimated_token_embeddings.unsqueeze(0).to(device)

                    # Build attention_mask for these token embeddings (assume non-padded if not provided)
                    seq_len = phi_input_embeds.size(1)
                    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
                        # no token ids here, so assume all tokens are valid (1s)
                        phi_attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
                    else:
                        phi_attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

                    # Pass embeddings through CLIP text model to get final pooled embedding
                    # text_model returns a ModelOutput with .text_embeds when using CLIPTextModelWithProjection
                    out_phi = clip_model.text_model(inputs_embeds=phi_input_embeds, attention_mask=phi_attention_mask)
                    # If using clip_model (wrapper), you might prefer clip_model.get_text_features but it expects token ids.
                    # out_phi.text_embeds is [1, D]
                    clip_emb_output = out_phi.text_embeds.squeeze(0)
                    if args.l2_normalize:
                        clip_emb_output = F.normalize(clip_emb_output, dim=-1)
                else:
                    raise ValueError(f"Unexpected phi output shape: {estimated_token_embeddings.shape}")

            # --- Compute cosine similarities in CLIP space ---
            # Ensure clip_emb_output is [D] and rel_embs_clip is [N, D]
            if clip_emb_output.dim() == 1:
                # expand and compare
                cos_sims = F.cosine_similarity(
                    clip_emb_output.unsqueeze(0).expand(rel_embs_clip.size(0), -1),
                    rel_embs_clip,
                    dim=1,
                )  # [N_rel]
            else:
                raise ValueError("clip_emb_output has unexpected dims")

            best_idx = torch.argmax(cos_sims).item()
            best_score = cos_sims[best_idx].item()
            best_rel = rel_texts[best_idx]

            scores.append(1 if best_rel == r else 0)

    # Compute accuracy
    accuracy = sum(scores) / len(scores) if len(scores) > 0 else 0.0

    # Save accuracy
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    results[f"run_{len(results)+1}"] = accuracy
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    return accuracy
