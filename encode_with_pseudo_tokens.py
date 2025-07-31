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

def extract_pseudo_tokens_with_phi(text_encoder, tokenizer, phi_for_eval,VRDDataset_test , relation_to_tokens_test, save_path, args,accelerator):
    scores = []

    # Precompute relation embeddings (so we don't recompute for every image)
    relation_embeddings = {}
    for rel, token_ids in relation_to_tokens_test.items():

        tokens = torch.tensor(token_ids).unsqueeze(0).to(accelerator.device)  # shape [1, seq_len]
        # TODO: Compare with the tokenized text, not embedded text!
        with torch.no_grad():
            out = text_encoder(input_ids=tokens)
            emb = out.text_embeds.squeeze(0)  # shape [dim]
            if args.l2_normalize:
                emb = F.normalize(emb, dim=-1)
        relation_embeddings[rel] = emb
    
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
            )["input_ids"].to(accelerator.device)

            # Encode triplet
            org = text_encoder(input_ids=tokenized_triplet)
            input_features = org.text_embeds.clone()
            if args.l2_normalize:
                input_features = F.normalize(input_features, dim=-1)

            # Phi prediction
            #estimated_token_embeddings = phi_for_eval(input_features).squeeze(0)  # shape [dim]
            estimated_token_embeddings = phi_for_eval(input_features)
            if estimated_token_embeddings.dim() == 2 and estimated_token_embeddings.size(0) == 1:
                estimated_token_embeddings = estimated_token_embeddings.squeeze(0)
            elif estimated_token_embeddings.dim() == 1:
                pass  # already in shape [dim]
            else:
                raise ValueError(f"Unexpected shape from phi: {estimated_token_embeddings.shape}")
            '''
            print("Type:", type(estimated_token_embeddings))
            print("Shape:", estimated_token_embeddings.shape)
            print("Dtype:", estimated_token_embeddings.dtype)
            print("Device:", estimated_token_embeddings.device)
            print("Example values:", estimated_token_embeddings[:5])  # first 5 numbers
            '''
            # Compute cosine similarity with all relation embeddings
            best_rel = None
            best_score = -1
            # TODO: Use vectorized operations
            for rel_text, rel_emb in relation_embeddings.items():
                cos_sim = F.cosine_similarity(
                    estimated_token_embeddings.unsqueeze(0),
                    rel_emb.unsqueeze(0)
                ).item()
                if cos_sim > best_score:
                    best_score = cos_sim
                    best_rel = rel_text

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
'''
    for imgTest, targetTest in VRDDataset_test:
        relations = targetTest.get('relations_text', [])
        for s, r, o in relations :
            
            triplet_string = f"{s} {r} {o}"
            tokenized_triplet= tokenizer(triplet_string, return_tensors='pt', padding='max_length', truncation=True, max_length=77)['input_ids'][0]
            
            ##################
            tokenized_triplet = tokenized_triplet.to(args.accelerator.device)
            
    
            org = text_encoder(input_ids=tokenized_triplet)
            original_text_embeddings, original_last_hidden_states = org.text_embeds, org.last_hidden_state
            input_features = original_text_embeddings.clone()

            
            if args.l2_normalize:
                input_features = F.normalize(input_features, dim=-1)
            #################
            #Predict pseudo-token embeddings
            estimated_token_embeddings = phi_for_eval(input_features)
            ##################
'''     
