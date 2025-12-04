'''
LinCIR
Copyright (c) 2023-present NAVER Corp.
CC BY-NC-4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
'''
import json
import os
import pickle
import random
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Tuple, Dict, List, Set
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from loader import build_loader , create_relation_to_tokens_test
from encode_with_pseudo_tokens import encode_with_pseudo_tokens_HF , extract_pseudo_tokens_with_phi , calculate_validation2
from models import build_text_encoder, Phi, EMAModel


import transformers
from transformers import get_scheduler
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
################################
from VRDDataset import createDataset

logger = get_logger(__name__)


def parse_args():
    parser = ArgumentParser()
    ######################
    parser.add_argument("--mode", default="train", type=str,
                        help="Whether to train a new model or only evaluate the latest checkpoint")
    ######################
    parser.add_argument("--output_dir", default="trained_models", type=str,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--logging_dir", default="logs", type=str, help="tensorboard logs will saved here")
    parser.add_argument("--cache_dir", default="/data/reyDataset/hf_models", type=str,
                        help="Path to model cache folder")
    parser.add_argument("--report_to", default="tensorboard", type=str, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument("--clip_model_name", default="giga", type=str,
                        help="CLIP model to use, e.g 'large', 'giga'")
    #########################################################################
    parser.add_argument("--cirr_dataset_path", type=str, help="Path to CIRR dataset", required=True)
    parser.add_argument("--keywords_path", type=str, help="Path to keywords json file")
    parser.add_argument("--resume", default=None, type=str, help="Path to pretrained ckpt")
    #########################################################################
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--max_train_steps", type=int, default=50000, help="Total number of training steps to perform")
    parser.add_argument("--phi_dropout", default=0.5, type=float, help="Dropout probability for the phi network")
    parser.add_argument("--l2_normalize", action="store_true", help="Whether or not to use l2 normalization")
    parser.add_argument("--batch_size", default=256, type=int, help="Phi training batch size")
    parser.add_argument("--num_workers", default=10, type=int, help="Number of workers")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--max_grad_norm", default=None, type=float, help="Max gradient norm.")
    parser.add_argument("--mixed_precision", default=None, type=str, choices=["no", "fp16", "bf16"], help="mixed precision")
    parser.add_argument("--validation_steps", default=None, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--checkpointing_steps", default=None, type=int, help="Save a checkpoint of the training state every X updates")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")

    parser.add_argument("--seed", type=int, default=None, help="seed for reproducibility")

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def save_phi(name: str, cur_epoch: int, model_to_save: Phi, training_path: Path) -> None:
    """
    Save the weights of Phi during training
    """
    models_path = os.path.join(training_path, "checkpoints")
    os.makedirs(models_path, exist_ok=True)
    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': cur_epoch,
        model_name: model_to_save.state_dict(),
    }, os.path.join(models_path, f'{name}.pt'))


def train_phi(args):
    # We are going to use the pre-extracted clip image features. so we do not need image_encoder anymore.

    ### init accelerator here
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_dir=logging_dir,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    ### Define the text encoder from clip
    image_encoder, clip_preprocess, text_encoder, tokenizer = build_text_encoder(args)

    ### Define the phi model
    phi = Phi(input_dim=text_encoder.config.projection_dim,
                    hidden_dim=text_encoder.config.projection_dim * 4,
                    output_dim=text_encoder.config.hidden_size, dropout=args.phi_dropout)

    if args.resume:

        # it is the path of model with checkpoints to continue training at that moment
        phi.load_state_dict(
                torch.load(args.resume, map_location=accelerator.device)[
                phi.__class__.__name__])


    ### GPU handling
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    image_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    image_encoder.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.use_ema:
        import copy
        ema_phi = copy.deepcopy(phi)
        ema_phi = EMAModel(ema_phi.parameters())
        ema_phi.to(accelerator.device, dtype=weight_dtype)

    ### Define the train datasets
    '''
    load datasets of only captions and create 
    '''
    print('pytorch loader')
    if args.mode == "train":
        VRDDataset_train = createDataset(root='/data/reyDataset/vrd_kaggle/vrd' ,split='train',vocab='', save_vocab='/root/reyhane/CIR/vocab.json', tokenizer=tokenizer, ratio=1 )
        train_dataset = build_loader(args, tokenizer ,VRDDataset_train )
    VRDDataset_test  = createDataset(root='/data/reyDataset/vrd_kaggle/vrd' ,split='test',vocab='/root/reyhane/CIR/vocab.json', save_vocab='', tokenizer=tokenizer, ratio=1 )
    relation_to_tokens_test  = create_relation_to_tokens_test(VRDDataset_test , tokenizer)
    print('relation_to_tokens_test loaded successfully!!!!!!')
   
    # Define the optimizer, the loss and the grad scaler
    '''
    optimizer --> Updates weights using gradients -->	Learning happens
    scheduler --> Changes learning rate during training -->	Helps better and stable training
    Accelerate--> Makes multi-GPU, FP16, distributed easy -->	Clean and scalable training

    '''
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(phi.parameters(),
                              lr=args.learning_rate,
                              weight_decay=args.weight_decay)

    lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps * accelerator.num_processes,
    )
    if args.mode == "train":
        phi, optimizer, lr_scheduler, train_dataset = accelerator.prepare(
                phi, optimizer, lr_scheduler, train_dataset
        )
    if args.mode == "eval":
        phi, optimizer, lr_scheduler = accelerator.prepare(
                phi, optimizer, lr_scheduler
        )

    if accelerator.is_main_process:
        accelerator.init_trackers("zeroshot-cir", config=vars(args))
    ###############################
    # output of the text embedding
    '''
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
        print(emb)
        print(emb.size())
        exit(1)
    
    '''
    
    ################################
    # Start with the training loop
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total steps = {args.max_train_steps}")
    ################################TODO
    if args.mode == "eval":
        phi.to(accelerator.device)
        phi.eval()
        accuracy = calculate_validation2(text_encoder,text_encoder , tokenizer, phi,VRDDataset_test , relation_to_tokens_test
                                                       ,save_path='/root/reyhane/CIR/results/accuracy_scores.json' ,args= args, accelerator=accelerator)
        print(accuracy)
    #################################
    if args.mode == "train":
        phi.train()

        train_loss = 0.0
        global_step = 0
        '''
        is_local_main_process ? 
        Only show the tqdm progress bar if this process is the main one locally.
        To avoid having multiple progress bars from multiple GPUs flooding the screen.
        '''
        progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        while True:
            '''
            original_tokens : consist of tokens from origin prompts
            replaced_tokens : consist of tokens contain [$] 
            '''
            for idx, (original_tokens, replaced_tokens, indicators) in enumerate(train_dataset):
                original_tokens = original_tokens.to(accelerator.device)
                replaced_tokens = replaced_tokens.to(accelerator.device)
                #encode the original_tokens  from prompts
                # encode grand truth by encoder 
                org = text_encoder(input_ids=original_tokens)
                original_text_embeddings, original_last_hidden_states = org.text_embeds, org.last_hidden_state
                input_features = original_text_embeddings.clone()
                # add noise 
                '''
                ε_i = α_i * N(0, 1)
                α_i ∈ [0,1] is random per sample (controls noise strength),
                N(0,1) is standard Gaussian noise over D dimensions.
                '''
                input_features += 1.0 * torch.rand(input_features.shape[0], device=input_features.device).unsqueeze(-1) * torch.randn(input_features.shape, device=input_features.device)

                # normalize test
                if args.l2_normalize:
                    input_features = F.normalize(input_features, dim=-1)
                #################
                #Predict pseudo-token embeddings
                estimated_token_embeddings = phi(input_features)
                '''
                1) first of all it embedds the replaced_tokens using text encoder
                Returns [batch_size, seq_len, hidden_dim] → the normal word embeddings
                2) For any token in text(replaced_tokens) that has ID 259, replace its embedding with pseudo_tokens[i].
                3) add positional embedding because the transformers do not know the positions
                    This adds a unique vector for each position in the sentence:

                    0th token → add vector P₀

                    1st token → add vector P₁

                    2nd token → add vector P₂
                4) Feeds the modified embeddings into CLIP’s transformer
                5) Normalize final layer
                x_last: This is the last hidden state before projection.
                6) pooling : For each token (word), the model has a vector.
                    But we need just ONE vector per sentence to represent the whole thing.
                    pool the sequence of token embeddings into one vector per sentence.
                7)  projects into CLIP’s shared embedding space
                8)  outputes:
                    x: the final embedding used for comparison/loss
                    x_last: all hidden states (useful for loss inspection or token-wise analysis)
                '''
                replaced_text_embeddings, replaced_last_hidden_states = encode_with_pseudo_tokens_HF(text_encoder, replaced_tokens, estimated_token_embeddings, return_last_states=True)

                loss = F.mse_loss(replaced_text_embeddings.float(), original_text_embeddings.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagation and optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(phi.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                    if args.use_ema:
                        ema_phi.step(phi.module.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    '''
                    Logs training loss

                    Logs learning rate

                    Logs ratio of examples with real token replaced
                    '''
                    accelerator.log({"train/train_loss": train_loss}, step=global_step)
                    train_loss = 0.0
                    accelerator.log({'train/lr': lr_scheduler.get_last_lr()[0]}, step=global_step)
                    accelerator.log({'train/preproc_rate': torch.sum(indicators).item() / len(indicators)}, step=global_step)
                    '''
                    If step is a multiple of args.checkpointing_steps, it saves:

                    phi_latest

                    phi_<step>

                    EMA (Exponential Moving Average) versions if enabled
                    '''
                    if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            logger.info(f"model saving... step: {global_step}")
                            save_phi(f"phi_{global_step:09}", global_step, accelerator.unwrap_model(phi), args.output_dir)
                            save_phi(f"phi_latest", global_step, accelerator.unwrap_model(phi), args.output_dir)
                        if args.use_ema:
                            phi_for_saving = copy.deepcopy(accelerator.unwrap_model(phi))
                            ema_phi.copy_to(phi_for_saving.parameters())
                            save_phi(f"ema_phi_{global_step:09}", global_step, phi_for_saving, args.output_dir)
                            save_phi(f"ema_phi_latest", global_step, phi_for_saving, args.output_dir)
                
                    #######################################3
                    if args.validation_steps and (global_step % args.validation_steps == 0 or global_step == 50 ):
                        if accelerator.is_main_process:
                            logger.info(f"evaluate model... step: {global_step}")

                            if args.use_ema:
                                phi_for_eval = copy.deepcopy(accelerator.unwrap_model(phi))
                                ema_phi.copy_to(phi_for_eval.parameters())
                            else:
                                phi_for_eval = phi

                            phi_for_eval.eval()

                            # Extract the pseudo tokens for the VRD test set using Phi
                            accuracy = calculate_validation2(text_encoder,text_encoder , tokenizer, phi_for_eval,VRDDataset_test , relation_to_tokens_test
                                                        ,save_path='/root/reyhane/CIR/results/accuracy_scores.json' ,args= args, accelerator=accelerator)
                            # accuracy = calculate_validation2(text_encoder,text_encoder , tokenizer, phi,VRDDataset_test , relation_to_tokens_test
                            #                            ,save_path='/root/reyhane/CIR/results/accuracy_scores.json' ,args= args, accelerator=accelerator)
                            print(accuracy)
                            phi.train()

                if global_step >= args.max_train_steps:
                    exit()



if __name__ == '__main__':
    args = parse_args()

    train_phi(args)

