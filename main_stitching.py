#!/usr/bin/env python
# coding=utf-8
import argparse
import logging
import os
import datasets

import transformers
import sys
from utils.model_utils import get_model
from utils.dataset_utils import get_text_dataset
from utils.dataloader_utils import get_dataloader
from utils.tokenizer_utils import get_tokenizer, get_tokenizer3
import torch
from transformers import AutoTokenizer

from useless_layers.stitching_eval import stitched_performance_eval

from datasets import load_from_disk
import pickle

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--text_dataset",
        type=str,
        default=None,
        help="A csv or a json file containing the text dataset.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--out_filename", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Print every logging_steps samples processed.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="model_name.",
    )
    parser.add_argument(
        "--reps_dir",
        type=str,
        help="folder with representation for stitching.",
        default='./',
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    world_size = 1


    if args.model_name in ["llama2-7b", "llama2-13b"]:
        model = get_model(
            config_name=args.config_name,
            model_name_or_path=args.checkpoint_dir,
            precision=torch.bfloat16,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            use_flash_attn=args.use_flash_attn,
            logger=None,
        )

        tokenizer = get_tokenizer(
            tokenizer_path=args.tokenizer_name, model_path=args.checkpoint_dir
        )
    elif args.model_name == "llama3-8b":
        model = get_model(
            config_name=args.config_name,
            model_name_or_path=args.checkpoint_dir,
            precision=torch.bfloat16,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            use_flash_attn=args.use_flash_attn,
            logger=None,
        )

        tokenizer = get_tokenizer3(
            tokenizer_path=args.tokenizer_name, model_path=args.checkpoint_dir
        )
    else:
        print(args.model_name)
        sys.stdout.flush()
        raise NameError(
            f"{args.model_name} not supported. Possible values are: 'llama-2-7b'"
        )

    pad_token_id = tokenizer.pad_token_id
    n_layer = model.config.num_hidden_layers
    print("model loaded. \n\n")
    sys.stdout.flush()

    dataset = get_text_dataset(
        filepath=args.text_dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_len,
        num_processes=args.preprocessing_num_workers
    )


    dataloader = get_dataloader(
        dataset,
        args.batch_size,
        pad_token_id,
        max_seq_len=2048,
        world_size=world_size,
        shuffle=False,
        num_processes=args.preprocessing_num_workers,
    )
    # ***********************************************************************

    Ngpus = torch.cuda.device_count()
    # Put model into GPU
    model.to('cuda')

    nsamples = len(dataloader.dataset)
    print("num_total_samples", nsamples)

    indexes_high = range(4,33)
    index_min = 3
    nums_anchors = [15000]

    results = stitched_performance_eval(model=model, 
                              model_name=args.model_name,
                              nums_anchors=nums_anchors, 
                              index_min=index_min, 
                              indexes_high=indexes_high, 
                              dataloader=dataloader,
                              tokenizer=tokenizer,
                              representation_folder=args.reps_dir,
                              printout=True )

if __name__ == "__main__":
    main()
