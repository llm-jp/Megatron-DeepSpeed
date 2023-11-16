import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional



import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.text_generation_utils import generate_and_write_samples_unconditional
from megatron.text_generation_utils import generate_samples_input_from_file
from megatron.text_generation_utils import generate_samples_interactive
import deepspeed
import torch

from megatron.arguments import core_transformer_config_from_args
from megatron import get_args
from megatron.arguments import (parse_args, validate_args)
from megatron.global_vars import set_global_variables

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()
    config = core_transformer_config_from_args(args)
    print_rank_0('building GPT model ...')
    model = GPTModel(config=config, num_tokentypes=0, parallel_output=False,
                     pre_process=pre_process, post_process=post_process,
                     return_moe_loss=False) # we need to set "return_moe_loss" for the inference_mode
    return model



def main():
    args_defaults = {}
    args = parse_args()

    
    # if args.use_checkpoint_args or args_defaults.get('use_checkpoint_args', False):
    #     assert args.load is not None, '--use-checkpoints-args requires --load argument'
    #     load_args_from_checkpoint(args)

    validate_args(args, args_defaults)
    set_global_variables(args)
        
    args = get_args()
    model = model_provider()
    
    if args.load is not None:
        out = load_checkpoint(model, None, None)
    breakpoint()
        

if __name__ == "__main__":
    main()
