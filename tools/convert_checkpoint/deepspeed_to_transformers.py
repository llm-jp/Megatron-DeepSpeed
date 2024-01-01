#!/usr/bin/env python

import os
import torch
import json
import shutil
import argparse
import tempfile
import glob

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

from deepspeed_checkpoint import DeepSpeedCheckpoint
from deepspeed_to_megatron import _create_rank_checkpoint

# the import was tested to work with this version
# https://github.com/huggingface/transformers/commit/0af901e83 if it diverges we may consider
# copying that version here instead
from convert_megatron_gpt2_checkpoint import convert_megatron_checkpoint
from transformers import AutoConfig, modeling_utils
from huggingface_hub import snapshot_download
from safetensors.torch import save_file as safe_save_file


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, help='Input DeepSpeed Checkpoint folder', required=True)
    parser.add_argument('--output_folder', type=str, help='Output Megatron checkpoint folder', required=True)
    parser.add_argument('--target_tp', default=1, type=int, help='Target TP degree')
    parser.add_argument('--target_pp', default=1, type=int, help='Target PP degree')
    parser.add_argument('--for_release', action='store_true', help='Convert for release purpose, reset some (progress) counters.')
    parser.add_argument('--base_hf_model_name_or_path', type=str, help='Base HF model name or path', default=os.path.join(THIS_FILE_DIR, "llm-jp-gpt"))
    parser.add_argument('--temp_dir', type=str, help='Temp dir')
    parser.add_argument('--bos_token_id', type=int, default=7)
    parser.add_argument('--eos_token_id', type=int, default=7)
    args = parser.parse_args()
    print(f'args = {args}')
    return args

def main():

    # this first part comes mainly from deepspeed_to_megatron.main
    args = parse_arguments()
    print(f'Converting DeepSpeed checkpoint in {args.input_folder} to HF Transformers checkpoint in {args.output_folder}')

    ds_checkpoint = DeepSpeedCheckpoint(args.input_folder, args.target_tp, args.target_pp)
    iteration = ds_checkpoint.get_iteration()
    input_state_dict = _create_rank_checkpoint(ds_checkpoint, 0, 0, args.for_release)
    
    # the 2nd part comes from transformers.models.megatron_gpt2.convert_megatron_gpt2_checkpoint.main
    # Spell out all parameters in case the defaults change.

    if args.temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    else:
        temp_dir = args.temp_dir
        
    # Download the base model.
    print("Loading base model")
    if os.path.isdir(args.base_hf_model_name_or_path):
        for path in glob.glob(os.path.join(args.base_hf_model_name_or_path, "*.py")):
            shutil.copy(path, temp_dir)
        shutil.copy(
            os.path.join(args.base_hf_model_name_or_path, "config.json"),
            temp_dir
        )
        shutil.copy(
            os.path.join(args.base_hf_model_name_or_path, "generation_config.json"),
            temp_dir
        )
    else:
        snapshot_download(
            repo_id=args.base_hf_model_name_or_path,
            repo_type="model",
            local_dir=temp_dir,
            local_dir_use_symlinks=False,
            force_download=True,
            allow_patterns=["*.py"],
        )
        snapshot_download(
            repo_id=args.base_hf_model_name_or_path,
            repo_type="model",
            local_dir=temp_dir,
            local_dir_use_symlinks=False,
            force_download=True,
            allow_patterns=["config.json", "generation_config.json"],
        )
        
    base_condig = AutoConfig.from_pretrained(temp_dir, trust_remote_code=True)
    config_cls = base_condig.__class__
    
    config = config_cls(
        vocab_size=50257,
        n_positions=2048,
        n_ctx=1024,
        n_embd=1024,
        n_layer=24,
        n_head=16,
        n_inner=4096,
        activation_function="gelu",  # used to be "gelu_new" in earlier versions
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        gradient_checkpointing=False,
        use_cache=True,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
    )

    # Convert.
    print("Converting to HF Checkpoint")
    output_state_dict = convert_megatron_checkpoint(
        args, input_state_dict, config, not_transpose_linear_layer_weights=True
    )
    
    basename = args.output_folder
    os.makedirs(basename, exist_ok=True)
    
    # Print the structure of converted state dict.
    #if args.print_checkpoint_structure:
    #    recursive_print(None, output_state_dict)
    
    # Store the config to file.
    output_config_file = os.path.join(basename, "config.json")
    output_config = config.to_dict()

    output_config["architectures"] = base_condig.architectures
    output_config["model_type"] = base_condig.model_type
    output_config["auto_map"] = base_condig.auto_map
    
    
    print(f'Saving config to "{output_config_file}"')
    with open(output_config_file, "w") as f:
        json.dump(output_config, f, indent=4)
        
    # Store the state_dict to file.
    output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    shards, index = modeling_utils.shard_checkpoint(output_state_dict, max_shard_size="5GB")
    if index:
        for shard_file, shard in shards.items():
            torch.save(shard, os.path.join(output_checkpoint_file, shard_file))
        save_index_file = os.path.join(basename, "pytorch_model.bin.index.json")
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
    else:
        torch.save(output_state_dict, output_checkpoint_file)

    print("Now add remote code")
    # Copy all python scripts in the temp dir to the output dir.
    for path in glob.glob(os.path.join(temp_dir, "*.py")):
        shutil.copy(path, basename)
    # Copy generation_config.json to the output dir.
    shutil.copy(os.path.join(temp_dir, "generation_config.json"), basename)
    shutil.rmtree(temp_dir)
    print(f"Attention: You need to add tokenizer files to {basename}.")
    

if __name__ == "__main__":
    main()
