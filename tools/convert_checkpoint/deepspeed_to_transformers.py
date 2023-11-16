#!/usr/bin/env python

import os
import torch
import json
import shutil
import argparse
import tempfile
import glob


from deepspeed_checkpoint import DeepSpeedCheckpoint
from deepspeed_to_megatron import _create_rank_checkpoint

# the import was tested to work with this version
# https://github.com/huggingface/transformers/commit/0af901e83 if it diverges we may consider
# copying that version here instead
from convert_megatron_gpt2_checkpoint import convert_megatron_checkpoint
from transformers import AutoConfig
from huggingface_hub import snapshot_download


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, help='Input DeepSpeed Checkpoint folder', required=True)
    parser.add_argument('--output_folder', type=str, help='Output Megatron checkpoint folder', required=True)
    parser.add_argument('--target_tp', default=1, type=int, help='Target TP degree')
    parser.add_argument('--target_pp', default=1, type=int, help='Target PP degree')
    parser.add_argument('--for_release', action='store_true', help='Convert for release purpose, reset some (progress) counters.')
    parser.add_argument('--base_hf_model_name_or_path', type=str, help='Base HF model name or path', required=True)
    parser.add_argument('--temp_dir', type=str, help='Temp dir')
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
    print("Downloading base model")
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
        bos_token_id=50256,
        eos_token_id=50256,
    )

    # Convert.
    print("Converting to HF Checkpoint")
    output_state_dict = convert_megatron_checkpoint(args, input_state_dict, config)
    
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
    
    # output_config["architectures"] = ["MyGPT2LMHeadModel"]
    # output_config["model_type"] = "my_gpt2"
    
    # output_config["auto_map"] = {
    #     "AutoConfig": "configuration_my_gpt2.MyGPT2Config",
    #     "AutoModel": "modeling_my_gpt2.MyGPT2Model",
    #     "AutoModelForSequenceClassification": "modeling_my_gpt2.MyGPT2ForSequenceClassification",
    #     "AutoModelForTokenClassification": "modeling_my_gpt2.MyGPT2ForTokenClassification",
    #     "AutoModelForQuestionAnswering": "modeling_my_gpt2.MyGPT2ForQuestionAnswering",
    #     "AutoModelForCausalLM": "modeling_my_gpt2.MyGPT2LMHeadModel"
    # }
    
    print(f'Saving config to "{output_config_file}"')
    with open(output_config_file, "w") as f:
        json.dump(output_config, f)

    # Store the state_dict to file.
    output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)
    
    # print("Now add tokenizer files and upload to the hub")

    print("Now add remote code")
    # remote_code_dir = os.path.expanduser(
    #     "~/lab/llm-jp/convert_to_transformers/experiments/dev/my_gpt"
    # )
    # shutil.copy(
    #     os.path.join(remote_code_dir, "modeling_my_gpt2.py"),
    #     os.path.join(basename, "modeling_my_gpt2.py")
    # )
    # shutil.copy(
    #     os.path.join(remote_code_dir, "configuration_my_gpt2.py"),
    #     os.path.join(basename, "configuration_my_gpt2.py")
    # )
    
    # Copy all python scripts in the temp dir to the output dir.
    for path in glob.glob(os.path.join(temp_dir, "*.py")):
        shutil.copy(path, basename)
    # Copy generation_config.json to the output dir.
    shutil.copy(os.path.join(temp_dir, "generation_config.json"), basename)
    shutil.rmtree(temp_dir)
    
    
if __name__ == "__main__":
    main()
