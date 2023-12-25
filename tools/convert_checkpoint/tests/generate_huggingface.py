import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)



def main(args):
    assert args.load_fp16 + args.load_8bit + args.load_4bit <= 1, "Only one of --load_fp16, --load_8bit, --load_4bit can be specified."
    kwargs = {}

    if args.model_parallel:
        assert args.use_gpu, "Model parallel is only supported on GPU."
        kwargs["device_map"] = "auto"
    if args.load_fp16:
        assert args.use_gpu, "FP16 is only supported on GPU."
        kwargs["torch_dtype"] = torch.float16
    elif args.load_8bit:
        assert args.use_gpu, "8-bit quantization is only supported on GPU."
        kwargs["load_in_8bit"] = True
    elif args.load_4bit:
        assert args.use_gpu, "4-bit quantization is only supported on GPU."
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["quantization_config"] = quantization_config


        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        **kwargs
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.eval()
    
    if args.lora:
        peft_config = LoraConfig(
            r=2,
            target_modules=["c_attn"],
            lora_alpha=2.0,
            bias="none",
            inference_mode=False,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        
    device = model.transformer.wte.weight.device
    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    position_ids = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.bool, device=device)
    
    output = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask
    )
    from IPython import embed; embed()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_parallel", action="store_true")
    parser.add_argument("--load_fp16", action="store_true")
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_gpu", action='store_true')
    args = parser.parse_args()
    main(args)
