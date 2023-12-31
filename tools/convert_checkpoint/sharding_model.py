import sys

import torch
from transformers import AutoModelForCausalLM


def main():
  src = sys.argv[1]
  dst = sys.argv[2]
  model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16, trust_remote_code=True)
  model.save_pretrained(dst)


if __name__ == "__main__":
  main()
