# LLM-JP-GPT converter
## Convert to huggingface transformers
```bash
MDS_MODEL_DIR=/path/to/model/of/megatron_deepspeed_format
OUTPUT_DIR=/path/to/output/dir/of/huggingface_format
zsh ./scripts/convert_to_transformers.sh $MDS_MODEL_DIR $OUTPUT_DIR
```
**Do not forget to edit `rotary_ndims` in generated `config.json` when you use rotary position embedding.** It is not automatically done by this script.




## Inference huggingface
```bash
HF_MODEL_DIR=/path/to/model/of/huggingface_formats
zsh ./scripts/generate_from_hugggingface_ckpt.sh $HF_MODEL_DIR
```
Check the content of ~output~ variable in the ipython interactive shell manually. 
(i.e. compare the value of logits with the one in the megatron format)

```bash
HF_MODEL_DIR=/path/to/model/of/huggingface_formats
zsh ./scripts/inference_validity_check.py $HF_MODEL_DIR
```


## Convert to megatron
```bash
MDS_MODEL_DIR=/path/to/model/of/megatron_deepspeed_format
OUTPUT_DIR=/path/to/output/dir/of/megatron_format
zsh ./scripts/convert_to_megatron.sh $MDS_MODEL_DIR $OUTPUT_DIR
```

## Infernce megatron
```bash
MEGATRON_MODEL_DIR=/path/to/model/of/megatron_format
TOKENIZER_MODEL_PATH=/path/to/tokenizer/model
zsh ./scripts/generate_from_megatron_ckpt.sh $MEGATRON_MODEL_DIR
```
