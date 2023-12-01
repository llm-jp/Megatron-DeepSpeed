import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Convert from (seq_len, num_layer, batch_size, hidden_size) to (num_layers, batch_size, seq_len, hidden_size)
def reorder_hidden_states(generate_hidden_states):
    states_list = []
    for state_per_token in generate_hidden_states:
        state_per_token_list = []
        for state_per_layer in state_per_token:
            state_per_token_list.append(
                state_per_layer.view(
                    state_per_layer.size(0),
                    -1,
                )
            )
        states_list.append(
            torch.stack(state_per_token_list, dim=0)
        )
        
    reordered_states = torch.stack(states_list, dim=0)
    return reordered_states.permute(1, 2, 0, 3).contiguous()


@torch.no_grad()
def main(args):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    input_ids = torch.tensor([[616], [128]], dtype=torch.long)
    # position_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    
    model.eval()
    generate_output_use_cache = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=10,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    generate_hidden_states = reorder_hidden_states(generate_output_use_cache.hidden_states)
    decoded_ids = generate_output_use_cache.sequences[:, :-1]
    attention_mask = torch.ones_like(decoded_ids)
    
    output = model(
        input_ids=decoded_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hidden_states = torch.stack(output.hidden_states, dim=0)
    
    # Compare output ids from generate() and forward()
    is_same = (output.logits.argmax(-1)[:, :-1] == decoded_ids[:, 1:]).all()
    assert is_same, "output ids from generate() and forward() are different"
    print("=" * 30)
    print(f"Matched output ids from generate() and forward(): {is_same}")
    print("Hidden states mean (forward): ", hidden_states.mean().item())
    print("Hidden states mean (generate): ", generate_hidden_states.mean().item())
    print("Hidden states max (forward): ", hidden_states.max().item())
    print("Hidden states max (generate): ", generate_hidden_states.max().item())
    print("Hidden states min (forward): ", hidden_states.min().item())
    print("Hidden states min (generate): ", generate_hidden_states.min().item())
    print("Max diff: ", (generate_hidden_states - hidden_states).abs().max().item())
    print("=" * 30)
    
    
    
    from IPython import embed; embed()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
