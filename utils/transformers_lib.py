from typing import Optional
from transformers import (AutoConfig, PretrainedConfig ,AutoModelForCausalLM, 
                          AutoTokenizer, PreTrainedTokenizerBase,LlamaConfig)

def get_config_from_hf(
    model: str,
    trust_remote_code: bool,
) -> all:
    return AutoConfig.from_pretrained(model, trust_remote_code = trust_remote_code)

def get_max_len_from_hf(
    hf_model_config: PretrainedConfig,
    max_model_len: Optional[int],
) -> int:
    max_len_hf = hf_model_config.max_length
    default_max_len = 2048
    
    if max_model_len is not None:
        if max_len_hf == float("inf"):
            return max_model_len

        if max_model_len <= 0:
            raise ValueError(
                f"User-specified max_model_len must be positive"
            )
        elif max_model_len > max_len_hf:
            raise ValueError(
                f"User-specified max_model_len is greater than size in config.json "
                f"({max_model_len} > {max_len_hf})."
            )
        else:
            max_len = min(max_len_hf, max_model_len)
            
    else:
        if max_len_hf == float("inf"):
            max_len = default_max_len
        else :
             max_len = max_len_hf
    return int(max_len)
    
            
            
    
        
    