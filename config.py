from utils.transformers_lib import get_config_from_hf, get_max_len_from_hf

class ModelConfig:
    def __init__(
        self,
        model: str,
        tokenizer: str,
        trust_remote_code: bool,
        seed: int,
        max_model_len: int
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.trust_remote_code = trust_remote_code
        self.seed = seed

        self.hf_model_config = get_config_from_hf(model, trust_remote_code)
        self.max_model_len = get_max_len_from_hf(self.hf_model_config, max_model_len)
        


