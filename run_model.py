import torch
import torch.nn as nn
from config import ModelConfig
import contextlib
from transformers import PretrainedConfig
from typing import Type,Optional
import importlib

class ModelRegistry:

    @staticmethod
    def load_model_cls(model_arch: str) -> Optional[Type[nn.Module]]:
        module_name, model_cls_name =  ("llama", "LlamaForCausalLM")
        module = importlib.import_module(
            f"vllm.model_executor.models.{module_name}")  # vllm.model_executor.models.llama
        return getattr(module, model_cls_name, None)

@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return model_cls
           
def get_model(model_config: ModelConfig) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_model_config)  # 从模型列表中找到与模型参数文件对应的模型

    # Get the (maybe quantized) linear method.
    linear_method = None

    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        with torch.device("cuda"):
            model = model_class(model_config.hf_model_config, linear_method)  # 在GPU中建立模型，同时根据TP将模型进行切分，因此开辟的显存空间是切分后的模型大小
            # Load the weights from the cached or downloaded files.
        model.load_weights(model_config.model, model_config.download_dir,
                            model_config.load_format, model_config.revision)
    return model.eval()