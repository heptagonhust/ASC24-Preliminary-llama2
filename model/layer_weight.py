import torch
import numpy as np


class BaseLayerWeight:
    def __init__(self):
        self.tp_rank_ = None

    def load_hf_weights(self, weights):
        """
        load weights
        """
        pass

    def init_static_params(self):
        """
        design for some static init params, many model dont need do this.
        """
        pass

    def verify_load(self):
        """
        verify all load is ok
        """
        raise Exception("must verify weights load ok")
        pass

    def _cuda(self, cpu_tensor):
        if self.tp_rank_ is None:
            return cpu_tensor.contiguous().to(self.data_type_).cuda()
        else:
            return cpu_tensor.contiguous().to(self.data_type_).cuda(self.tp_rank_)
        
class TransformerLayerWeight(BaseLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode):
        self.layer_num_ = layer_num
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.data_type_ = data_type
        self.network_config_ = network_config
        self.mode = mode
        self.init_static_params()
        return

class LlamaTransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        return

    def load_hf_weights(self, weights):
        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)
        return
    
    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.att_norm_weight_,
                   self.q_weight_,
                   self.k_weight_,
                   self.v_weight_,
                   self.o_weight_,
                   self.ffn_norm_weight_,
                   self.up_proj,
                   self.gate_proj,
                   self.down_proj
                   ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return 
    
    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"])

        n_embed = self.network_config_["hidden_size"]
        q_split_n_embed = n_embed // self.world_size_
        kv_split_n_embed = n_embed // self.network_config_["num_attention_heads"] * self.network_config_["num_key_value_heads"] // self.world_size_
        # q k v weights for llama
        if f"model.layers.{self.layer_num_}.self_attn.q_proj.weight" in weights:
            self.q_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"]
            self.q_weight_ = self.q_weight_[q_split_n_embed * self.tp_rank_: q_split_n_embed * (self.tp_rank_ + 1), :]
            self.q_weight_ = self._cuda(self.q_weight_.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.self_attn.k_proj.weight" in weights:
            self.k_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"]
            self.k_weight_ = self.k_weight_[kv_split_n_embed * self.tp_rank_: kv_split_n_embed * (self.tp_rank_ + 1), :]
            self.k_weight_ = self._cuda(self.k_weight_.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.self_attn.v_proj.weight" in weights:
            self.v_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"]
            self.v_weight_ = self.v_weight_[kv_split_n_embed * self.tp_rank_: kv_split_n_embed * (self.tp_rank_ + 1), :]
            self.v_weight_ = self._cuda(self.v_weight_.transpose(0, 1))
        
        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            self.o_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"]
            self.o_weight_ = self.o_weight_[:, q_split_n_embed * self.tp_rank_: q_split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_ = self._cuda(self.o_weight_.transpose(0, 1))
        return
    
    def _load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"])
    
        inter_size = self.network_config_['intermediate_size']
        split_inter_size = inter_size // self.world_size_

        if f"model.layers.{self.layer_num_}.mlp.up_proj.weight" in weights:
            self.up_proj = weights[f"model.layers.{self.layer_num_}.mlp.up_proj.weight"][split_inter_size *
                                                                                         self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            self.up_proj = self._cuda(self.up_proj.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.mlp.gate_proj.weight" in weights:
            self.gate_proj = weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.weight"][split_inter_size *
                                                                                             self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            self.gate_proj = self._cuda(self.gate_proj.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.mlp.down_proj.weight" in weights:
            self.down_proj = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.weight"][:,
                                                                                             split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.down_proj = self._cuda(self.down_proj.transpose(0, 1))
        return