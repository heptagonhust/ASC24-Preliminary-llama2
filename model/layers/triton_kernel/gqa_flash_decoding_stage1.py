import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage1(
    block_batch_ids, block_start_indexes,
    Q, K, V, sm_scale, Req_to_tokens, B_req_idx, B_Seqlen,
    Mid_O, # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum, #[batch, head, seq_block_num]
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    gqa_group_size,
    Q_HEAD_NUM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_block_id = tl.program_id(0)
    cur_kv_head = tl.program_id(1)

    cur_batch = tl.load(block_batch_ids + cur_block_id)
    seq_start_index = tl.load(block_start_indexes + cur_block_id)

    cur_q_head_offs = tl.arange(0, Q_HEAD_NUM)
    cur_q_head_range = cur_kv_head * gqa_group_size + cur_q_head_offs
    
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = seq_start_index
    cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)

    off_q = cur_batch * stride_qbs + cur_q_head_range[:, None] * stride_qh + offs_d[None, :]
    
    block_n_size = tl.where(cur_batch_end_index - cur_batch_start_index <= 0, 0, cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1) // BLOCK_N
    
    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)
    
    q = tl.load(Q + off_q, mask=cur_q_head_range[:, None] < (cur_kv_head + 1) * gqa_group_size, other=0.0)

    sum_exp = tl.zeros([Q_HEAD_NUM], dtype=tl.float32)
    max_logic = tl.zeros([Q_HEAD_NUM], dtype=tl.float32) - float("inf")
    acc = tl.zeros([Q_HEAD_NUM, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx +  offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        off_k = k_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
        k = tl.load(K + off_k, mask=offs_n_new[None, :] < cur_batch_end_index, other=0.0)
        att_value = tl.dot(q, k)
        att_value *= sm_scale
        att_value = tl.where(offs_n_new[None, :] < cur_batch_end_index, att_value, float("-inf"))
        v = tl.load(V + k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :], mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        
        cur_max_logic = tl.max(att_value, axis=1)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic[:, None])
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale[:, None]
        acc += tl.dot(exp_logic.to(v.dtype), v)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=1)
        max_logic = new_max_logic
    
    need_store = tl.where(block_n_size == 0, 0, 1)
    for _ in range(0, need_store, 1):
        seq_block_index = cur_batch_start_index // BLOCK_SEQ
        off_mid_o = cur_batch * stride_mid_ob + cur_q_head_range[:, None] * stride_mid_oh + seq_block_index * stride_mid_os + offs_d[None, :]
        off_mid_o_logexpsum = cur_batch * stride_mid_o_eb + cur_q_head_range * stride_mid_o_eh + seq_block_index
        tl.store(Mid_O + off_mid_o, acc / sum_exp[:, None], mask=cur_q_head_range[:, None] < (cur_kv_head + 1) * gqa_group_size)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp), mask=cur_q_head_range < (cur_kv_head + 1) * gqa_group_size)
    return


@torch.no_grad()
def flash_decode_stage1(block_batch_ids, block_start_indexes, q, k, v, Req_to_tokens, B_req_idx, B_seq_len, mid_out, mid_out_logsumexp, block_seq):
    BLOCK_SEQ = block_seq
    BLOCK_N = 16
    assert BLOCK_SEQ % BLOCK_N == 0
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk ** 0.5)
    batch, kv_head_num = B_req_idx.shape[0], k.shape[1]
    block_nums = len(block_batch_ids)
    grid = (block_nums, kv_head_num)
    gqa_group_size = q.shape[1] // k.shape[1]
    
    _fwd_kernel_flash_decode_stage1[grid](
        block_batch_ids, block_start_indexes,
        q, k, v, sm_scale, Req_to_tokens, B_req_idx, B_seq_len,
        mid_out,
        mid_out_logsumexp,
        Req_to_tokens.stride(0), Req_to_tokens.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_out_logsumexp.stride(0), mid_out_logsumexp.stride(1), mid_out_logsumexp.stride(2),
        gqa_group_size,
        Q_HEAD_NUM=max(16, triton.next_power_of_2(gqa_group_size)),
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK_N,
        num_warps=2,
        num_stages=2,
    )
    return