import dataclasses

import torch

from vllm_fl.dispatch.backends.vendor.iluvatar.impl.vendor_flash_attn import (
    _build_split,
    _decode_index,
)


def test_decode_index_reuses_same_tensor():
    _decode_index.cache_clear()
    first = _decode_index((0, 2, 4), torch.device("cuda"))
    second = _decode_index((0, 2, 4), torch.device("cuda"))
    assert first is second
    assert first.tolist() == [0, 2, 4]


def test_build_split_uses_gpu_metadata():
    @dataclasses.dataclass
    class Meta:
        query_start_loc: torch.Tensor
        seq_lens: torch.Tensor
        block_table: torch.Tensor
        num_actual_tokens: int = 6
        max_query_len: int = 3

    m = Meta(
        query_start_loc=torch.tensor([0, 1, 3, 6], device="cuda", dtype=torch.int64),
        seq_lens=torch.tensor([4, 5, 6], device="cuda", dtype=torch.int32),
        block_table=torch.tensor([[1, 2], [3, 4], [5, 6]], device="cuda"),
    )
    _build_split(m, [0, 1, 3, 6], [4, 5, 6])
    cu_l, seq_l, pre_idx, dec_rows, dec_meta = m._fl_split
    assert pre_idx == [1, 2]
    assert dec_rows.tolist() == [0]
    assert dec_meta.seq_lens.tolist() == [4]
    assert dec_meta.block_table.tolist() == [[1, 2]]
    assert dec_meta.num_actual_tokens == 1
    assert dec_meta.max_query_len == 1
    assert dec_meta.query_start_loc.tolist() == [0, 1]
