from dataclasses import dataclass

import torch

from vllm_fl.dispatch.backends.vendor.iluvatar.impl.vendor_flash_attn import (
    _build_split,
)


@dataclass
class Metadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    num_actual_tokens: int = 0
    max_query_len: int = 1


def _metadata(n):
    return Metadata(
        query_start_loc=torch.arange(n + 1, dtype=torch.int32),
        seq_lens=torch.arange(1, n + 1, dtype=torch.int32),
        block_table=torch.arange(n, dtype=torch.int32),
    )


def test_build_split_all_decode_reuses_metadata():
    m = _metadata(4)
    _build_split(m, [0, 1, 2, 3, 4], [11, 22, 33, 44])

    _, _, pre_idx, dec_rows, dec_meta = m._fl_split
    assert pre_idx == []
    assert torch.equal(dec_rows, torch.arange(4))
    assert dec_meta.seq_lens is m.seq_lens
    assert dec_meta.block_table is m.block_table
    assert dec_meta.num_actual_tokens == 4
    assert dec_meta.max_query_len == 1
    assert torch.equal(dec_meta.query_start_loc, torch.arange(5))


def test_build_split_mixed_batch_preserves_gather_semantics():
    m = _metadata(3)
    _build_split(m, [0, 1, 3, 4], [11, 22, 33])

    _, _, pre_idx, dec_rows, dec_meta = m._fl_split
    assert pre_idx == [1]
    assert torch.equal(dec_rows, torch.tensor([0, 3]))
    assert torch.equal(dec_meta.seq_lens, torch.tensor([11, 33]))
    assert torch.equal(dec_meta.block_table, torch.tensor([0, 2]))
    assert torch.equal(m._fl_cu_q, torch.tensor([[0, 2]], dtype=torch.int32))
    assert torch.equal(m._fl_cu_k, torch.tensor([[0, 22]], dtype=torch.int32))
