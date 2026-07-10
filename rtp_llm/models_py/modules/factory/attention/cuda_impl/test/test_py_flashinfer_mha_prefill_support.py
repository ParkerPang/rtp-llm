import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPagedPrefillImpl,
    PyFlashinferPrefillImpl,
)
from rtp_llm.ops import RopeStyle
from rtp_llm.ops.fused_rope_kvcache_op import FusedRopeKVCachePrefillOpBase


class TestPyFlashinferPrefillSupport(unittest.TestCase):
    """Support, construction, and wiring-contract tests."""

    def _start_patch(self, target):
        patcher = mock.patch(target)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def _make_paged_impl(self, rope_params):
        fmha_cls = self._start_patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.PyFlashinferPrefillPagedAttnOp"
        )
        rope_cls = self._start_patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.FusedRopeKVCachePrefillOpQOut"
        )
        self._start_patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.rtp_llm_ops.FlashInferMlaAttnParams"
        )
        self._start_patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.common.create_write_cache_store_impl"
        )
        rope_cls.return_value.prepare.return_value = rope_params
        impl = PyFlashinferPagedPrefillImpl(mock.Mock(), mock.sentinel.attn_inputs)
        self.assertIs(impl.fmha_impl, fmha_cls.return_value)
        return impl

    def _make_ragged_impl(self):
        fmha_cls = self._start_patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.PyFlashinferPrefillAttnOp"
        )
        rope_cls = self._start_patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.FusedRopeKVCachePrefillOpQKVOut"
        )
        self._start_patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.rtp_llm_ops.FlashInferMlaAttnParams"
        )
        self._start_patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.common.create_write_cache_store_impl"
        )
        attn_configs = SimpleNamespace(rope_config=SimpleNamespace(style=RopeStyle.No))
        impl = PyFlashinferPrefillImpl(attn_configs, mock.sentinel.attn_inputs)
        self.assertIs(impl.fmha_impl, fmha_cls.return_value)
        self.assertIs(impl.rope_kvcache_impl, rope_cls.return_value)
        return impl

    @staticmethod
    def _make_rope_params(kv_cache_offset=None, **overrides):
        params = {
            "padding_offset": None,
            "position_ids": torch.tensor([0], dtype=torch.int32),
            "cu_seqlens": torch.tensor([0, 1], dtype=torch.int32),
            "input_lengths": torch.tensor([1], dtype=torch.int32),
            "prefix_lengths": torch.tensor([0], dtype=torch.int32),
            "kv_cache_offset": kv_cache_offset,
            "max_seq_len": 1,
            "max_prefix_length": 0,
        }
        params.update(overrides)
        return SimpleNamespace(**params)

    @mock.patch(
        "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.PyFlashinferPrefillPagedAttnOp.support",
        return_value=True,
    )
    @mock.patch(
        "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.is_sm10x",
        return_value=False,
    )
    def test_paged_rejects_disabled_rope_kv_cache_contract(self, _is_sm10x, _support):
        attn_configs = SimpleNamespace(
            need_rope_kv_cache=False,
            rope_config=SimpleNamespace(style=RopeStyle.Base),
        )
        self.assertFalse(
            PyFlashinferPagedPrefillImpl.support(attn_configs, mock.Mock())
        )

    @mock.patch(
        "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.PyFlashinferPrefillAttnOp.support",
        return_value=True,
    )
    def test_ragged_rejects_rope_with_disabled_rope_kv_cache_contract(self, _support):
        attn_configs = SimpleNamespace(
            need_rope_kv_cache=False,
            rope_config=SimpleNamespace(style=RopeStyle.Base),
        )
        self.assertFalse(PyFlashinferPrefillImpl.support(attn_configs, mock.Mock()))

    @mock.patch(
        "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.PyFlashinferPrefillAttnOp.support",
        return_value=True,
    )
    def test_ragged_accepts_encoder_without_rope_or_kv_cache(self, _support):
        attn_configs = SimpleNamespace(
            need_rope_kv_cache=False,
            rope_config=SimpleNamespace(style=RopeStyle.No),
        )
        self.assertTrue(PyFlashinferPrefillImpl.support(attn_configs, mock.Mock()))

    @mock.patch(
        "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.common.apply_write_cache_store"
    )
    def test_ragged_without_rope_or_kv_cache_forwards_qkv_unchanged(
        self, apply_write_cache_store
    ):
        impl = self._make_ragged_impl()
        impl.fmha_impl.forward.return_value = mock.sentinel.output
        impl.write_cache_store_impl = mock.sentinel.write_cache_store
        impl.attn_inputs = mock.sentinel.attn_inputs
        qkv = mock.sentinel.qkv

        output = impl.forward(qkv, kv_cache=None)

        impl.rope_kvcache_impl.forward.assert_not_called()
        apply_write_cache_store.assert_called_once_with(
            mock.sentinel.write_cache_store, mock.sentinel.attn_inputs, None
        )
        impl.fmha_impl.forward.assert_called_once_with(qkv, None)
        self.assertIs(output, mock.sentinel.output)

    @mock.patch(
        "rtp_llm.ops.fused_rope_kvcache_op.get_scalar_type",
        return_value=mock.sentinel.scalar_type,
    )
    def test_prepare_accepts_empty_prefix_lengths(self, _get_scalar_type):
        attn_inputs = SimpleNamespace(
            kv_cache_kernel_block_id=None,
            combo_position_ids=None,
            context_parallel_info=None,
            padding_offset=None,
            cu_seqlens_device=torch.tensor([0, 1], dtype=torch.int32),
            cu_kv_seqlens_device=torch.tensor([0, 1], dtype=torch.int32),
            input_lengths=torch.tensor([1], dtype=torch.int32),
            prefix_lengths=torch.empty(0, dtype=torch.int32),
            sequence_lengths=torch.tensor([1], dtype=torch.int32),
            context_total_kv_length=1,
            dtype=torch.bfloat16,
        )

        params = FusedRopeKVCachePrefillOpBase(mock.Mock()).prepare(attn_inputs)

        self.assertEqual(params.max_prefix_length, 0)

    def test_cuda_graph_replay_rejects_offset_availability_change(self):
        impl = self._make_paged_impl(self._make_rope_params())
        impl.rope_kvcache_impl.prepare.return_value = self._make_rope_params(
            kv_cache_offset=torch.tensor([0], dtype=torch.int32),
            position_ids=impl.rope_params.position_ids,
            cu_seqlens=impl.rope_params.cu_seqlens,
            input_lengths=impl.rope_params.input_lengths,
            prefix_lengths=impl.rope_params.prefix_lengths,
        )

        with self.assertRaisesRegex(RuntimeError, r"capture=False, replay=True"):
            impl.prepare_cuda_graph(mock.sentinel.attn_inputs)

    def test_cuda_graph_replay_rejects_replaced_rope_input_storage(self):
        impl = self._make_paged_impl(self._make_rope_params())
        impl.rope_kvcache_impl.prepare.return_value = self._make_rope_params(
            position_ids=impl.rope_params.position_ids.clone(),
            cu_seqlens=impl.rope_params.cu_seqlens,
            input_lengths=impl.rope_params.input_lengths,
            prefix_lengths=impl.rope_params.prefix_lengths,
        )

        with self.assertRaisesRegex(RuntimeError, r"storage for position_ids"):
            impl.prepare_cuda_graph(mock.sentinel.attn_inputs)

    def test_cuda_graph_replay_rejects_changed_captured_scalar(self):
        for field, replay_value in (("max_seq_len", 2), ("max_prefix_length", 1)):
            with self.subTest(field=field):
                impl = self._make_paged_impl(self._make_rope_params())
                impl.rope_kvcache_impl.prepare.return_value = self._make_rope_params(
                    position_ids=impl.rope_params.position_ids,
                    cu_seqlens=impl.rope_params.cu_seqlens,
                    input_lengths=impl.rope_params.input_lengths,
                    prefix_lengths=impl.rope_params.prefix_lengths,
                    **{field: replay_value},
                )

                with self.assertRaisesRegex(RuntimeError, rf"exceeds captured {field}"):
                    impl.prepare_cuda_graph(mock.sentinel.attn_inputs)

    def test_cuda_graph_replay_accepts_smaller_captured_scalars(self):
        impl = self._make_paged_impl(
            self._make_rope_params(max_seq_len=4, max_prefix_length=2)
        )
        impl.rope_kvcache_impl.prepare.return_value = self._make_rope_params(
            position_ids=impl.rope_params.position_ids,
            cu_seqlens=impl.rope_params.cu_seqlens,
            input_lengths=impl.rope_params.input_lengths,
            prefix_lengths=impl.rope_params.prefix_lengths,
            max_seq_len=2,
            max_prefix_length=1,
        )

        impl.prepare_cuda_graph(mock.sentinel.attn_inputs)


if __name__ == "__main__":
    unittest.main()
