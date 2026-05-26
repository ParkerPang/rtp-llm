"""Tests for the GptModelBase input_embeddings guard.

Verifies the __init_subclass__ wrapper that rejects input_embeddings on
subclasses which haven't opted in via supports_input_embeddings=True.
"""

import types
import unittest
from typing import Any, List

from torch import nn

from rtp_llm.models_py.model_desc.module_base import GptModelBase


def _fake_inputs(input_embeddings=None):
    """Build a minimal stand-in for PyModelInputs that the wrapper inspects."""
    return types.SimpleNamespace(input_embeddings=input_embeddings)


class _Unaware(GptModelBase):
    """Subclass that does NOT opt in. The wrapper must reject embeddings."""

    def __init__(self):
        nn.Module.__init__(self)
        self.forward_called = False
        self.micro_called = False

    def forward(self, inputs: Any) -> str:
        self.forward_called = True
        return "forward-ran"

    def forward_micro_batch(self, inputs: List[Any]) -> str:
        self.micro_called = True
        return "micro-ran"


class _Aware(GptModelBase):
    """Subclass that opts in. The wrapper must pass embeddings through."""

    supports_input_embeddings = True

    def __init__(self):
        nn.Module.__init__(self)
        self.forward_called = False
        self.micro_called = False

    def forward(self, inputs: Any) -> str:
        self.forward_called = True
        return "forward-ran"

    def forward_micro_batch(self, inputs: List[Any]) -> str:
        self.micro_called = True
        return "micro-ran"


class SupportsInputEmbeddingsTest(unittest.TestCase):
    # ---- unaware subclass: wrapper must reject when embeddings present ----

    def test_unaware_rejects_forward_with_embeddings(self):
        model = _Unaware()
        inputs = _fake_inputs(input_embeddings=["fake-emb"])
        with self.assertRaises(NotImplementedError) as ctx:
            model.forward(inputs)
        self.assertIn("_Unaware.forward", str(ctx.exception))
        self.assertIn("supports_input_embeddings", str(ctx.exception))
        self.assertFalse(model.forward_called)

    def test_unaware_rejects_forward_micro_batch_with_embeddings(self):
        model = _Unaware()
        # forward_micro_batch receives a list; the guard must scan every item.
        batch = [_fake_inputs(), _fake_inputs(input_embeddings=["fake-emb"])]
        with self.assertRaises(NotImplementedError) as ctx:
            model.forward_micro_batch(batch)
        self.assertIn("_Unaware.forward_micro_batch", str(ctx.exception))
        self.assertFalse(model.micro_called)

    # ---- unaware subclass: no embeddings → forward runs normally ----

    def test_unaware_passes_when_input_embeddings_is_none(self):
        model = _Unaware()
        self.assertEqual(model.forward(_fake_inputs(None)), "forward-ran")
        self.assertTrue(model.forward_called)

    def test_unaware_passes_when_input_embeddings_is_empty_list(self):
        # Empty list must be treated the same as None (no-op, not a rejection).
        model = _Unaware()
        self.assertEqual(model.forward(_fake_inputs([])), "forward-ran")
        self.assertTrue(model.forward_called)

    def test_unaware_passes_when_micro_batch_has_no_embeddings(self):
        model = _Unaware()
        batch = [_fake_inputs(None), _fake_inputs([])]
        self.assertEqual(model.forward_micro_batch(batch), "micro-ran")
        self.assertTrue(model.micro_called)

    # ---- aware subclass: wrapper must NOT reject, even with embeddings ----

    def test_aware_passes_forward_with_embeddings(self):
        model = _Aware()
        inputs = _fake_inputs(input_embeddings=["fake-emb"])
        self.assertEqual(model.forward(inputs), "forward-ran")
        self.assertTrue(model.forward_called)

    def test_aware_passes_forward_micro_batch_with_embeddings(self):
        model = _Aware()
        batch = [_fake_inputs(input_embeddings=["fake-emb"])]
        self.assertEqual(model.forward_micro_batch(batch), "micro-ran")
        self.assertTrue(model.micro_called)

    # ---- wrapper idempotency: re-wrapping a subclass must not double-guard ----

    def test_wrapper_is_idempotent_across_class_definitions(self):
        # Defining a second subclass with the same forward must not stack
        # wrappers (would cause a chain of NotImplementedError messages).
        class _SecondUnaware(GptModelBase):
            def __init__(self):
                nn.Module.__init__(self)

            def forward(self, inputs: Any) -> str:
                return "ran"

        model = _SecondUnaware()
        # The wrapper exposes a marker attribute so we can assert single wrap.
        self.assertTrue(getattr(type(model).forward, "__embeddings_guarded__", False))
        # Calling without embeddings still works.
        self.assertEqual(model.forward(_fake_inputs(None)), "ran")

    # ---- default opt-in is False on the base class ----

    def test_default_supports_flag_is_false(self):
        self.assertFalse(GptModelBase.supports_input_embeddings)


if __name__ == "__main__":
    unittest.main()
