"""Tests for the GptModelBase input_embeddings guard.

Verifies the __init_subclass__ wrapper that:
  1. Rejects input_embeddings on subclasses which haven't opted in via
     supports_input_embeddings=True (pre-check before forward runs).
  2. Asserts opted-in subclasses actually call apply_input_embeddings()
     during forward, so a multimodal subclass that declares support but
     forgets the call fails loudly instead of dropping the data
     (post-check after forward returns).
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


class _AwareAndConsumes(GptModelBase):
    """Subclass that opts in AND consumes the embeddings."""

    supports_input_embeddings = True

    def __init__(self):
        nn.Module.__init__(self)
        self.forward_called = False
        self.micro_called = False

    def forward(self, inputs: Any) -> str:
        self.forward_called = True
        # Stand in for the real apply_input_embeddings(): set the marker that
        # the entry wrapper's post-check looks for.
        self._input_embeddings_consumed = True
        return "forward-ran"

    def forward_micro_batch(self, inputs: List[Any]) -> str:
        self.micro_called = True
        self._input_embeddings_consumed = True
        return "micro-ran"


class _AwareButForgets(GptModelBase):
    """Subclass that opts in but FORGETS to consume — must trigger post-check."""

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

    # ---- aware + consumes: wrapper must NOT reject, even with embeddings ----

    def test_aware_consumes_forward_with_embeddings(self):
        model = _AwareAndConsumes()
        inputs = _fake_inputs(input_embeddings=["fake-emb"])
        self.assertEqual(model.forward(inputs), "forward-ran")
        self.assertTrue(model.forward_called)

    def test_aware_consumes_forward_micro_batch_with_embeddings(self):
        model = _AwareAndConsumes()
        batch = [_fake_inputs(input_embeddings=["fake-emb"])]
        self.assertEqual(model.forward_micro_batch(batch), "micro-ran")
        self.assertTrue(model.micro_called)

    def test_aware_consumes_forward_runs_when_no_embeddings(self):
        # No embeddings → post-check is skipped; opt-in subclasses run normally.
        model = _AwareAndConsumes()
        self.assertEqual(model.forward(_fake_inputs(None)), "forward-ran")
        self.assertTrue(model.forward_called)

    # ---- aware but forgets to consume: post-check must reject ----

    def test_aware_but_forgets_raises_on_forward(self):
        model = _AwareButForgets()
        inputs = _fake_inputs(input_embeddings=["fake-emb"])
        with self.assertRaises(RuntimeError) as ctx:
            model.forward(inputs)
        msg = str(ctx.exception)
        self.assertIn("_AwareButForgets.forward", msg)
        self.assertIn("did not call", msg)
        self.assertIn("apply_input_embeddings", msg)
        # forward did execute — the post-check ran after it returned.
        self.assertTrue(model.forward_called)

    def test_aware_but_forgets_raises_on_forward_micro_batch(self):
        model = _AwareButForgets()
        batch = [_fake_inputs(input_embeddings=["fake-emb"])]
        with self.assertRaises(RuntimeError) as ctx:
            model.forward_micro_batch(batch)
        self.assertIn("_AwareButForgets.forward_micro_batch", str(ctx.exception))
        self.assertTrue(model.micro_called)

    def test_aware_but_forgets_passes_when_no_embeddings(self):
        # Post-check only fires when inputs actually carry embeddings.
        model = _AwareButForgets()
        self.assertEqual(model.forward(_fake_inputs(None)), "forward-ran")
        self.assertTrue(model.forward_called)

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
