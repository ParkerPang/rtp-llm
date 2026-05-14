from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops.compute_ops import PyModelInputs


class ApplyInputEmbeddingsTest(TestCase):
    HIDDEN_DIM = 64

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")
        torch.manual_seed(42)

    def _make_inputs(self, embeddings, locs):
        """Helper to build a PyModelInputs with input_embeddings set."""
        inputs = PyModelInputs()
        inputs.input_ids = torch.zeros(1, dtype=torch.int32)
        if embeddings is not None:
            inputs.input_embeddings = embeddings
            inputs.input_embeddings_locs = torch.tensor(locs, dtype=torch.int32)
        return inputs

    def test_single_embedding(self):
        D = self.HIDDEN_DIM
        inputs_embeds = torch.randn(5, D, dtype=torch.bfloat16)
        original = inputs_embeds.clone()

        emb = torch.randn(1, D, dtype=torch.bfloat16)
        inputs = self._make_inputs([emb], [2])

        result = GptModelBase.apply_input_embeddings(inputs_embeds, inputs)

        self.assertTrue(torch.equal(result[2], emb[0]))
        self.assertTrue(torch.equal(result[0], original[0]))
        self.assertTrue(torch.equal(result[1], original[1]))
        self.assertTrue(torch.equal(result[3], original[3]))
        self.assertTrue(torch.equal(result[4], original[4]))

    def test_multiple_embeddings(self):
        D = self.HIDDEN_DIM
        inputs_embeds = torch.randn(6, D, dtype=torch.bfloat16)
        original = inputs_embeds.clone()

        emb1 = torch.randn(1, D, dtype=torch.bfloat16)  # loc=0, length=1
        emb2 = torch.randn(2, D, dtype=torch.bfloat16)  # loc=3, length=2
        inputs = self._make_inputs([emb1, emb2], [0, 3])

        result = GptModelBase.apply_input_embeddings(inputs_embeds, inputs)

        self.assertTrue(torch.equal(result[0], emb1[0]))
        self.assertTrue(torch.equal(result[3:5], emb2))
        self.assertTrue(torch.equal(result[1], original[1]))
        self.assertTrue(torch.equal(result[2], original[2]))
        self.assertTrue(torch.equal(result[5], original[5]))

    def test_none_embeddings(self):
        D = self.HIDDEN_DIM
        inputs_embeds = torch.randn(5, D, dtype=torch.bfloat16)
        original = inputs_embeds.clone()

        inputs = self._make_inputs(None, None)

        result = GptModelBase.apply_input_embeddings(inputs_embeds, inputs)

        self.assertTrue(torch.equal(result, original))

    def test_dtype_conversion(self):
        D = self.HIDDEN_DIM
        inputs_embeds = torch.randn(5, D, dtype=torch.bfloat16)

        emb = torch.randn(1, D, dtype=torch.float32)  # fp32, should be cast to bf16
        inputs = self._make_inputs([emb], [1])

        result = GptModelBase.apply_input_embeddings(inputs_embeds, inputs)

        self.assertEqual(result.dtype, torch.bfloat16)
        expected = emb[0].to(torch.bfloat16)
        self.assertTrue(torch.equal(result[1], expected))

    def test_cpu_embedding_to_cuda(self):
        """CPU embedding should be moved to CUDA device automatically."""
        D = self.HIDDEN_DIM
        inputs_embeds = torch.randn(5, D, dtype=torch.bfloat16, device="cuda")

        # Explicitly create embedding on CPU with different dtype
        emb = torch.randn(1, D, dtype=torch.float32, device="cpu")
        inputs = self._make_inputs([emb], [2])

        result = GptModelBase.apply_input_embeddings(inputs_embeds, inputs)

        self.assertEqual(result.device.type, "cuda")
        self.assertEqual(result.dtype, torch.bfloat16)
        expected = emb[0].to(device="cuda", dtype=torch.bfloat16)
        self.assertTrue(torch.equal(result[2], expected))


if __name__ == "__main__":
    main()
