load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")

def sm120_suites():
    native.test_suite(
        name = "smoke_sm120_basic",
        tests = [
            smoke_test(
                name = "softmax_probs_sm120",
                task_info = "data/model/qwen25/q_r_softmax_probs_sm120.json",
                smoke_args = "--act_type FP16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "fp16_sm120",
                task_info = "data/model/qwen25/q_r_s_fp16_sm120.json",
                smoke_args = "--act_type FP16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "bf16_sm120",
                task_info = "data/model/qwen25/q_r_s_bf16_sm120.json",
                smoke_args = "--act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "random_seed_sm120",
                task_info = "data/model/qwen25/test_random_seed_sm120.json",
                smoke_args = "--act_type FP16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "logits_index_sm120",
                task_info = "data/model/qwen25/logits_index_q_r_sm120.json",
                smoke_args = "--act_type FP16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
        ],
    )

    native.test_suite(
        name = "smoke_sm120_dense",
        tests = [
            smoke_test(
                name = "dense_fp8pb_dynamic_sm120",
                task_info = "data/model/qwen3/q_r_fp8pb_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "dense_fp8pt_dynamic_sm120",
                task_info = "data/model/qwen3/q_r_h20_per_tensor_w13.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--quantization FP8_DYNAMIC_PER_TENSOR --act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "dense_fp8kv_cudagraph_sm120",
                task_info = "data/model/qwen25/q_r_fp8_kv_cache_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--warm_up 0 --seq_size_per_block 64 --act_type BF16 --test_block_num 1000 --fp8_kv_cache 1 --enable_cuda_graph 1",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "qwen2_0_5b_fp8pb_sm120",
                task_info = "data/model/qwen2/q_r_fp8pb_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "qwen2_0_5b_fp8pt_sm120",
                task_info = "data/model/qwen2/q_r_fp8pt_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--quantization FP8_DYNAMIC_PER_TENSOR --act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "qwen2_5_0_5b_fp8pb_sm120",
                task_info = "data/model/qwen25/q_r_fp8pb_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "qwen2_5_0_5b_fp8pt_sm120",
                task_info = "data/model/qwen25/q_r_fp8pt_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--quantization FP8_DYNAMIC_PER_TENSOR --act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "qwen3_8b_fp8pb_sm120",
                task_info = "data/model/qwen3/q_r_8b_fp8pb_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "qwen3_8b_fp8pt_sm120",
                task_info = "data/model/qwen3/q_r_8b_fp8pt_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--quantization FP8_DYNAMIC_PER_TENSOR --act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "qwen3_1_7b_prequant_fp8pb_sm120",
                task_info = "data/model/qwen3/q_r_1_7b_prequant_fp8_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "dense_fp8_random_seed_sm120",
                task_info = "data/model/qwen3/test_random_seed_fp8_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "dense_fp8_logits_index_sm120",
                task_info = "data/model/qwen3/logits_index_fp8_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            # ========== PR-5 扩展 (Tier 1: BF16 + dynamic FP8 across Qwen2-0.5B / Qwen2.5-0.5B / Qwen3-8B) ==========
            smoke_test(
                name="qwen2_0_5b_fp8pb_sm120",
                task_info="data/model/qwen2/q_r_fp8pb_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="qwen2_0_5b_fp8pt_sm120",
                task_info="data/model/qwen2/q_r_fp8pt_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_DYNAMIC_PER_TENSOR --act_type BF16",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="qwen2_5_0_5b_fp8pb_sm120",
                task_info="data/model/qwen25/q_r_fp8pb_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="qwen2_5_0_5b_fp8pt_sm120",
                task_info="data/model/qwen25/q_r_fp8pt_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_DYNAMIC_PER_TENSOR --act_type BF16",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="qwen3_8b_fp8pb_sm120",
                task_info="data/model/qwen3/q_r_8b_fp8pb_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="qwen3_8b_fp8pt_sm120",
                task_info="data/model/qwen3/q_r_8b_fp8pt_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_DYNAMIC_PER_TENSOR --act_type BF16",
                gpu_type=["RTX_5000_PRO"],
            ),
            # ========== PR-5 扩展 (Tier 2: 预量化 FP8 model load 路径) ==========
            # Qwen3-1.7B-FP8: HF quant_method=fp8 + weight_block_size=[128,128] → CudaFp8VllmBlockwiseLinear (PB)
            smoke_test(
                name="qwen3_1_7b_prequant_fp8pb_sm120",
                task_info="data/model/qwen3/q_r_1_7b_prequant_fp8_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            # ========== PR-5 扩展 (Tier 3: FP8 路径上的边角 case，参考 PR-4 random_seed/logits_index 模板) ==========
            smoke_test(
                name="dense_fp8_random_seed_sm120",
                task_info="data/model/qwen3/test_random_seed_fp8_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="dense_fp8_logits_index_sm120",
                task_info="data/model/qwen3/logits_index_fp8_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
        ],
    )
