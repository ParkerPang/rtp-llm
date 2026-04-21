import concurrent.futures
import gc
import logging
import multiprocessing.pool
import os
import queue
import signal
import threading
import time
from multiprocessing import Lock, shared_memory
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.autograd.profiler as tap

from rtp_llm.access_logger.access_logger import MMAccessLogger
from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import ProfilingDebugLoggingConfig, VitConfig
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import MultimodalInputsPB
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_util import (
    trans_mm_input,
    url_data_cache_,
    vit_emb_cache_,
)
from rtp_llm.utils.base_model_datatypes import (
    MMPreprocessConfig,
    MMUrlType,
    MultimodalInput,
)
from rtp_llm.utils.time_util import Timer, timer_wrapper

mm_embedding_lock = Lock()
pool_lock = Lock()
_worker_vit_config: Optional[VitConfig] = None
_worker_preprocess_params: Optional[dict] = None
_worker_preprocess_func: Optional[Callable] = None


def _tensor_to_shm(tensor: torch.Tensor) -> Tuple[str, tuple, str]:
    """将 tensor 写入共享内存，返回 (shm_name, shape, dtype_str)"""
    arr = tensor.numpy()
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr[:]
    result = (shm.name, arr.shape, str(arr.dtype))
    shm.close()
    return result


def _shm_to_tensor(shm_name: str, shape: tuple, dtype_str: str) -> torch.Tensor:
    """从共享内存重建 tensor，零拷贝直接引用 shm buffer。
    调用方需要在 tensor 使用完毕后调用 tensor._shm_handle.close() 和 .unlink() 释放。
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    tensor = torch.from_numpy(arr)  # 零拷贝，直接引用 shm buffer
    tensor._shm_handle = shm  # 防止 GC 回收 shm，保持 buffer 有效
    return tensor


def _result_to_shm(result: Any) -> Any:
    """将预处理结果中的 tensor 转为共享内存引用"""
    if isinstance(result, tuple):
        return tuple(
            _tensor_to_shm(t) if isinstance(t, torch.Tensor) else t for t in result
        )
    if isinstance(result, torch.Tensor):
        return _tensor_to_shm(result)
    return result


def _result_from_shm(shm_result: Any) -> Any:
    """从共享内存引用重建预处理结果中的 tensor"""
    if isinstance(shm_result, tuple) and len(shm_result) > 0:
        # 判断是否是 shm 元数据 (name, shape, dtype_str)
        if (
            len(shm_result) == 3
            and isinstance(shm_result[0], str)
            and isinstance(shm_result[1], tuple)
        ):
            return _shm_to_tensor(*shm_result)
        # 否则是 tuple of (可能是 shm 元数据)
        return tuple(_result_from_shm(item) for item in shm_result)
    return shm_result


def _release_shm_tensor(tensor: Any) -> None:
    """释放 tensor 引用的共享内存"""
    if isinstance(tensor, torch.Tensor) and hasattr(tensor, "_shm_handle"):
        try:
            tensor._shm_handle.close()
            tensor._shm_handle.unlink()
        except Exception:
            pass


def _release_preprocess_result(result: Any) -> None:
    """释放预处理结果中所有 tensor 的共享内存"""
    if isinstance(result, tuple):
        for item in result:
            _release_shm_tensor(item)
    elif isinstance(result, torch.Tensor):
        _release_shm_tensor(result)


def _worker_initializer(
    vit_config: VitConfig,
    preprocess_params: dict,
    preprocess_func: Callable,
) -> None:
    """
    每个工作进程启动时调用的初始化函数。
    接收一次不变的参数，并将其存储在进程的全局变量中。
    """
    global _worker_vit_config, _worker_preprocess_params, _worker_preprocess_func
    # 让工作进程忽略 SIGINT 信号，这样主进程的 Ctrl+C 不会杀死它们
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _worker_vit_config = vit_config
    _worker_preprocess_params = preprocess_params
    _worker_preprocess_func = preprocess_func
    logging.info(f"Worker process {os.getpid()} initialized.")


def _worker_process_task(
    mm_inputs: List[MultimodalInput],
) -> Tuple[Any, float]:
    """
    只接收变化的 `mm_inputs` 参数。
    结果中的 tensor 通过共享内存传回，pipe 只传元数据。
    """
    if _worker_preprocess_func is None:
        raise RuntimeError("Worker process has not been initialized correctly.")

    with Timer() as route_timer:
        result = _worker_preprocess_func(
            mm_inputs, _worker_vit_config, **_worker_preprocess_params
        )
    shm_result = _result_to_shm(result)
    return shm_result, route_timer.cost_ms()


class PreprocessExecutor:
    """预处理执行器抽象基类，封装预处理逻辑"""

    def submit(self, work_item: "MMWorkItem") -> None:
        raise NotImplementedError

    def get_result(self, work_item: "MMWorkItem") -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass


class LocalPreprocessExecutor(PreprocessExecutor):
    """本地预处理执行器（同步执行）"""

    def __init__(
        self,
        preprocess_func: Callable,
        vit_config: VitConfig,
        preprocess_params: dict,
    ):
        self.preprocess_func = preprocess_func
        self.vit_config = vit_config
        self.preprocess_params = preprocess_params

    def submit(self, work_item: "MMWorkItem") -> None:
        if work_item.embedding_result is not None:
            return

        try:
            with Timer() as route_timer:
                result = self.preprocess_func(
                    work_item.mm_inputs, self.vit_config, **self.preprocess_params
                )
            preprocess_time = route_timer.cost_ms()
            work_item.preprocess_result = result
            # 使用简单的对象模拟 future 行为
            work_item.future = _LocalResult(result, preprocess_time)
        except Exception as e:
            logging.error(f"Error in local preprocessing: {e}", exc_info=True)
            raise

    def get_result(self, work_item: "MMWorkItem") -> None:
        if work_item.future is None:
            if work_item.embedding_result is None:
                raise ValueError("Embedding result and future cannot both be None")
            return

        try:
            _, preprocess_time = work_item.future.get()
            kmonitor.report(GaugeMetrics.VIT_PREPROCESS_RT_METRIC, preprocess_time)
        except Exception as e:
            logging.error(f"Error getting local preprocess result: {e}", exc_info=True)
            raise


class MultiprocessPreprocessExecutor(PreprocessExecutor):
    """多进程预处理执行器"""

    def __init__(
        self,
        mp_context: multiprocessing.context.BaseContext,
        vit_config: VitConfig,
        preprocess_params: dict,
        preprocess_func: Callable,
    ):
        self.mp_context = mp_context
        self.vit_config = vit_config
        self.preprocess_params = preprocess_params
        self.preprocess_func = preprocess_func
        self.pool: Optional[multiprocessing.pool.Pool] = None
        self._create_pool()

    def _create_pool(self) -> None:
        """创建进程池"""
        logging.info(
            f"Creating multiprocessing pool for preprocessing with {self.vit_config.mm_preprocess_max_workers} workers"
        )
        self.pool = self.mp_context.Pool(
            processes=self.vit_config.mm_preprocess_max_workers,
            initializer=_worker_initializer,
            initargs=(
                self.vit_config,
                self.preprocess_params,
                self.preprocess_func,
            ),
        )

    def submit(self, work_item: "MMWorkItem") -> None:
        if work_item.embedding_result is not None:
            return

        max_retries = 2
        for attempt in range(max_retries):
            try:
                work_item.future = self.pool.apply_async(
                    _worker_process_task, args=(work_item.mm_inputs,)
                )
                return
            except (BrokenPipeError, EOFError, OSError) as e:
                logging.warning(
                    f"Broken pool detected on submit (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    self._recover_pool()
                else:
                    logging.error(
                        f"Failed to recover from broken pool after {max_retries} attempts"
                    )
                    raise RuntimeError(
                        "Preprocessing pool is permanently broken."
                    ) from e
            except Exception as e:
                logging.error(f"Unexpected error during submission: {e}", exc_info=True)
                raise

    def get_result(self, work_item: "MMWorkItem") -> None:
        if work_item.future is None:
            if work_item.embedding_result is None:
                raise ValueError("Embedding result and future cannot both be None")
            return

        try:
            shm_result, preprocess_time = work_item.future.get(
                timeout=work_item.mm_timeout_ms / 1000.0
            )
            work_item.preprocess_result = _result_from_shm(shm_result)
            kmonitor.report(GaugeMetrics.VIT_PREPROCESS_RT_METRIC, preprocess_time)
        except multiprocessing.pool.TimeoutError:
            raise TimeoutError(
                f"Preprocessing timeout after {work_item.mm_timeout_ms}ms"
            )
        except (BrokenPipeError, EOFError, OSError) as e:
            logging.error(f"Broken pool detected while waiting for result: {e}")
            self._recover_pool()
            raise RuntimeError(
                "Preprocessing failed due to a broken worker process."
            ) from e
        except Exception as e:
            logging.error(f"Error getting preprocess result: {e}", exc_info=True)
            raise

    def _recover_pool(self) -> None:
        old_pool = self.pool
        if old_pool is None:
            return

        with pool_lock:
            if self.pool is not old_pool:
                logging.debug("Pool already recovered by another thread")
                return

            kmonitor.report(AccMetrics.VIT_PROCESS_POOL_RESTART_QPS_METRIC, 1)
            child_pids = self._get_child_pids_from_pool(old_pool)

            logging.warning(
                f"Broken process pool detected. Terminating pool with PIDs: {child_pids}"
            )

            try:
                old_pool.terminate()
                old_pool.join()
            except Exception as e:
                logging.warning(f"Error during pool termination: {e}", exc_info=True)

            try:
                self._create_pool()
                logging.info("Recreated ProcessPool after it was broken.")
            except Exception as e:
                logging.error(f"Failed to create new ProcessPool: {e}", exc_info=True)
                raise

    @staticmethod
    def _get_child_pids_from_pool(pool: multiprocessing.pool.Pool) -> List[int]:
        try:
            return [p.pid for p in pool._pool if p.is_alive()]
        except Exception:
            return []

    def shutdown(self) -> None:
        if self.pool is None:
            return
        logging.info("Shutting down the preprocessing pool...")
        self.pool.close()
        self.pool.join()
        logging.info("Preprocessing pool shut down.")


class _LocalResult:
    """本地预处理结果的简单包装类"""

    def __init__(self, result: Any, time: float):
        self.result = result
        self.time = time

    def get(self, timeout: Optional[float] = None) -> Tuple[Any, float]:
        return (self.result, self.time)


class _ProfilerSaveWorker:
    """后台线程异步保存 profiler trace，避免阻塞推理。
    借鉴 main 分支 ProfilerSaveWorker 设计。
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, result: Any, file_name: str) -> None:
        self._queue.put((result, file_name))

    def _run(self):
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                if self._stop:
                    return
                continue
            result, file_name = item
            try:
                logging.info(f"VitProfiler: saving trace to {file_name} (async)")
                result.save(file_name)
                logging.info(f"VitProfiler: trace saved: {file_name}")
            except Exception as e:
                logging.error(f"VitProfiler: failed to save trace {file_name}: {e}")

    def shutdown(self):
        self._stop = True
        self._thread.join(timeout=10.0)


class VitProfiler:
    """VIT Server 的 profiler，借鉴 main 分支 StepWindowProfiler 设计。

    使用 PyTorch Kineto profiler 采集 CPU + CUDA activity。
    全局配置开启（gen_vit_timeline_sync），跳过 warmup 后连续采集多次 forward，
    合并到一个 trace 文件中异步保存。

    使用方式：
        GEN_TIMELINE_SYNC=1 启动服务即可。
    """

    def __init__(
        self,
        enabled: bool = False,
        output_dir: str = ".",
        server_id: int = 0,
        warmup_steps: int = 2,
        num_steps: int = 5,
    ):
        self._enabled = enabled
        self._output_dir = output_dir
        self._server_id = server_id
        self._warmup_steps = warmup_steps
        self._num_steps = num_steps  # 一个 session 采集多少次 forward
        self._step_count = 0
        self._profiled_steps = 0
        self._profiling_active = False
        self._session_count = 0
        self._save_worker = _ProfilerSaveWorker() if enabled else None

        # Kineto profiler config (与 main 分支 TorchProfile 对齐)
        self._config = tap.ProfilerConfig(
            state=tap.ProfilerState.KINETO,
            report_input_shapes=True,
            profile_memory=False,
            with_stack=True,
            with_flops=False,
            with_modules=False,
            experimental_config=tap._ExperimentalConfig(),
        )
        self._activities = {tap.ProfilerActivity.CPU, tap.ProfilerActivity.CUDA}

        if enabled:
            logging.info(
                f"VitProfiler: enabled (output_dir={output_dir}, "
                f"server_id={server_id}, warmup_steps={warmup_steps}, "
                f"num_steps={num_steps})"
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def profile_embedding(self, fn: Callable, *args, **kwargs) -> Any:
        """包裹 embedding forward，按 session 采集 timeline。

        流程：
        1. 前 warmup_steps 次：直接执行，不采集
        2. 第 warmup_steps+1 次：启动 profiler
        3. 连续采集 num_steps 次 forward
        4. 第 warmup_steps+num_steps 次后：停止 profiler，异步保存一个文件
        5. 之后不再采集（单次 session）
        """
        if not self._enabled:
            return fn(*args, **kwargs)

        self._step_count += 1

        # Phase 1: warmup — 直接执行
        if self._step_count <= self._warmup_steps:
            logging.debug(
                f"VitProfiler: warmup step {self._step_count}/{self._warmup_steps}"
            )
            return fn(*args, **kwargs)

        # Phase 2: 启动 profiler（仅在 warmup 结束后的第一次）
        if not self._profiling_active:
            self._session_count += 1
            logging.info(
                f"VitProfiler: starting profiling session {self._session_count} "
                f"(will collect {self._num_steps} steps)"
            )
            tap._prepare_profiler(self._config, self._activities)
            tap._enable_profiler(self._config, self._activities)
            self._profiling_active = True
            self._profiled_steps = 0

        # Phase 3: 执行 forward（profiler 正在采集）
        result = fn(*args, **kwargs)
        self._profiled_steps += 1

        # Phase 4: 达到 num_steps，停止 profiler 并保存
        if self._profiled_steps >= self._num_steps:
            prof_result = tap._disable_profiler()
            self._profiling_active = False

            file_name = (
                f"{self._output_dir}/vit_profiler_s{self._server_id}"
                f"_n{self._num_steps}"
                f"_{self._session_count}.json"
            )
            self._save_worker.enqueue(prof_result, file_name)
            logging.info(
                f"VitProfiler: session {self._session_count} done, "
                f"collected {self._profiled_steps} steps, saving to {file_name}"
            )

            # 单次 session 后关闭
            self._enabled = False

        return result

    def shutdown(self):
        # 如果 profiler 还在运行（未达到 num_steps 就 shutdown），也要保存
        if self._profiling_active:
            try:
                prof_result = tap._disable_profiler()
                self._profiling_active = False
                file_name = (
                    f"{self._output_dir}/vit_profiler_s{self._server_id}"
                    f"_n{self._profiled_steps}"
                    f"_{self._session_count}_partial.json"
                )
                if self._save_worker:
                    self._save_worker.enqueue(prof_result, file_name)
                    logging.info(
                        f"VitProfiler: partial session saved ({self._profiled_steps} steps)"
                    )
            except Exception as e:
                logging.error(f"VitProfiler: error during shutdown: {e}")
        if self._save_worker:
            self._save_worker.shutdown()


class MMEmbeddingRes:
    """Result container for multimodal embedding operations."""

    def __init__(
        self,
        embeddings: List[torch.Tensor],
        position_ids: Optional[List[torch.Tensor]] = None,
        deepstack_embeds: Optional[List[torch.Tensor]] = None,
    ):
        self.embeddings = embeddings
        self.position_ids = position_ids
        self.deepstack_embeds = deepstack_embeds

    def __str__(self) -> str:
        return f"MMEmbeddingRes(length={len(self.embeddings)})"


class MMWorkItem:
    """Represents a work item for processing multimodal inputs."""

    def __init__(
        self, mm_inputs: List[MultimodalInput], mm_timeout_ms: Optional[int] = 120000
    ):
        if not mm_inputs:
            raise ValueError("No mm_input for work item")

        self.mm_inputs = mm_inputs
        self.mm_timeout_ms = (
            self.mm_inputs[0].config.mm_timeout_ms
            if self.mm_inputs[0].config.mm_timeout_ms != -1
            else mm_timeout_ms
        )
        self.mm_type = self.mm_inputs[0].mm_type

        self.preprocess_result: Optional[Any] = None
        self.embedding_result: Optional[Any] = None

        self.need_check_cache = len(mm_inputs) == 1 and mm_inputs[0].url is not None
        self.cache_key = (
            self.mm_inputs[0].to_string() if self.need_check_cache else None
        )
        self.embedding_result = vit_emb_cache_.check_cache(self.cache_key)

        # future 可以是 ApplyResult (multiprocess) 或 _LocalResult (local)
        self.future: Optional[Any] = None


class MMProcessEngine:
    """Engine for processing multimodal inputs with preprocessing and embedding."""

    def __init__(
        self,
        mm_part: MultiModalEmbeddingInterface,
        model_config: ModelConfig,
        vit_config: VitConfig,
        profiling_debug_logging_config: ProfilingDebugLoggingConfig,
        server_id: int = 0,
        is_proxy_mode: bool = False,
    ):
        """
        Initialize the multimodal process engine.

        Args:
            model: 模型实例
            server_id: 服务器 ID
            vit_config: VIT 配置
            profiling_debug_logging_config: 性能调试日志配置
            is_proxy_mode: 是否在 proxy 模式下运行
                          - True: proxy 模式下的 worker 进程，QPS 由 proxy 层记录，此处不记录
                          - False: standalone 模式，需要在此处记录 QPS
        """
        self.server_id = server_id
        self.vit_config = vit_config
        self.is_proxy_mode = is_proxy_mode
        self.contains_pos: bool = (
            model_config.mm_model_config.mm_position_ids_style != 0
        )
        self.mm_preprocess_batch_size: int = (
            model_config.mm_related_params.preprocess_batch_size
        )

        self.mp_context = multiprocessing.get_context("spawn")

        self.mm_part = mm_part

        # 创建 VIT profiler
        self._vit_profiler = VitProfiler(
            enabled=vit_config.gen_vit_timeline_sync,
            output_dir=profiling_debug_logging_config.torch_cuda_profiler_dir or ".",
            server_id=server_id,
        )

        # 根据 vit_config 创建预处理执行器
        preprocess_params = self.mm_part.get_preprocess_params()
        preprocess_func = self.mm_part.preprocess_input

        if vit_config.use_local_preprocess:
            self.preprocess_executor: PreprocessExecutor = LocalPreprocessExecutor(
                preprocess_func, vit_config, preprocess_params
            )
            logging.info(
                f"MMProcessEngine: Using LOCAL preprocessing mode (no subprocess pool)"
            )
        else:
            mp_context = multiprocessing.get_context("spawn")
            self.preprocess_executor = MultiprocessPreprocessExecutor(
                mp_context, vit_config, preprocess_params, preprocess_func
            )
            logging.info(
                f"MMProcessEngine: Using MULTIPROCESS preprocessing mode with {vit_config.mm_preprocess_max_workers} workers"
            )

        self.query_num: int = 0
        self._access_logger = MMAccessLogger(
            get_log_path(),
            profiling_debug_logging_config.log_file_backup_count,
        )

        vit_emb_cache_.resize_cache(self.vit_config.mm_cache_item_num)
        url_data_cache_.resize_cache(self.vit_config.url_cache_item_num)

    def inc_query_num(self) -> None:
        """Increment the query counter."""
        self.query_num += 1

    def dec_query_num(self) -> None:
        """Decrement the query counter."""
        self.query_num -= 1

    def get_query_num(self) -> int:
        """Get the current number of active queries."""
        return self.query_num

    @staticmethod
    def _maybe_tensor_to_list(tensor: Any, dim: int = 2) -> List[Any]:
        """Convert tensor to list format if needed."""
        if tensor is None:
            return []
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if len(tensor.shape) > dim:
            return list(tensor)
        return [tensor]

    def mm_embedding_rpc(self, mm_inputs: MultimodalInputsPB) -> MMEmbeddingRes:
        """Process multimodal inputs from RPC protocol buffer."""
        converted_inputs = trans_mm_input(mm_inputs)
        return self.mm_embedding_impl(converted_inputs)

    def mm_embedding_cpp(
        self,
        urls: List[str],
        types: List[int],
        tensors: List[torch.Tensor],
        mm_preprocess_configs: List[Any],
    ) -> MMEmbeddingRes:
        """Process multimodal inputs from C++ interface."""
        mm_inputs = [
            MultimodalInput(
                url, MMUrlType(url_type), tensor, MMPreprocessConfig(*config)
            )
            for url, url_type, tensor, config in zip(
                urls, types, tensors, mm_preprocess_configs
            )
        ]
        res = self.mm_embedding_impl(mm_inputs)
        res.position_ids = [pos.cpu() for pos in res.position_ids]
        return res

    def mm_embedding_impl(self, mm_inputs: List[MultimodalInput]) -> MMEmbeddingRes:
        """Core implementation for multimodal embedding processing."""
        logging.debug(f"{self.server_id} request received")
        try:
            with Timer() as e2e_timer:
                # 如果不是 proxy 模式（即 standalone 模式），记录 QPS
                if not self.is_proxy_mode:
                    kmonitor.report(
                        AccMetrics.VIT_QPS_METRIC, 1, {"source": "mm_embedding"}
                    )

                self.inc_query_num()
                if not self.vit_config.disable_access_log:
                    self._access_logger.log_query_access(mm_inputs)

                # 上报图片数量
                kmonitor.report(GaugeMetrics.VIT_IMAGE_NUM_METRIC, len(mm_inputs))

                work_items = self._create_work_items(mm_inputs)
                with Timer() as preprocess_timer:
                    with torch.profiler.record_function("vit::preprocess_wait"):
                        self._wait_for_preprocessing(work_items)
                preprocess_wait_ms = preprocess_timer.cost_ms()
                logging.info(
                    f"mm_preprocess latency: {preprocess_wait_ms:.2f}ms, items: {len(work_items)}"
                )
                kmonitor.report(
                    GaugeMetrics.VIT_PREPROCESS_WAIT_RT_METRIC, preprocess_wait_ms
                )

                with Timer() as embedding_timer:
                    with torch.profiler.record_function("vit::compute_embeddings"):
                        emb_res, pos_res, deepstack_embeds_res = (
                            self._compute_embeddings(work_items)
                        )
                logging.info(f"mm_embedding latency: {embedding_timer.cost_ms():.2f}ms")

                work_items = self._create_work_items(mm_inputs)
                with torch.profiler.record_function("vit::preprocess_wait"):
                    self._wait_for_preprocessing(work_items)
                with torch.profiler.record_function("vit::compute_embeddings"):
                    emb_res, pos_res, deepstack_embeds_res = self._compute_embeddings(
                        work_items
                    )

                result = MMEmbeddingRes(emb_res, pos_res, deepstack_embeds_res)
                if not self.vit_config.disable_access_log:
                    self._access_logger.log_success_access(mm_inputs, str(result))

                # 如果不是 proxy 模式（即 standalone 模式），记录成功 QPS
                if not self.is_proxy_mode:
                    kmonitor.report(AccMetrics.VIT_SUCCESS_QPS_METRIC, 1)

            # 上报端到端耗时
            kmonitor.report(GaugeMetrics.VIT_E2E_RT_METRIC, e2e_timer.cost_ms())

            return result
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            # 如果不是 proxy 模式（即 standalone 模式），记录错误 QPS
            if not self.is_proxy_mode:
                kmonitor.report(AccMetrics.VIT_ERROR_QPS_METRIC, 1)
            self._access_logger.log_exception_access(mm_inputs, e)
            raise
        finally:
            self.dec_query_num()

    def _create_work_items(self, mm_inputs: List[MultimodalInput]) -> List[MMWorkItem]:
        """Create work items and submit preprocessing tasks."""
        batch_size = (
            self.mm_preprocess_batch_size
            if self.mm_preprocess_batch_size != -1
            else len(mm_inputs)
        )

        work_items = []
        for index in range(0, len(mm_inputs), batch_size):
            batch = mm_inputs[index : index + batch_size]
            work_item = MMWorkItem(batch, mm_timeout_ms=self.vit_config.mm_timeout_ms)
            self.preprocess_executor.submit(work_item)
            work_items.append(work_item)

        return work_items

    def _wait_for_preprocessing(
        self,
        work_items: List[MMWorkItem],
    ) -> None:
        """Wait for all preprocessing tasks to complete."""
        for work_item in work_items:
            self.preprocess_executor.get_result(work_item)

    def _compute_embeddings(
        self, work_items: List[MMWorkItem]
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        """Compute embeddings for all work items."""
        emb_res, pos_res, tensor_res = [], [], []

        ordered_emb: List[Optional[Any]] = [None] * len(work_items)
        ordered_pos: List[Optional[Any]] = [None] * len(work_items)
        ordered_tensor: List[Optional[Any]] = [None] * len(work_items)

        pending_items: List[Tuple[int, MMWorkItem]] = []
        for idx, work_item in enumerate(work_items):
            if work_item.embedding_result is not None:
                ordered_emb[idx] = work_item.embedding_result[0]
                ordered_pos[idx] = work_item.embedding_result[1]
                if len(work_item.embedding_result) > 2:
                    ordered_tensor[idx] = work_item.embedding_result[2]
            else:
                pending_items.append((idx, work_item))

        if pending_items:
            batch_outputs = None
            data_list = [wi.preprocess_result for _, wi in pending_items]
            type_list = [wi.mm_type for _, wi in pending_items]
            lock_wait_start = time.time()
            with Timer() as route_timer:
                with mm_embedding_lock:
                    kmonitor.report(
                        GaugeMetrics.VIT_LOCK_WAIT_RT_METRIC,
                        (time.time() - lock_wait_start) * 1000,
                    )
                    batch_outputs = self._vit_profiler.profile_embedding(
                        self.mm_part.batched_embedding,
                        data_list,
                        type_list,
                    )
            kmonitor.report(GaugeMetrics.VIT_EMBEDDING_RT_METRIC, route_timer.cost_ms())

            # Release shm after embedding is done — tensors are no longer needed
            for _, wi in pending_items:
                _release_preprocess_result(wi.preprocess_result)
                wi.preprocess_result = None

            if batch_outputs is not None:
                for (idx, work_item), result in zip(pending_items, batch_outputs):
                    work_item.embedding_result = result
                    if work_item.need_check_cache:
                        vit_emb_cache_.insert_cache(work_item.cache_key, result)
                    ordered_emb[idx] = result[0]
                    ordered_pos[idx] = result[1]
                    if len(result) > 2:
                        ordered_tensor[idx] = result[2]

        for emb, pos, tensor in zip(ordered_emb, ordered_pos, ordered_tensor):
            emb_res.extend(self._maybe_tensor_to_list(emb, dim=2))
            pos_res.extend(self._maybe_tensor_to_list(pos, dim=2))
            tensor_res.extend(self._maybe_tensor_to_list(tensor, dim=3))
        return emb_res, pos_res, tensor_res

    def stop(self) -> None:
        """Shutdown the preprocessing executor and profiler."""
        self._vit_profiler.shutdown()
        self.preprocess_executor.shutdown()
