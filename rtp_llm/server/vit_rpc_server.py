import logging
from concurrent import futures

import grpc

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    CacheStatusPB,
    CacheVersionPB,
    MMPreprocessConfigPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
    MultimodalOutputsPB,
    StatusVersionPB,
    WorkerStatusPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.distribute.distributed_server import get_world_info
from rtp_llm.model_factory import ModelFactory
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes, MMProcessEngine
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.base_model_datatypes import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.grpc_util import trans_from_tensor, trans_tensor


def trans_output(res: MMEmbeddingRes):
    output_pb = MultimodalOutputsPB()
    contain_pos = (res.position_ids is not None) and (len(res.position_ids) > 0)
    contain_deepstack = (res.deepstack_embeds is not None) and (
        len(res.deepstack_embeds) > 0
    )
    for i in range(len(res.embeddings)):
        output = MultimodalOutputPB(
            multimodal_embedding=trans_from_tensor(res.embeddings[i]),
            multimodal_pos_id=(
                trans_from_tensor(res.position_ids[i]) if contain_pos else None
            ),
            multimodal_deepstack_embedding=(
                trans_from_tensor(res.deepstack_embeds[i])
                if contain_deepstack
                else None
            ),
        )
        output_pb.multimodal_outputs.append(output)
    return output_pb


class MultimodalRpcServer(MultimodalRpcServiceServicer):
    def __init__(self, mm_process_engine: MMProcessEngine):
        self.engine = mm_process_engine

    def RemoteMultimodalEmbedding(self, multimodal_inputs: MultimodalInputsPB, context):
        res: MMEmbeddingRes = self.engine.mm_embedding_rpc(multimodal_inputs)
        return trans_output(res)

    def GetWorkerStatus(self, request: StatusVersionPB, context):
        worker_status = WorkerStatusPB()
        worker_status.role = "VIT"
        worker_status.status_version = 1
        worker_status.alive = True
        return worker_status

def vit_start_server():
    py_env_configs = setup_args()
    setup_and_configure_server(py_env_configs)
    url_data_cache_.resize_cache(py_env_configs.vit_config.url_cache_item_num)
    vit_emb_cache_.resize_cache(py_env_configs.vit_config.mm_cache_item_num)

    # Create and fully initialize engine config (global singleton, ports from config)
    engine_config = EngineConfig.create(py_env_configs, nccl_comm_config=None)

    # Create model configs (ModelConfig construction is handled in ModelFactory)
    # All model metadata (lora_infos, multi_task_prompt, model_name, template_type, mm_model_config)
    # is set in model_config by create_model_config()
    model_config = ModelFactory.create_model_config(
        model_args=py_env_configs.model_args,
        lora_config=py_env_configs.lora_config,
        kv_cache_config=engine_config.kv_cache_config,
        profiling_debug_logging_config=engine_config.profiling_debug_logging_config,
        generate_env_config=py_env_configs.generate_env_config,
        embedding_config=py_env_configs.embedding_config,
        quantization_config=py_env_configs.quantization_config,
        render_config=py_env_configs.render_config,
    )

    # Update engine_config based on model_config
    ModelFactory.update_engine_config_from_model_config(
        engine_config=engine_config,
        model_config=model_config,
    )

    # Create model using new API
    # All metadata is already in model_config (including mm_model_config)
    # vit_config is needed for multimodal models
    model = ModelFactory.from_model_configs(
        model_config=model_config,
        engine_config=engine_config,
        world_info=get_world_info(
            py_env_configs.server_config,
            py_env_configs.distribute_config,
            py_env_configs.parallelism_config,
        ),
        vit_config=py_env_configs.vit_config,
    )

def create_rpc_server():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=200),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
            ("grpc.max_concurrent_streams", -1),
            ("grpc.http2.min_ping_interval_without_data_ms", 1000),
            ("grpc.http2.max_ping_strikes", 1000),
        ],
    add_MultimodalRpcServiceServicer_to_server(
        MultimodalRpcServer(MMProcessEngine(model, model.vit_config)), server
    )
    logging.info(f"rpc_server_port: {py_env_configs.server_config.rpc_server_port}")
    server.add_insecure_port(f"0.0.0.0:{py_env_configs.server_config.rpc_server_port}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    vit_start_server()
