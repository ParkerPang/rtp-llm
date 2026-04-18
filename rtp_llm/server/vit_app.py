import asyncio
import base64
import logging
import socket
import threading
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import Body, FastAPI, HTTPException
from fastapi import Request as RawRequest
from fastapi import status
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from typing_extensions import override
from uvicorn import Config, Server
from uvicorn.loops.auto import auto_loop_setup

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.uvicorn_config import get_uvicorn_logging_config
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.distribute.worker_info import WorkerInfo
from rtp_llm.metrics import kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.ops import RoleType
from rtp_llm.server.vit_rpc_server import (
    MultimodalRpcServer,
    create_rpc_server,
    trans_output,
)
from rtp_llm.utils.base_model_datatypes import (
    MMPreprocessConfig,
    MMUrlType,
    MultimodalInput,
)


class GracefulShutdownServer(Server):
    def set_server(self, vit_endpoint_server):
        self.vit_endpoint_server = vit_endpoint_server

    @override
    async def shutdown(self, sockets: Optional[List[socket.socket]] = None) -> None:
        self.vit_endpoint_server.stop()
        await super().shutdown(sockets)


class VitEndpointApp:
    def __init__(
        self,
        py_env_configs: PyEnvConfigs,
        vit_process_engine: Optional[MMProcessEngine],
    ):
        self.py_env_configs = py_env_configs
        self.vit_endpoint_server = VitEndpointServer(
            self.py_env_configs, vit_process_engine
        )

    def start(
        self,
        grpc_port: int,
        http_port: Optional[int] = None,
    ):
        """
        启动 VIT 端点应用

        Args:
            worker_info: Worker 信息
            grpc_port: gRPC 端口号（从外部传入）
            http_port: HTTP 端口号（从外部传入，如果为 None 且最终计算后仍为 None，表示工作进程模式（不启动 HTTP 服务器）
        """
        # 启动 gRPC 服务器
        self._start_grpc(grpc_port)

        # 如果 http_port 为 None，表示工作进程模式（不启动 HTTP 服务器，只由主进程提供 HTTP）
        if http_port is None:
            logging.info(
                f"Vit Worker App: skipping HTTP server (server_id={self.py_env_configs.server_config.vit_server_id})"
            )
            # 只启动 gRPC 服务器，不启动 HTTP
            self.vit_endpoint_server.wait_for_termination()
            return

        # 启动 HTTP 服务器
        self._start_http(http_port)

    def _start_grpc(self, grpc_port: int):
        self.vit_endpoint_server.start(grpc_port)

    def _start_http(self, http_port: int):
        logging.info(f"Vit App start in http port {http_port}")

        # 设置事件循环
        loop = self._setup_event_loop()

        # 创建 FastAPI 应用
        app = self.create_app()

        # 创建并配置 socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind(("0.0.0.0", http_port))
        sock.listen()
        fd = sock.fileno()

        # 获取配置
        timeout_keep_alive = self.py_env_configs.server_config.timeout_keep_alive

        # 创建 uvicorn 配置
        config = Config(
            app,
            fd=fd,
            loop=loop,
            log_config=get_uvicorn_logging_config(),
            timeout_keep_alive=timeout_keep_alive,
            h11_max_incomplete_event_size=1024 * 1024,
        )

        # 启动 HTTP 服务器
        try:
            server = GracefulShutdownServer(config)
            server.set_server(self.vit_endpoint_server)
            server.run()
        except BaseException as e:
            self.vit_endpoint_server.stop()
            raise e

    def _setup_event_loop(self) -> str:
        """
        设置事件循环

        Returns:
            事件循环类型字符串 ("auto" 或 "none")
        """
        loop = "auto"
        if threading.current_thread() != threading.main_thread():
            # NOTE: asyncio
            loop = "none"
            auto_loop_setup()
            asyncio.set_event_loop(asyncio.new_event_loop())
        return loop

    def create_app(self):
        middleware = [
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        ]
        app = FastAPI(middleware=middleware)

        @app.get("/health")
        @app.post("/health")
        @app.post("/health_check")
        async def health():
            return "ok"

        @app.get("/worker_status")
        @app.post("/worker_status")
        async def worker_status():
            return self.vit_endpoint_server.worker_status()

        @app.post("/v1/multimodal/embedding")
        async def multimodal_embedding(request: Dict[str, Any]):
            """
            HTTP 推理接口：接收图片并返回 multimodal embedding。

            请求格式:
            {
                "images": [
                    {"url": "data:image/jpeg;base64,..."},
                    {"url": "https://example.com/image.jpg"},
                    {"base64": "/9j/4AAQ..."}
                ],
                "config": {  // 可选
                    "min_pixels": 200704,
                    "max_pixels": 1003520
                }
            }
            """
            import traceback

            try:
                result = self.vit_endpoint_server.multimodal_embedding_http(request)
                return result
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                error_detail = traceback.format_exc()
                logging.error(f"Multimodal embedding error: {e}\n{error_detail}")
                raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

        return app


class VitEndpointServer:
    def __init__(
        self,
        py_env_configs: PyEnvConfigs,
        vit_process_engine: Optional[MMProcessEngine],
    ):
        self.rpc_server = None
        self.mm_rpc_server = None
        self.py_env_configs = py_env_configs
        self.mm_process_engine = vit_process_engine

        if self.mm_process_engine is None:
            return

        self.mm_rpc_server = MultimodalRpcServer(self.mm_process_engine)
        self.rpc_server = create_rpc_server()
        add_MultimodalRpcServiceServicer_to_server(self.mm_rpc_server, self.rpc_server)
        kmonitor.init()

    def wait_for_termination(self):
        """等待 gRPC 服务器终止"""
        if self.rpc_server:
            self.rpc_server.wait_for_termination()

    def start(self, grpc_port: int):
        if self.mm_process_engine is None:
            return

        self.rpc_server.add_insecure_port(f"0.0.0.0:{grpc_port}")
        self.rpc_server.start()
        logging.info(f"Vit Server started on grpc port {grpc_port} (bind=0.0.0.0)")

    def stop(self):
        if self.rpc_server is not None:
            self.rpc_server.stop(grace=None)
        if self.mm_rpc_server is not None:
            self.mm_rpc_server.stop()

    def worker_status(self):
        return {}

    def multimodal_embedding_http(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理 HTTP 推理请求，返回 multimodal embedding 结果。

        请求格式:
        {
            "images": [
                {"url": "data:image/jpeg;base64,..."},
                {"url": "https://example.com/image.jpg"},
                {"base64": "/9j/4AAQ..."}
            ],
            "config": {  // 可选
                "min_pixels": 200704,
                "max_pixels": 1003520
            }
        }
        """
        if self.mm_process_engine is None:
            raise ValueError("Multimodal process engine is not initialized")

        images = request.get("images", [])
        if not images:
            raise ValueError("No images provided in request")

        config_dict = request.get("config", {})
        preprocess_config = MMPreprocessConfig(
            width=config_dict.get("width", 0),
            height=config_dict.get("height", 0),
            min_pixels=config_dict.get("min_pixels", 256 * 28 * 28),
            max_pixels=config_dict.get("max_pixels", 1280 * 28 * 28),
            fps=config_dict.get("fps", 2),
            min_frames=config_dict.get("min_frames", 4),
            max_frames=config_dict.get("max_frames", 768),
        )

        mm_inputs = []
        for image_item in images:
            if isinstance(image_item, str):
                if image_item.startswith(("http", "data:")):
                    url = image_item
                elif image_item.startswith("base64:"):
                    # "base64:image/jpeg;base64,..." → "data:image/jpeg;base64,..."
                    url = "data:" + image_item[len("base64:") :]
                else:
                    url = f"data:image/jpeg;base64,{image_item}"
            elif isinstance(image_item, dict):
                if "url" in image_item:
                    url = image_item["url"]
                elif "base64" in image_item:
                    url = f"data:image/jpeg;base64,{image_item['base64']}"
                else:
                    raise ValueError(
                        f"Image item must contain 'url' or 'base64' key, got: {list(image_item.keys())}"
                    )
            else:
                raise ValueError(f"Unsupported image item type: {type(image_item)}")

            mm_inputs.append(
                MultimodalInput(
                    url=url,
                    mm_type=MMUrlType.IMAGE,
                    tensor=torch.empty(0),
                    config=preprocess_config,
                )
            )

        res = self.mm_process_engine.mm_embedding_impl(mm_inputs)

        result = {
            "split_size": [e.shape[0] for e in res.embeddings],
        }
        if res.embeddings:
            embedding_tensor = torch.concat(res.embeddings)
            result["embedding_shape"] = list(embedding_tensor.shape)
            result["embedding_dtype"] = str(embedding_tensor.dtype)
            result["embedding_base64"] = base64.b64encode(
                embedding_tensor.cpu().to(torch.float16).numpy().tobytes()
            ).decode("utf-8")
        if res.position_ids and len(res.position_ids) > 0:
            pos_tensor = torch.concat(res.position_ids)
            result["position_ids_shape"] = list(pos_tensor.shape)

        return result
