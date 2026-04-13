# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time
from typing import Any, Optional
from uuid import uuid4

import requests

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RerankerTool(BaseTool):
    """A demo tool for calculating the reward of gsm8k.

    - `get_openai_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "reranker",
                "description": "A tool for searching and reranking documents for a query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search for",
                        },
                    },
                    "required": ["query"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.reranker_server_url = config.get("reranker_server_url", "http://localhost:8002/rerank")
        self.reranker_topk = config.get("reranker_topk", 3)
        self.return_full_document = config.get("return_full_document", False)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, dummy: Optional[float] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "query": "",
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query = parameters.get("query", "")
        self._instance_dict[instance_id]["query"] = query

        data = {"query": query}

        rerank_text = await self.rerank_request(data)
        return ToolResponse(text=rerank_text), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

    async def rerank_request(self, data: dict) -> str:
        retries = 3
        for i in range(retries):
            try:
                response = requests.post(self.reranker_server_url, json=data)
                response.raise_for_status()
                data = response.json()  # Expects a list of results

                rerank_text = ""
                # rerank_text += "<information>\n"
                for idx, (document, reasoning) in enumerate(zip(data["documents"][:self.reranker_topk], data["reasonings"][:self.reranker_topk], strict=False)):
                    content = document["text"]
                    if self.return_full_document:
                        title = content.split("\n")[0]
                        text = "\n".join(content.split("\n")[1:])
                        rerank_text += (
                            f"Doc {idx + 1}(Title: {title}) {text}\n"
                            f"Reasoning about Doc {idx + 1}'s relevance to the query: {reasoning}\n"
                        )
                    else:
                        rerank_text += f"Reasoning about Doc {idx + 1}'s relevance to the query: {reasoning}\n"
                # rerank_text += "</information>\n"
                return rerank_text
            except requests.exceptions.RequestException as e:
                print(f"Retriever request failed: {e}. Retrying ({i + 1}/{retries})...")
                if i < retries - 1:
                    time.sleep(0.5)
                else:
                    raise RuntimeError("API call to reranker server failed") from e
