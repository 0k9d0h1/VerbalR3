import uvicorn
import argparse
import os
import signal
import json
import asyncio
from tqdm.asyncio import tqdm_asyncio
import openai
import torch
import requests
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Optional, Dict
from transformers import AutoTokenizer

# --- Imports from previous skeleton ---
from dataclasses import dataclass
from abc import ABC, abstractmethod


# --- 2. Pydantic Models for API Contract ---
class DocumentIn(BaseModel):
    id: str
    text: str
    score: float


class RerankSingleRequest(BaseModel):
    query: str


class RerankBatchRequest(BaseModel):
    queries: List[str]
    retriever_initial_topk: int
    search_turns_stats: List[Any] = None
    search_original_batch_idx: List[int] = None


# --- 3. Internal Data Structures & Reranker Logic (same as before) ---
@dataclass
class Document:
    id: str
    text: str
    score: float


@dataclass
class RerankResult:
    documents: List[Document]
    prompts: List[str] = None
    reasoning: List[str] = None
    score: List[int] = None
    total_token_count: int = None
    responses: List[str] = None
    logit: List[float] = None


class BaseReranker(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def rerank(self, query: str, documents: List[Document]) -> RerankResult:
        pass

    @property
    def has_reasoning(self) -> bool:
        return False


class APIRetriever:
    """Handles communication with the remote retriever server."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        """Retrieves documents for a single query."""
        payload = {"queries": [query], "topk": top_k, "return_scores": True}
        while True:
            try:
                response = requests.post(self.config["url"], json=payload)
                response.raise_for_status()
                # Assuming the response format is like: {"result": [[{"document": {"contents": "...", "id": ...}, "score": ...}]]}
                results = response.json()["result"]
                documents = []
                for j, r in enumerate(results[0]):
                    documents.append(
                        Document(
                            id=r["document"].get("id", str(j)),
                            text=r["document"]["contents"],
                            score=r["score"],
                        )
                    )
                return documents
            except requests.exceptions.RequestException as e:
                print(f"Retriever request failed: {e}. Retrying...")
                time.sleep(0.5)

    def batch_retrieve(self, queries: List[str], top_k: int) -> List[List[Document]]:
        """Retrieves documents for a single query."""
        payload = {"queries": queries, "topk": top_k, "return_scores": True}
        while True:
            try:
                response = requests.post(self.config["url"], json=payload)
                response.raise_for_status()
                # Assuming the response format is like: {"result": [[{"document": {"contents": "...", "id": ...}, "score": ...}]]}
                results = response.json()["result"]
                batched_documents = []
                for res in results:
                    documents = []
                    for j, r in enumerate(res):
                        documents.append(
                            Document(
                                id=r["document"].get("id", str(j)),
                                text=r["document"]["contents"],
                                score=r["score"],
                            )
                        )
                    batched_documents.append(documents)
                return batched_documents
            except requests.exceptions.RequestException as e:
                print(f"Retriever request failed: {e}. Retrying...")
                time.sleep(0.5)


class SplitRAGReranker:
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        max_output_tokens: int = 8192,
        max_score: int = 5,
        concurrency: int = 32,
        retriever_url: str = "http://localhost:8888/retrieve",
        vllm_url: List[str] = [
            "http://localhost:8001/v1",
            "http://localhost:8002/v1",
            "http://localhost:8003/v1",
            "http://localhost:8004/v1",
        ],  # <-- This will now be a comma-separated string
        temperature: float = 0.6,
        top_p: float = 0.95,
        jsonl_file_path: str = None,
        **kwargs,
    ):
        self.max_output_tokens = max_output_tokens
        self.model_name_or_path = model_name_or_path
        self.max_score = max_score

        # Initialize tokenizer with max length of
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.score_token_list = [
            self.tokenizer.encode(str(i))[0] for i in range(1, self.max_score + 1)
        ]

        self.retriever_config = {
            "url": retriever_url,
        }
        self.retriever = APIRetriever(config=self.retriever_config)

        # --- MODIFICATION FOR REPLICATION ---
        # vllm_url is now a comma-separated string, e.g., "url1,url2,url3"
        vllm_urls = vllm_url.split(",")
        if not vllm_urls:
            raise ValueError("vLLM URL is not provided.")

        print(
            f"Initializing reranker with {len(vllm_urls)} vLLM instances (replication)."
        )

        # Create a list of clients for load balancing
        self.clients = [
            openai.AsyncOpenAI(
                api_key="vllm",
                base_url=url.strip(),
            )
            for url in vllm_urls
        ]
        self.client_counter = 0
        self.num_clients = len(self.clients)
        # --- END MODIFICATION ---

        self.semaphore = asyncio.Semaphore(concurrency)

        self.sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_output_tokens,
            "logprobs": 20,
        }
        self.jsonl_file_path = jsonl_file_path

    # --- NEW METHOD FOR ROUND-ROBIN ---
    def get_next_client(self):
        """Selects the next vLLM client in a round-robin fashion."""
        client = self.clients[self.client_counter]
        self.client_counter = (self.client_counter + 1) % self.num_clients
        return client

    # --- END NEW METHOD ---

    async def get_completion(self, model_name, prompt, sampling_params, semaphore):
        # --- MODIFICATION: Use round-robin client ---
        async with self.semaphore:
            # Attempt the request up to 'num_clients' times
            # This ensures we try every available server before giving up
            for attempt in range(self.num_clients):
                client = self.get_next_client()  # Get the next client in rotation

                try:
                    response = await client.completions.create(
                        model=model_name,
                        prompt=prompt,
                        **sampling_params,
                    )
                    return response.choices[0]  # Success! Return immediately

                except Exception as e:
                    # Log the failure, but DO NOT return None yet
                    print(f"[Warning] Request failed on {client.base_url}: {e}")
                    print("Retrying with the next client...")
                    # The loop continues to the next iteration (next client)

            # If we reach here, it means ALL clients failed
            print("[Error] All clients failed to process the request.")
            return None
        # --- END MODIFICATION ---

    async def _process_with_vllm(self, prompts):
        system_prompt = """You are an evaluator that judges how informative a document is for answering a given question. You will receive a Question and a Document.

Carefully assess relevance and usefulness (reason internally), then output only a score and a brief, evidence-based comment that supports downstream filtering (mention concrete entities/claims/dates/metrics when helpful).

Scoring rubric (1-5):
1 — Unrelated: The document has nothing to do with the question. It does not contain any potentially relevant information or an answer to the question.
2 — Loosely related: Contains information that might potentially help or include the answer to the question, but is unlikely to do so.
3 — Partially informative: Contains information that can potentially help answer the question in some way.
4 — Substantively informative: Related to the question and includes information that is relevant to it.
5 — Direct answer: Clearly related and includes key information that can be used to directly answer the question.

Output format (exactly):
Comment: <concise justification citing specific evidence from the document; e.g., “Since the document states A and B, it is relevant to the question about C.”>
Score: <1-5>
"""
        prompts_with_chat_template = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

        # --- MODIFICATION: `get_completion` no longer needs client arg ---
        tasks = [
            self.get_completion(
                self.model_name_or_path,
                prompt_with_chat_template,
                self.sampling_params,
                self.semaphore,
            )
            for prompt_with_chat_template in prompts_with_chat_template
        ]
        # --- END MODIFICATION ---

        outputs = await tqdm_asyncio.gather(*tasks)

        # Pre-allocate lists with None values
        total_length = len(prompts)
        all_outputs = [None] * total_length
        all_output_token_counts = [None] * total_length
        all_scores = [None] * total_length
        all_reasonings = [None] * total_length

        # Process complete responses first
        for i, output in enumerate(outputs):
            # --- ADDED: Handle failed requests ---
            if output is None:
                all_outputs[i] = "Error: Request failed"
                all_output_token_counts[i] = 0
                all_scores[i] = (0, 0)
                all_reasonings[i] = "Error: Request failed"
                continue
            # --- END ADDED ---

            text = output.text
            tokens = output.logprobs.tokens
            token_count = len(tokens)

            reasoning = (
                text.split("Comment:")[1].split("Score:")[0].strip()
                if "Comment:" in text and "Score:" in text
                else None
            )
            all_reasonings[i] = reasoning

            score_token_idx = None
            if len(tokens) < 6:
                inspecting_range = -1
            else:
                inspecting_range = len(tokens) - 6
            for idx in range(len(tokens) - 1, inspecting_range, -1):
                if tokens[idx] in set([str(i) for i in range(1, self.max_score + 1)]):
                    score_token_idx = idx
                    break

            if score_token_idx is None:
                all_outputs[i] = text
                all_output_token_counts[i] = token_count
                all_scores[i] = (0, 0)
                continue

            score_integer = int(tokens[score_token_idx])
            logprob_score = output.logprobs.token_logprobs[score_token_idx]

            all_outputs[i] = text
            all_output_token_counts[i] = token_count
            all_scores[i] = (score_integer, logprob_score)

        return (
            all_outputs,
            all_reasonings,
            all_output_token_counts,
            all_scores,
            prompts_with_chat_template,
        )

    def return_prompt(self, query, doc_content) -> str:
        return f"""Question: {query}
Document: {doc_content}
"""  # force the model to start with this

    @torch.inference_mode()
    async def predict(self, queries, passages, **kwargs):
        """This is setup to run with mteb but can be adapted to your purpose
        input_to_rerank: {"queries": queries, "documents": documents}
        """
        prompts = [
            self.return_prompt(query, passage)
            for query, passage in zip(queries, passages)
        ]
        # print(f"Example prompt: ```\n{prompts[0]}\n```")

        (
            texts,
            reasonings,
            token_counts,
            scores,
            prompts,
        ) = await self._process_with_vllm(prompts)
        return texts, reasonings, token_counts, scores, prompts

    async def rerank(self, query: str) -> RerankResult:
        documents = self.retriever.retrieve(
            query, self.retriever_config["top_k_initial"]
        )
        queries = [query] * len(documents)
        corpus = [doc.text for doc in documents]
        _, reasonings, token_counts, scores = await self.predict(
            queries=queries, passages=corpus
        )
        # Sort documents and reasoning by scores
        for document, score in zip(documents, scores):
            document.score = score
        sorted_indices = sorted(
            range(len(scores)), key=lambda i: (scores[i][0], scores[i][1]), reverse=True
        )
        sorted_documents = [documents[i] for i in sorted_indices]
        sorted_reasonings = [reasonings[i] for i in sorted_indices]

        return RerankResult(documents=sorted_documents, reasoning=sorted_reasonings)

    async def batch_rerank(
        self,
        queries: List[str],
        retriever_initial_topk: int,
        search_turns_stats: List[Any] = None,
        search_original_batch_idx: List[int] = None,
    ) -> List[RerankResult]:
        batched_documents = self.retriever.batch_retrieve(
            queries, retriever_initial_topk
        )
        num_queries = len(queries)
        queries_extended = []
        for query, documents in zip(queries, batched_documents):
            queries_extended.extend([query] * len(documents))
        retriever_k = len(queries_extended) // num_queries if num_queries > 0 else 0
        corpus = []
        for documents in batched_documents:
            for doc in documents:
                corpus.append(doc.text)
        if len(corpus) == 0 or len(queries_extended) == 0:
            return []

        texts, reasonings, token_counts, scores, prompts = await self.predict(
            queries=queries_extended, passages=corpus
        )

        # Split reasonings and scores back to per-query lists
        split_texts = []
        split_reasonings = []
        split_scores = []
        split_prompts = []
        split_token_counts = []
        for i in range(num_queries):
            split_texts.append(texts[i * retriever_k : (i + 1) * retriever_k])
            split_reasonings.append(reasonings[i * retriever_k : (i + 1) * retriever_k])
            split_scores.append(scores[i * retriever_k : (i + 1) * retriever_k])
            split_prompts.append(prompts[i * retriever_k : (i + 1) * retriever_k])
            split_token_counts.append(
                sum(token_counts[i * retriever_k : (i + 1) * retriever_k])
            )
        results = []
        # Sort documents and reasoning by scores
        for i, (
            texts,
            reasonings,
            scores,
            prompts,
            token_count,
            query,
            search_turns_stat,
            search_original_batch_i,
        ) in enumerate(
            zip(
                split_texts,
                split_reasonings,
                split_scores,
                split_prompts,
                split_token_counts,
                queries,
                search_turns_stats,
                search_original_batch_idx,
            )
        ):
            documents = batched_documents[i]
            for document, score in zip(documents, scores):
                document.score = score
            sorted_indices = sorted(
                range(len(scores)),
                key=lambda i: (scores[i][0], scores[i][1]),
                reverse=True,
            )
            sorted_documents = [documents[k] for k in sorted_indices]
            sorted_texts = [texts[k] for k in sorted_indices]
            sorted_reasonings = [reasonings[k] for k in sorted_indices]
            sorted_scores = [scores[k][0] for k in sorted_indices]
            sorted_logits = [scores[k][1] for k in sorted_indices]
            sorted_prompts = [prompts[k] for k in sorted_indices]
            query = (
                sorted_prompts[0].split("Question:")[1].split("Document:")[0].strip()
            )

            if self.jsonl_file_path is not None:
                # Save to JSONL file
                output_data = {
                    "query": query,
                    "search_turns_stat": search_turns_stat,
                    "search_original_batch_i": search_original_batch_i,
                    "ranked_documents": [
                        {
                            "id": doc.id,
                            "document": doc.text,
                            "response": text,
                        }
                        for doc, text in zip(sorted_documents, sorted_texts)
                    ],
                }
                with open(self.jsonl_file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_data, ensure_ascii=False) + "\n")

            results.append(
                RerankResult(
                    documents=sorted_documents,
                    reasoning=sorted_reasonings,
                    total_token_count=token_count,
                    score=sorted_scores,
                    logit=sorted_logits,
                )
            )
        return results


# --- 4. FastAPI Application Setup ---
app = FastAPI()
reranker_model: Optional[Any] = None


@app.post("/rerank/single")
async def rerank_endpoint(request: RerankSingleRequest):
    """Endpoint to rerank documents using the loaded model."""
    if reranker_model is None:
        raise HTTPException(status_code=503, detail="Reranker model not loaded.")
    result = await reranker_model.rerank(request.query)

    return {
        "documents": [d.__dict__ for d in result.documents],
        "reasonings": result.reasoning,
    }


@app.post("/rerank/batch")
async def rerank_endpoint(request: RerankBatchRequest):
    """Endpoint to rerank documents using the loaded model."""
    if reranker_model is None:
        raise HTTPException(status_code=503, detail="Reranker model not loaded.")

    results = await reranker_model.batch_rerank(
        request.queries,
        request.retriever_initial_topk,
        request.search_turns_stats,
        request.search_original_batch_idx,
    )

    return [
        {
            "documents": [d.__dict__ for d in result.documents],
            "reasonings": result.reasoning,
            "total_token_count": result.total_token_count,
            "scores": result.score,
            "logits": result.logit,
        }
        for result in results
    ]


@app.post("/shutdown")
def shutdown_endpoint():
    """Endpoint to gracefully shut down the Uvicorn server."""
    print("Shutdown request received. Terminating server.")

    # Get the Process ID (PID) of the current process
    pid = os.getpid()

    # Send the SIGINT signal (same as Ctrl+C) to the process
    # Uvicorn is designed to catch this and shut down cleanly.
    os.kill(pid, signal.SIGINT)

    return {"message": "Server is shutting down."}


# --- 5. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch a server for a single reranker model."
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="The name or path of the reranker to load.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8192,
        help="The maximum output token length of the model.",
    )
    parser.add_argument(
        "--max-score",
        type=int,
        default=5,
        help="Maximum scale of relevance score.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="The concurrency for asyncio.",
    )
    parser.add_argument(
        "--retriever-url",
        type=str,
        required=True,
        help="The url for the retriever.",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        required=True,
        help="The url for the vLLM server.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="The temperature to use for sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="The top-p sampling probability.",
    )
    parser.add_argument(
        "--jsonl_file_path",
        type=str,
        default=None,
        help="Path to the JSONL file containing the data.",
    )
    args = parser.parse_args()

    print(f"Using single reranker model: {args.model_name_or_path}")
    reranker_model = SplitRAGReranker(
        model_name_or_path=args.model_name_or_path,
        max_output_tokens=args.max_output_tokens,
        max_score=args.max_score,
        concurrency=args.concurrency,
        retriever_url=args.retriever_url,
        vllm_url=args.vllm_url,
        temperature=args.temperature,
        top_p=args.top_p,
        jsonl_file_path=args.jsonl_file_path,
    )

    print(f"Model {args.model_name_or_path} loaded. Starting server...")

    uvicorn.run(app, host="0.0.0.0", port=8005)
