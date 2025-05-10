from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever

from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle, Node

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core import Settings

from typing import Optional, List, Mapping, Any
from typing import Any, Iterator
import asyncio

from llm_scratch import gemini as llm

class OurCustomLLM(CustomLLM):
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "custom_model"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        print(prompt)
        return CompletionResponse(text=llm(prompt))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> Iterator[CompletionResponse]:
        # Implement streaming completion if your model supports it
        # For this example, we'll just yield the entire response at once
        yield CompletionResponse(text=llm(prompt))

class DuckDuckGoRetriever(BaseRetriever):
    def _retrieve(self, query_bundle):
        query = query_bundle.query_str
        urls = search(query)
        content = read_urls(urls)
        results = find_relevant_content(query, content, chunk_size=200, k=2)
        nodes = []
        for i, result in enumerate(results):
            if isinstance(result, str):
                node = TextNode(text=result)
                score = 1.0 / (i + 1)
                nodes.append(NodeWithScore(node=node, score=score))
            else:
                print(f"Unexpected result type: {type(result)}")
        
        return nodes

def calculate(what):
    """Runs a calculation and returns the number - must be valid Python code"""
    return eval(what)

def search_web(query: str) -> List[str]:
    """
    Useful for when you need to search the internet for current information on a topic.
    """
    retriever = DuckDuckGoRetriever()
    results = retriever.retrieve(query)
    return [node.node.text for node in results]

def load_url(url):
    """Load the text content of a URL."""
    return get_plain_text_from_url(url)

custom_llm = OurCustomLLM()
workflow = AgentWorkflow.from_tools_or_functions(
    [search_web, load_url, calculate],
    llm=custom_llm,
    system_prompt="You are a helpful assistant that can search the web for information.",
)

async def query(query_str: str) -> List[str]:
    return await workflow.run(user_msg=query_str)

def q(query_str: str) -> List[str]:
    return asyncio.run(query(query_str))

print(q("For aesthetics, what is the optimal number of paddles in a water wheel?"))
