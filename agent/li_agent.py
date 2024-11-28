from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle, Node
from llama_index.core import Settings

from llama_index.core.agent import ReActAgent

from typing import Optional, List, Mapping, Any
from typing import Any, Iterator

#from llm_scratch import llama_cpp as llm
from llm_scratch import meta_llama as llm
from llm_scratch import search, read_urls, find_relevant_content, get_plain_text_from_url
from llm_scratch import text_to_voice

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
        #print(prompt)
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

# direct retriever use
#retriever = DuckDuckGoRetriever()
#query_engine = RetrieverQueryEngine.from_args(retriever=retriever)
#response = query_engine.query("Who shot JR?")
#print(response.response)

def calculate(what):
    """Runs a calculation and returns the number - must be valid Python code"""
    return eval(what)
calculate_tool = FunctionTool.from_defaults(fn=calculate)

def search_duckduckgo(query: str) -> List[str]:
    retriever = DuckDuckGoRetriever()
    results = retriever.retrieve(query)
    return [node.node.text for node in results]
duckduckgo_tool = FunctionTool.from_defaults(
    fn=search_duckduckgo,
    name="DuckDuckGo",
    description="Useful for when you need to search the internet for current information on a topic."
)

def load_url(url):
    return get_plain_text_from_url(url)
load_url_tool = FunctionTool.from_defaults(
    fn=load_url,
    name="LoadURL",
    description="Load the text content of a URL."
)

tools = [duckduckgo_tool, calculate_tool, load_url_tool]

custom_llm = OurCustomLLM()
Settings.llm = custom_llm

agent = ReActAgent.from_tools(tools, llm=custom_llm, verbose=True, max_iterations=50)
def q(question):
    response = agent.chat(question)
    #text_to_voice(response.response, "/home/dev/test.mp3")
    return response.response

#q("How many lego pieces are made each day?")
#q("How many hotdogs would it take to reach the moon?")
