import os, requests, json, time, sys
import multiprocessing

def get_plain_text_from_url(url):
    from bs4 import BeautifulSoup
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except:
        return ""


def llm_answer(question, relevant_content, llm):
    #Given the above, concisely, in as few words as possible, answer the following question:
    return llm(f"""
    {relevant_content}

    Given the above, answer the following question:

    {question}
    """)

def parallel(func, args):
    t0 = time.time()
    cpus = multiprocessing.cpu_count()
    cpus = 1
    with multiprocessing.Pool(processes=cpus) as pool:
        result = pool.map(func, args)
    t1 = time.time()
    print(f"Parallel execution took {t1-t0} seconds for {str(func)} on {len(args)} items")
    return result

def find_relevant_content(query, content, chunk_size=500, k=5):
    from ragatouille import RAGPretrainedModel
    from ragatouille.data import CorpusProcessor

    t0 = time.time()
    corpus_processor = CorpusProcessor()
    documents = [x["content"] for x in corpus_processor.process_corpus(content, chunk_size=chunk_size)]
    t1 = time.time()
    print(f"Processed {len(documents)} documents in {t1-t0} seconds.. reranking with Colbert")

    t0 = time.time()
    colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    #print(query)
    #print(documents)
    relevant_content = [x["content"] for x in colbert.rerank(query=query, documents=documents, k=k)]
    t1 = time.time()
    print(f"Got {len(relevant_content)} relevant documents in {t1-t0} seconds..")
    return relevant_content

def read_urls(urls, query=None, assess_relevance=False, chunk_size=1000, k=5):
    t0 = time.time()
    content = parallel(get_plain_text_from_url, urls)
    t1 = time.time()
    print(f"Got {len(content)} documents..")

    if assess_relevance:
        print("Assessing relevance..")
        return find_relevant_content(query, content, chunk_size, k)

    return content

def search(query: str):
    from duckduckgo_search import DDGS
    ddgs = DDGS()
    results = ddgs.text(query)
    return [result['href'] for result in results]


def research(question: str, llm, k=8):
    from duckduckgo_search import DDGS

    t_start = time.time()
    t0 = time.time()
    urls = search(question)
    t1 = time.time()
    print(f"Got {len(urls)} search results in {t1-t0} seconds..")

    content = parallel(get_plain_text_from_url, urls)
    print(f"Got {len(content)} documents.. splitting into chunks")

    relevant_content = find_relevant_content(question, content, llm)

    #answers = parallel(llm, relevant_content)

    t0 = time.time()
    final_answer = llm_answer(query, "\n\n...".join(relevant_content), llm)
    t1 = time.time()
    print(f"{final_answer} \n\nLLM returned in {t1-t0} seconds..")
    t_end = time.time()
    print(f"Total time: {t_end-t_start} seconds..")
    return (final_answer, relevant_content)
