import boto3
import json
from typing import List, Literal, Tuple
from multiprocessing.pool import ThreadPool
import boto3
from pinecone import Pinecone
import torch
from functools import cache
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from functools import cache
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection

AWS_PROFILE_NAME = "sigir-participant"
AWS_REGION_NAME = "us-east-1"
OPENSEARCH_INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"

def get_ssm_value(key: str, profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME) -> str:
    """Get a cleartext value from AWS SSM."""
    session = boto3.Session(profile_name=profile, region_name=region)
    ssm = session.client("ssm")
    return ssm.get_parameter(Name=key)["Parameter"]["Value"]

def get_ssm_secret(key: str, profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME):
    session = boto3.Session(profile_name=profile, region_name=region)
    ssm = session.client("ssm")
    return ssm.get_parameter(Name=key, WithDecryption=True)["Parameter"]["Value"]


PINECONE_INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"
PINECONE_NAMESPACE="default"

@cache
def has_mps():
    return torch.backends.mps.is_available()

@cache
def has_cuda():
    return torch.cuda.is_available()

@cache
def get_tokenizer(model_name: str = "intfloat/e5-base-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

@cache
def get_model(model_name: str = "intfloat/e5-base-v2"):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    if has_mps():
        model = model.to("mps")
    elif has_cuda():
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    return model

def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def embed_query(query: str,
                query_prefix: str = "query: ",
                model_name: str = "intfloat/e5-base-v2",
                pooling: Literal["cls", "avg"] = "avg",
                normalize: bool =True) -> list[float]:
    return batch_embed_queries([query], query_prefix, model_name, pooling, normalize)[0]

def batch_embed_queries(queries: List[str], query_prefix: str = "query: ", model_name: str = "intfloat/e5-base-v2", pooling: Literal["cls", "avg"] = "avg", normalize: bool =True) -> List[List[float]]:
    with_prefixes = [" ".join([query_prefix, query]) for query in queries]
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)
    with torch.no_grad():
        encoded = tokenizer(with_prefixes, padding=True, return_tensors="pt", truncation="longest_first")
        encoded = encoded.to(model.device)
        model_out = model(**encoded)
        match pooling:
            case "cls":
                embeddings = model_out.last_hidden_state[:, 0]
            case "avg":
                embeddings = average_pool(model_out.last_hidden_state, encoded["attention_mask"])
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()

@cache
def get_pinecone_index(index_name: str = PINECONE_INDEX_NAME):
    pc = Pinecone(api_key=get_ssm_secret("/pinecone/ro_token"))
    index = pc.Index(name=index_name)
    return index

def query_pinecone(query: str, top_k: int = 10, namespace: str = PINECONE_NAMESPACE) -> dict:
    index = get_pinecone_index()
    results = index.query(
        vector=embed_query(query),
        top_k=top_k,
        include_values=False,
        namespace=namespace,
        include_metadata=True
    )

    return results

@cache
def get_client(profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME):
    credentials = boto3.Session(profile_name=profile).get_credentials()
    auth = AWSV4SignerAuth(credentials, region=region)
    host_name = get_ssm_value("/opensearch/endpoint", profile=profile, region=region)
    aos_client = OpenSearch(
        hosts=[{"host": host_name, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )
    return aos_client

def query_opensearch(query: str, top_k: int = 10) -> dict:
    """Query an OpenSearch index and return the results."""
    client = get_client()
    results = client.search(index=OPENSEARCH_INDEX_NAME, body={"query": {"match": {"text": query}}, "size": top_k})
    return results

def get_pinecone_results(questions: list[dict], num_threads: int = 4) -> list[dict]:
    def process_question(q):
        matches = []
        # Add progress bar for sub-questions
        for sub_q in tqdm(q['sub-questions'], 
                         desc=f"Processing sub-questions for Q{q['id']}", 
                         leave=False):
            query_result = query_pinecone(sub_q, top_k = 10)
            # Convert each match to a dictionary with only serializable values
            for match in query_result['matches']:
                match_dict = {
                    'id': match['id'],
                    'score': match['score'],
                    'metadata': match['metadata']
                }
                matches.append(match_dict)
        
        return {
            'question_id': q['id'],
            'question': q['question'],
            'docs': matches
        }

    # Use ThreadPool to process questions in parallel
    with ThreadPool(processes=num_threads) as pool:
        results = list(tqdm(pool.imap(process_question, questions), 
                          total=len(questions),
                          desc=f"Processing main questions with {num_threads} threads"))

    # Save results to json file
    with open('pinecone_results_with_sub_questions_session1.json', 'w') as f:
        json.dump(results, f, indent = 4)
   
def get_opensearch_results(questions: list[dict], num_threads: int = 4) -> list[dict]:
    def process_question(question):
        matches = []
        # Add progress bar for sub-questions
        for sub_q in tqdm(question['sub-questions'], 
                         desc=f"Processing sub-questions for Q{question['id']}", 
                         leave=False):
            query_result = query_opensearch(sub_q, top_k = 10)
            
            for match in query_result['hits']['hits']:
                match_dict = {
                    'id': match['_id'],
                    'score': match['_score'],
                    'metadata': match['_source']
                }
                matches.append(match_dict)
            
        return {
            'question_id': question['id'],
            'question': question['question'],
            'docs': matches
        }

    # Use ThreadPool to process questions in parallel
    with ThreadPool(processes=num_threads) as pool:
        results = list(tqdm(pool.imap(process_question, questions), 
                          total=len(questions),
                          desc=f"Processing main questions with {num_threads} threads"))

    # Save results to json file
    with open('opensearch_results_with_sub_questions_session1.json', 'w') as f:
        json.dump(results, f, indent = 4)

def main():
    questions = []
    with open('LiveRAG_LCD_Session1_Question_with_subqueries.jsonl', 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    
    num_threads = 8
    get_pinecone_results(questions, num_threads)
    get_opensearch_results(questions, num_threads)

if __name__ == "__main__":
    main()