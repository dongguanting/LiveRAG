from ai71 import AI71
import json
from tqdm import tqdm
import pandas as pd
import time


# setting
AI71_API_KEY = ""
retrieval_results_path = "pinecone_results.json"
corrected_questions_path = "corrected_questions.jsonl"
top_k = 10

# define prompt
def rag_prompt(question, passages):
    passage_text = "\n".join([f"- Passage {i+1}: {p}" for i, p in enumerate(passages)])
    prompt = f"""Find the useful content from the provided documents, then answer the question. Answer the question directly. Your response should be very concise.
    
    The following are given documents:
    {passage_text}
    
    Answer the question directly. Your response should be very concise.
    **Question**: {question}
    **Response**:"""
    return prompt

# load client
client = AI71(AI71_API_KEY)

# load data
with open(retrieval_results_path, "r") as f:
    data = json.load(f)[:10]

# correct spelling
with open(corrected_questions_path, "r") as f:
    corrected_questions = pd.read_json(f, lines=True).to_dict(orient="records")
for i in range(len(data)):
    data[i]["corrected_question"] = corrected_questions[i]['corrected_question']

# define output
answers = {
    "id": [],
    "question": [],
    "passages": [],
    "final_prompt": [],
    "answer": []
}

# generation
for row in tqdm(data):
    # prepare input
    question = row["question"]
    corrected_question = row["corrected_question"]
    passages = [row['docs'][i]['metadata']['text'] for i in range(top_k)]
    prompt = rag_prompt(corrected_question, passages)

    # generate
    response = client.chat.completions.create(
        model="tiiuae/falcon-180B-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    ).choices[0].message.content

    # update results
    answers['id'].append(int(row["question_id"]))
    answers['question'].append(question)
    answers['passages'].append([
        {
            "passage": row['docs'][i]['metadata']['text'],
            "doc_IDs": [row['docs'][i]['metadata']['doc_id']]
        }
        for i in range(top_k)])
    answers['final_prompt'].append(prompt)
    answers['answer'].append(response)

    time.sleep(0.1)

# save results
pd.DataFrame(answers).to_json("answers.jsonl", orient='records', lines=True, force_ascii=False)