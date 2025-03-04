from flask import Flask, render_template, request, jsonify
import openai
from pinecone import Pinecone
from datasets import load_dataset
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "code-assistant"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine'
    )
index = pc.Index(index_name)

def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

def load_and_process_data():
    dataset = load_dataset("code_search_net", "python", split="train", streaming=True)
    for i, item in enumerate(dataset.take(1000)):
        code = item['code']
        embedding = get_embedding(code)
        index.upsert(vectors=[(str(i), embedding, {"code": code})])

def semantic_search(query, k=3):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=k, include_metadata=True)
    return [match['metadata']['code'] for match in results['matches']]

def generate_response(query, context):
    prompt = f"Context:\n{context}\n\nQuery: {query}\n\nResponse:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json['query']
    relevant_chunks = semantic_search(user_query)
    context = "\n".join(relevant_chunks)
    response = generate_response(user_query, context)
    return jsonify({'response': response})

if __name__ == '__main__':
    load_and_process_data()
    app.run(debug=True)
