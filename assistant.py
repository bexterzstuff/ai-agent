import ollama
import chromadb
import sqlite3
import json
import psycopg
from psycopg.rows import dict_row


client = chromadb.Client()
convo = []
DB_PARAMS = {
    'dbname': 'vector.s3db'
}


def connect_db():
    conn = sqlite3.connect(DB_PARAMS['dbname'])
    return conn


def fetch_conversations(json_str=True):
    conn = connect_db()
    conn.row_factory = sqlite3.Row
    db = conn.cursor()
    conversations = db.execute('''
        SELECT * FROM conversations
    ''').fetchall()
    conn.commit()
    conn.close()
    return conversations


def store_conversations(prompt, response):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversations (timestamp, prompt, response) VALUES (current_timestamp, ?, ?)", (prompt, response) )
    conn.commit()
    conn.close()


def stream_response(prompt):
    convo.append({'role': 'user', 'content': prompt})
    response = ''
    stream = ollama.chat(model='qwen2:1.5b', messages=convo, stream=True)
    print('\nAssistant:')

    for chunk in stream:
        content = chunk['message']['content']
        response += content
        print(content, end='', flush=True)

    print('\n')
    store_conversations(prompt=prompt, response=response)
    convo.append({'role': 'assistant', 'content': response})


def create_vector_db(conversations):
    vector_db_name = 'conversations'

    try:
        client.delete_collection(name=vector_db_name)
    except ValueError:
        pass

    vector_db = client.create_collection(name=vector_db_name)

    for c in conversations:
        serialized_convo = f"prompt: {c['prompt']} response: {c['response']}"
        response = ollama.embeddings(model='nomic-embed-text', prompt=serialized_convo)
        embedding = response['embedding']

        vector_db.add(
            ids=[str(c['id'])],
            embeddings=[embedding],
            documents=[serialized_convo]
        )


def retrieve_embeddings(prompt):
    response = ollama.embeddings(model='nomic-embed-text', prompt=prompt)
    prompt_embedding = response['embedding']

    vector_db = client.get_collection(name='conversations')
    results = vector_db.query(query_embeddings=[prompt_embedding], n_results=1)
    best_embedding = results['documents'][0][0]

    return best_embedding


conversations = fetch_conversations()
create_vector_db(conversations=conversations)
print(fetch_conversations())

while True:
    try:
        prompt = input('User: \n')
        if prompt == 'exit':
            break

        context = retrieve_embeddings(prompt=prompt)
        prompt = f'USER PROMPT: {prompt} \nCONTEXT FROM EMBEDDINGS: {context}'
        stream_response(prompt=prompt)

    except KeyboardInterrupt:
        print("Bye")
        sys.exit()

