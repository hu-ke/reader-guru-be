import os
import tempfile
from langchain.document_loaders import PyPDFLoader, UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from sklearn.cluster import KMeans
import numpy as np
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from fastapi import FastAPI, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from extractImg import extract_cover
from turnTextIntoTokens import num_tokens_from_string
from dotenv import load_dotenv
import bmemcached
load_dotenv()
BOOKS_DIR = Path() / 'books'
BOOKS_COVERS_DIR = Path() / 'book_covers'

app = FastAPI()
origins = [
    "http://localhost:5173",
    "https://reader.guru",
    "http://reader.guru"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_api_key = os.getenv('OPENAI_API_KEY')
client = bmemcached.Client((os.getenv('MEMCACHED_HOST')), os.getenv('MEMCACHED_ACCOUNT_ID'), os.getenv('MEMCACHED_PWD'))

def load_book(file_obj, file_extension):
    """Load the content of a book based on its file type."""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file_obj.read())
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file.name)
            pages = loader.load()
            text = "".join(page.page_content for page in pages)
        elif file_extension == ".epub":
            loader = UnstructuredEPubLoader(temp_file.name)
            data = loader.load()
            text = "\n".join(element.page_content for element in data)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        os.remove(temp_file.name)
    text = text.replace('\t', ' ')
    return text

def split_and_embed(text, openai_api_key):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000)
    docs = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    content_list = []
    for x in docs:
        content_list.append(x.page_content)
    vectors = embeddings.embed_documents(content_list)
    tokens = num_tokens_from_string(' '.join(content_list), 'cl100k_base')
    # vectors = embeddings.embed_documents([x.page_content for x in docs])
    return docs, vectors, embeddings, tokens


def cluster_embeddings(vectors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    closest_indices = [np.argmin(np.linalg.norm(vectors - center, axis=1)) for center in kmeans.cluster_centers_]
    return sorted(closest_indices)


def summarize_chunks(docs, selected_indices, openai_api_key):
    llm3_turbo = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=1000, model='gpt-3.5-turbo-16k')
    map_prompt = """
    这里有一段来自一本书的段落。你的任务是对这段文字进行全面的总结。确保准确性，避免添加任何不在原文中的解释或额外细节。摘要至少应有三段，完整地表达出原文的要点。
    ```{text}```
    总结:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    selected_docs = [docs[i] for i in selected_indices]
    summary_list = []
    chain = load_summarize_chain(llm=llm3_turbo, chain_type="stuff", prompt=map_prompt_template)

    for doc in selected_docs:
        chunk_summary = chain.run([doc])
        print(chunk_summary)
        summary_list.append(chunk_summary)
    
    return "\n".join(summary_list)

def create_final_summary(summaries, openai_api_key):
    llm4 = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=3000, model='gpt-4', request_timeout=120)
    combine_prompt = """
    这里有一些段落摘要，它们来自于一本书。你的任务是把这些摘要编织成一个连贯且详细的总结。读者应该能够从你的总结中理解书中的主要事件或要点。确保保持内容的准确性，并以清晰而引人入胜的方式呈现。
    ```{text}```
    综合总结:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    reduce_chain = load_summarize_chain(llm=llm4, chain_type="stuff", prompt=combine_prompt_template)
    final_summary = reduce_chain.run([Document(page_content=summaries)])
    return final_summary

def extract_book_texts(uploaded_file):
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    text = load_book(uploaded_file, file_extension)
    return text 

@app.post("/api/uploadFile/")
async def create_upload_file(file_upload: UploadFile, deviceId: str = Header(None, alias="deviceId")):
    print('start uploading.')
    data = await file_upload.read()
    personal_book_directory = BOOKS_DIR / deviceId
    target_file =  personal_book_directory / file_upload.filename
    with open(target_file, 'wb') as f:
        f.write(data)

    return {
        'code': 200,
        'msg': 'Your book has been uploaded successfully! And now we are creating workspace for your book, just a moment.',
        'data': {
            'fileName': file_upload.filename
        }
    }

@app.post("/api/queryBook")
async def query_book(request: dict, deviceId: str = Header(None, alias="deviceId")):
    query = request['query']
    mem_key_prefix = deviceId + "_" + request['filename']
    # get documents
    doc_key = f'{mem_key_prefix}_doc'
    documents = client.get(doc_key)
    print(len(documents))
    db = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=openai_api_key))
    answer = db.similarity_search(query)
    return {
        'code': 200,
        'msg': 'successful',
        'data': {
            'answer': len(answer)
        }
    }


@app.post("/api/generateFileInfo")
async def generate_file_info(request: dict, deviceId: str = Header(None, alias="deviceId")):
    print('start generating file info.', request['filename'])
    personal_book_directory = BOOKS_DIR / deviceId
    target_file =  personal_book_directory / request['filename']

    cover_name = os.path.splitext(request['filename'])[0].lower()
    cover_name = f"{cover_name}.png"

    target_book_cover = BOOKS_COVERS_DIR / cover_name
    if not os.path.exists(personal_book_directory):
        os.makedirs(personal_book_directory)
    print('start to generate book cover')
    extract_cover(target_file, target_book_cover)
    print('book cover generated.')
    mem_key_prefix = deviceId + "_" + request['filename']
    print(mem_key_prefix)

    with open(target_file, 'rb') as file:
        texts = extract_book_texts(file)
        print('text generated.')
        
    docs, vectors, embeddings, tokens = split_and_embed(texts, openai_api_key)
    print('docs, vectors, tokens generated.')
    # memcache documents
    doc_key = f'{mem_key_prefix}_doc'
    client.set(doc_key, docs, 600)
    # memcache vectors
    vector_key = f'{mem_key_prefix}_vector'
    client.set(vector_key, vectors, 600)
    # memcache embeddings
    embedding_key = f'{mem_key_prefix}_embedding'
    print('embedding_key', embedding_key)
    # client.set(embedding_key, embeddings, 600)
    # print('embedding is ok.')

    tokens_of_first_doc = num_tokens_from_string(docs[0].page_content, 'cl100k_base')

    return {
        'code': 200,
        'msg': 'The workspace is ready. Now you can summarize or chat with the chatbot.',
        'data': {
            'coverImgUrl': f'http://reader.guru/images/{cover_name}',
            'numsOfTokens': tokens,
            'numsOfDocs': len(docs),
            'tokensOfFirstDoc': tokens_of_first_doc,
            'fileName': request['filename']
        }
    }
    

@app.post("/api/summarize")
async def summarize_file(request: dict, deviceId: str = Header(None, alias="deviceId")):
    print('start summarizing')
    mem_key_prefix = deviceId + "_" + request['filename']
    vectors = client.get(f'{mem_key_prefix}_vector')
    docs = client.get(f'{mem_key_prefix}_doc')
    print(len(vectors))
    selected_indices = cluster_embeddings(vectors, 11)
    summaries = summarize_chunks(docs, selected_indices, openai_api_key)
    final_summary = create_final_summary(summaries, openai_api_key)

    return {
        'code': 200,
        'msg': 'the summarization has been generated successfully.',
        'data': {
            'fileName': request['filename'],
            'summary': final_summary,
        }
    }

@app.get('/api')
async def root(): 
    return { 'message': 'Hello World'}

# Testing the summarizer
# if __name__ == '__main__':
#     openai_api_key = os.getenv('OPENAI_API_KEY')
#     # book_path = "./thethreekingdoms.pdf"
#     # book_path = './wentiejun.pdf'
#     book_path = './IntoThinAirBook.pdf'
#     with open(book_path, 'rb') as uploaded_file:
#         summary = generate_summary(uploaded_file, openai_api_key, verbose=True)
#         print(summary)