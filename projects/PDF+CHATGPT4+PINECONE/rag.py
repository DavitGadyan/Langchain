import os
import time
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# load environemtal variables API KEY for OPENAI & PINECONE
load_dotenv()

# load pdf documents
loader = DirectoryLoader('shell-payment2gov', glob="*.pdf", show_progress=True, use_multithreading=True)
docs = loader.load()

# split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=2000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
docs_splitted = text_splitter.split_documents(docs)

# define pinecone object
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# define index and if does not exist create
index_name = "shell"  
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# add splitted documents to Pinecone
vectorstore = PineconeVectorStore(index_name=index_name, embedding=OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-3-large",))
vectorstore.add_documents(docs_splitted)

# CHAT GPT4.o model set up
model = ChatOpenAI(model="gpt-4o", openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0)

# define PromptTemplate
template = """Answer the question based on the context:{context}. Question: {question}"""
prompt = ChatPromptTemplate.from_template(template)

# define retriever
docsearch = PineconeVectorStore.from_documents(docs_splitted, OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-3-large",), index_name=index_name)
retriever = docsearch.as_retriever(search_type="mmr")

# create the chain and run it
chain = (
  {"context": retriever, "question": RunnablePassthrough()}
  | prompt
  | model
  | StrOutputParser())

# invoke chain with question
question = "For each year from 2012-2021 which was the largest country by year where Shell paid the most royalties? Return Year-Country-Amount in USD"
result = chain.invoke(question)

print(f'Based on question: {question}\nRAG answer: {result}')
