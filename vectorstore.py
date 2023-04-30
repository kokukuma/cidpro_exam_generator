import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

from lib.stdin_loader import STDInLoader

def read_from_std():
    print("Input text, Ctrl+D to finish")
    text = sys.stdin.read()
    loader = STDInLoader(text)
    pages = loader.load_and_split()
    if len(pages) < 4:
        pages = pages * 4
    return Chroma.from_documents(pages, embedding=OpenAIEmbeddings())

def read_from_url(url):
    loader = UnstructuredURLLoader(urls=[url])
    pages = loader.load_and_split()
    if len(pages) < 4:
        pages = pages * 4
    return Chroma.from_documents(pages, embedding=OpenAIEmbeddings())

def read_from_source_dict(files, persist_directory):
    vectorstore = None
    documents = []
    nonexisting_files = files
    existing_source = set()
    embeddings = OpenAIEmbeddings()

    if os.path.exists(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        existing_source = set([k["source"] for k in vectorstore._collection.get()["metadatas"]])
        nonexisting_files = [f for f in files if f not in existing_source]
        if len(nonexisting_files) == 0:
            return vectorstore

    print(nonexisting_files)

    # load from urls
    documents.extend(_read_urls(nonexisting_files))

    # load from local pdf_docs directly
    documents.extend(_read_local_pdf(nonexisting_files))

    # split docuemnts
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
    documents = text_splitter.split_documents(documents)

    if len(documents) > 0:
        if vectorstore:
            vectorstore.add_documents(documents)
        else:
            vectorstore = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persist_directory)
        vectorstore.persist()
    return vectorstore

def _read_urls(files):
    urls =[u for u in files if u.startswith('https://')]
    loader = UnstructuredURLLoader(urls=urls)
    return loader.load()

def _read_local_pdf(files):
    documents = []
    pdfs =[u for u in files if u.endswith('.pdf')]
    for pdf in pdfs:
        if os.path.isfile(pdf):
            loader = UnstructuredPDFLoader(pdf)
            documents.extend(loader.load())
    return documents
