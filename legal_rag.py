"""
LexiQuery — Legal Document Q&A
Standard RAG | Domain: Legal Tech | API: Google Gemini Flash
"""

import os
import glob
from dotenv import load_dotenv

# Modern LangChain Ecosystem Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Import chains from the classic package
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load API keys from .env file
load_dotenv()
# Read GOOGLE_API_KEY as defined in your environment file
API_KEY = os.getenv('GOOGLE_API_KEY')  

if not API_KEY:
    raise ValueError("CRITICAL ERROR: GOOGLE_API_KEY is not set or couldn't be loaded from your .env file.")

# Folder containing your sample contracts
DATA_DIR = 'lexiquery_sample_contracts'


# ── STEP 1: Document Loading ─────────────────────────────────────────────
def load_single_document(file_path: str):
    """
    Loads a single PDF or text document.
    PyPDFLoader handles multi-page PDFs, returning one Document per page.
    This is crucial for legal docs that can span 100+ pages.
    """
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    # Tag every page with which contract it came from, so citations later
    # can say "Source: Employment_Agreement.pdf, Page 2" instead of just "Page 2"
    source_name = os.path.basename(file_path)
    for doc in documents:
        doc.metadata['source_file'] = source_name

    print(f'Loaded {len(documents)} pages/sections from {source_name}')
    return documents


def load_documents_from_folder(folder_path: str = DATA_DIR):
    """
    Loads every PDF (and .txt, if present) in the given folder.
    This lets LexiQuery index a whole contract library instead of one file.
    """
    file_paths = sorted(
        glob.glob(os.path.join(folder_path, '*.pdf'))
        + glob.glob(os.path.join(folder_path, '*.txt'))
    )
    if not file_paths:
        raise FileNotFoundError(
            f"No .pdf or .txt files found in '{folder_path}'. "
            f"Make sure your contracts are unzipped into that folder."
        )

    all_documents = []
    for file_path in file_paths:
        all_documents.extend(load_single_document(file_path))

    print(f'\nLoaded {len(all_documents)} total pages from {len(file_paths)} contract(s).')
    return all_documents


# ── STEP 2: Text Chunking ────────────────────────────────────────────────
def split_documents(documents):
    """
    RecursiveCharacterTextSplitter is preferred for legal text because it tries
    to split on natural boundaries (paragraphs, sentences) before characters.
    chunk_overlap=200 ensures legal clauses aren't split mid-sentence, losing context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # ~250 tokens — fits within context without overloading
        chunk_overlap=200,    # 200-char overlap prevents losing cross-chunk references
        separators=['\n\n', '\n', '. ', ' ', '']  # Try these delimiters in order
    )
    chunks = splitter.split_documents(documents)
    print(f'Created {len(chunks)} chunks from documents')
    return chunks


# ── STEP 3: Embedding Model ──────────────────────────────────────────────
def get_embeddings():
    """
    all-MiniLM-L6-v2 is the go-to free embedding model:
    - 384 dimensions (fast, memory-efficient)
    - Trained on 1B+ sentence pairs (great semantic understanding)
    - Fully local: no API calls, no cost, no rate limits
    """
    return HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU available
        encode_kwargs={'normalize_embeddings': True}  # Cosine similarity ready
    )


# ── STEP 4: Vector Store ─────────────────────────────────────────────────
def build_vector_store(chunks, embeddings, persist_path='legal_faiss_index'):
    """
    FAISS (Facebook AI Similarity Search) creates an in-memory index of vectors.
    save_local() persists it to disk — so you don't re-embed on every run.
    This is production behavior: embed once, query many times.
    """
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(persist_path)  # Save to disk for reuse
    print(f'Vector store saved to ./{persist_path}/')
    return vectorstore


def load_vector_store(embeddings, persist_path='legal_faiss_index'):
    """Load existing FAISS index instead of rebuilding — saves time and compute."""
    return FAISS.load_local(
        persist_path, embeddings,
        allow_dangerous_deserialization=True  # Required flag for FAISS in LangChain
    )


# ── STEP 5 & 6: Retrieval + Generation Chain (MODERNIZED) ────────────────
def build_qa_chain(vectorstore):
    """
    Builds the core RAG chain using modern LCEL (LangChain Expression Language).
    The legal prompt enforces citation behavior and prevents hallucination.
    """
    # 1. Define the System Prompt
    system_prompt = (
        "You are a legal document analysis assistant. Answer ONLY based on "
        "the provided context. If the information is not in the context, say "
        "'This information is not found in the provided document.'\n\n"
        "IMPORTANT RULES:\n"
        "- Cite specific sections or clauses when possible\n"
        "- Use exact legal language from the document\n"
        "- Never infer or assume information not stated\n\n"
        "Context (extracted from the legal document):\n"
        "{context}"
    )

    # 2. Combine into a ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Legal Question: {input}")
    ])

    # 3. Setup the Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model='gemini-3.5-flash',
        api_key=API_KEY,
        temperature=0.1,  # Low temp for factual legal answers
        max_tokens=1024
    )

    # 4. Create the document chain (Stuffs the context into the prompt)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # 5. Create the retriever
    retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 4}  # Retrieve top-4 most relevant chunks
    )

    # 6. Combine retriever and document chain into a final retrieval chain
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return qa_chain


# ── Main Interface (MODERNIZED) ───────────────────────────────────────────
def query_legal_doc(qa_chain, question: str):
    """
    Runs a single query through the RAG pipeline.
    Formats the response with the answer + source citations.
    """
    # Modern chains use the 'input' key by default
    result = qa_chain.invoke({'input': question})

    print('\n' + '=' * 60)
    print('LEGAL ANALYSIS RESULT')
    print('=' * 60)
    print(f'Question: {question}')
    
    # Modern chains return the generated text under the 'answer' key
    print(f'\nAnswer:\n{result["answer"]}')
    
    print('\n--- Source Passages Used ---')
    # Modern chains return the retrieved chunks under the 'context' key
    for i, doc in enumerate(result['context'], 1):
        # Show which contract + page each source chunk came from
        source_file = doc.metadata.get('source_file', 'Unknown document')
        page = doc.metadata.get('page', 'N/A')
        print(f'[Source {i}] {source_file}, Page {page}: {doc.page_content[:200]}...')
        
    return result


# ── Entry Point ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    embeddings = get_embeddings()

    # First run: load and index every contract in DATA_DIR
    if not os.path.exists('legal_faiss_index'):
        docs = load_documents_from_folder(DATA_DIR)
        chunks = split_documents(docs)
        vectorstore = build_vector_store(chunks, embeddings)
    else:
        # Subsequent runs: load existing index (much faster)
        vectorstore = load_vector_store(embeddings)

    qa = build_qa_chain(vectorstore)

    # Example legal queries
    questions = [
        'What are the termination conditions in this contract?',
        'What are the payment terms and deadlines?',
        'What liabilities does each party accept?',
        'What are the confidentiality obligations?',
        'What happens if a party breaches the agreement?',
    ]
    for q in questions:
        query_legal_doc(qa, q)