"""
MedLit Search — Medical Literature Assistant
HyDE RAG | Domain: Healthcare/Research | API: Groq (Llama 3)
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import PromptTemplate  # Modernized import
from langchain_core.output_parsers import StrOutputParser
import numpy as np

# Load API keys from .env file
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')  

# Fallback block to guard against empty or missing API keys
if not GROQ_API_KEY:
    raise ValueError("CRITICAL ERROR: GROQ_API_KEY is not set or could not be read from your .env file.")

# ── CONFIGURATION ─────────────────────────────────────────────────────────
EMBED_MODEL = 'sentence-transformers/all-mpnet-base-v2'  # Better than MiniLM for medical
GROQ_MODEL = 'llama-3.3-70b-versatile'  # Groq's best free model
CHUNK_SIZE = 800   # Smaller chunks for denser medical passages
TOP_K = 5          # Retrieve top-5 passages

# Folder containing your sample abstracts
PAPERS_DIR = 'medlit_sample_abstracts'


# ── STEP 1: Build Medical Knowledge Base ──────────────────────────────────
def build_medical_kb(papers_dir: str = PAPERS_DIR):
    """
    Loads all .txt medical paper abstracts from a directory.
    In a real system, you'd use a PubMed API loader or arXiv loader.
    For the portfolio project: save a few paper abstracts as .txt files.
    """
    if not os.path.isdir(papers_dir):
        raise FileNotFoundError(
            f"Papers directory '{papers_dir}' not found. "
            f"Unzip your medlit_sample_abstracts.zip into that folder first."
        )

    loader = DirectoryLoader(
        papers_dir,
        glob='**/*.txt',
        loader_cls=TextLoader
    )
    docs = loader.load()
    if not docs:
        raise FileNotFoundError(
            f"No .txt files found in '{papers_dir}'. "
            f"Make sure your abstract files were unzipped (not left inside a nested folder)."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local('medical_hyde_index')
    print(f'Built medical KB with {len(chunks)} chunks from {len(docs)} abstract(s)')
    return vectorstore, embeddings


# ── CORE HyDE: Generate Hypothetical Document ─────────────────────────────
def generate_hypothetical_document(question: str, llm) -> str:
    """
    THE KEY INNOVATION: Ask the LLM to write a passage that WOULD answer
    the question as if it were excerpted from a real medical research paper.
    This produced text exists in the same vector space as real paper passages,
    so retrieving similar vectors finds real, relevant passages.
    """
    hyde_prompt = PromptTemplate(
        input_variables=['question'],
        template="""You are a medical research expert. Write a detailed research paper
passage that directly answers the following clinical question. Write it in the style
of an academic medical publication — use clinical terminology, cite mechanisms,
and include specific details like pathways, biomarkers, or clinical findings.

Do NOT say 'I think' or 'According to'. Write as if this is an actual research passage.
Length: 150-200 words.

Clinical Question: {question}

Research Passage:"""
    )
    chain = hyde_prompt | llm | StrOutputParser()
    hypothetical_doc = chain.invoke({'question': question})
    print(f'\n[HyDE] Generated hypothetical passage ({len(hypothetical_doc)} chars)')
    print(f'Preview: {hypothetical_doc[:150]}...')
    return hypothetical_doc


# ── HyDE Retrieval: Embed Hypothetical, Retrieve Real ─────────────────────
def hyde_retrieve(question: str, vectorstore, embeddings, llm, k=TOP_K):
    """
    Full HyDE retrieval pipeline:
    1. Generate hypothetical document
    2. Embed the hypothetical document
    3. Retrieve real documents similar to the hypothetical
    Returns both the retrieved docs AND the hypothetical for transparency.
    """
    # Step A: Generate a hypothetical answer passage
    hypothetical_doc = generate_hypothetical_document(question, llm)

    # Step B: Embed the hypothetical passage (not the original question!)
    # embed_query() returns a single vector for one text
    hyp_embedding = embeddings.embed_query(hypothetical_doc)

    # Step C: Convert to numpy array format FAISS expects (kept for reference;
    # similarity_search_by_vector accepts the plain list too)
    hyp_vector = np.array([hyp_embedding], dtype=np.float32)

    # Step D: Search FAISS with the hypothetical embedding
    # similarity_search_by_vector() takes an embedding, not text
    retrieved_docs = vectorstore.similarity_search_by_vector(
        hyp_embedding, k=k
    )
    return retrieved_docs, hypothetical_doc


# ── Final Answer Generation ────────────────────────────────────────────────
def generate_final_answer(question: str, retrieved_docs, llm) -> str:
    """
    Standard generation step: use REAL retrieved passages to produce the answer.
    Note: we answer from REAL docs (not the hypothetical) for accuracy.
    The hypothetical was only a search key, not a source of truth.
    """
    context = '\n\n---\n\n'.join([doc.page_content for doc in retrieved_docs])

    answer_prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template="""You are a medical literature expert. Based ONLY on the research
passages provided, answer the clinical question. Include specific mechanisms,
studies, and clinical implications mentioned in the passages.

Research Passages:
{context}

Clinical Question: {question}

Evidence-Based Answer:"""
    )
    chain = answer_prompt | llm | StrOutputParser()
    return chain.invoke({'context': context, 'question': question})


# ── Main HyDE RAG Pipeline ────────────────────────────────────────────────
def hyde_rag_query(question: str, vectorstore, embeddings, llm):
    """Complete HyDE RAG pipeline with full transparency output."""
    print(f'\n{"=" * 65}')
    print('HYDE RAG MEDICAL QUERY')
    print(f'{"=" * 65}')
    print(f'Original Question: {question}')

    # HyDE retrieval (the key differentiator)
    retrieved_docs, hypothetical = hyde_retrieve(
        question, vectorstore, embeddings, llm
    )

    # Final answer from real retrieved passages
    answer = generate_final_answer(question, retrieved_docs, llm)

    print(f'\nFINAL ANSWER:\n{answer}')
    print('\nSOURCES USED:')
    for i, doc in enumerate(retrieved_docs, 1):
        src = doc.metadata.get('source', 'Unknown Paper')
        print(f'  [{i}] {src}: {doc.page_content[:100]}...')
    return answer, hypothetical, retrieved_docs


# ── COMPARISON: Standard vs HyDE ──────────────────────────────────────────
def compare_retrieval(question: str, vectorstore, embeddings, llm):
    """
    Demonstrates the difference between standard and HyDE retrieval.
    Run this to show interviewers/viewers the concrete improvement.
    """
    print('\n--- STANDARD RETRIEVAL (embedding the question) ---')
    standard_docs = vectorstore.similarity_search(question, k=3)
    for i, doc in enumerate(standard_docs, 1):
        print(f'[S{i}] {doc.page_content[:120]}...')

    print('\n--- HyDE RETRIEVAL (embedding a hypothetical answer) ---')
    hyde_docs, hyp = hyde_retrieve(question, vectorstore, embeddings, llm, k=3)
    for i, doc in enumerate(hyde_docs, 1):
        print(f'[H{i}] {doc.page_content[:120]}...')


if __name__ == '__main__':
    # Initialize models
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.2)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Build or load KB
    if not os.path.exists('medical_hyde_index'):
        vectorstore, embeddings = build_medical_kb(PAPERS_DIR)
    else:
        vectorstore = FAISS.load_local(
            'medical_hyde_index', embeddings,
            allow_dangerous_deserialization=True
        )

    # Example clinical queries — these match the abstracts in data/medlit
    questions = [
        'What cellular mechanisms cause type 2 diabetes progression?',
        'How does metformin affect insulin sensitivity at the molecular level?',
        'What biomarkers predict cardiovascular risk in diabetic patients?',
    ]
    for q in questions:
        hyde_rag_query(q, vectorstore, embeddings, llm)
        compare_retrieval(q, vectorstore, embeddings, llm)