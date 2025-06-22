import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter # A more robust splitter
from src.Helpers.Config import get_settings , Settings
from typing import List
from dataclasses import dataclass

print(os.path.isfile(r"D:\Rebota-ChatBot\src\Assets\files\hrppm_company_policy.pdf"))
@dataclass
class Document:
    page_content: str
    metadata: dict


settings = get_settings()


def main():
    # chunks = process_file_content()
    # print(f"Loaded and split into {len(chunks)} chunks.\n")

    # # Print first 10 chunks only
    # for i, chunk in enumerate(chunks[:10]):
    #     print(f"\n--- Chunk {i + 1} ---")
    #     print(chunk.page_content)
    # 2. Split the document into chunks for better retrieval
    chunks = process_file_content()
    print(f"Split the document into {len(chunks)} chunks.")

    # 3. Create embeddings
    embedding_llm = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        model=settings.EMBEDDING_MODEL_ID,
        # dimensions=settings.EMBEDDING_MODEL_SIZE # Some models don't need this, check API
    )

    docs_ids = list( range( len(chunks) ) )
    docs_ids = [ str(d) for d in docs_ids ]

    # 4. Create and persist the vector store
    # This will create files in the 'chroma_db' directory.
    print(f"Creating and persisting vector store at: {settings.CHROMA_DB_DIR}")
    Chroma.from_documents(
        chunks,
        embedding_llm,
        persist_directory=settings.CHROMA_DB_DIR,
        ids=docs_ids
    )

    print("\n--- Ingestion Complete! ---")
    print(f"Vector store is ready for use in '{settings.CHROMA_DB_DIR}'.")

def process_simpler_splitter(texts: List[str], metadatas: List[dict], chunk_size: int = 200, splitter_tag: str="\n"):
        
        full_text = " ".join(texts)

        # split by splitter_tag
        lines = [ doc.strip() for doc in full_text.split(splitter_tag) if len(doc.strip()) > 1 ]

        chunks = []
        current_chunk = ""

        for line in lines:
            current_chunk += line + splitter_tag
            if len(current_chunk) >= chunk_size:
                chunks.append(Document(
                    page_content=current_chunk.strip(),
                    metadata={}
                ))

                current_chunk = ""

        if len(current_chunk) >= 0:
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata={}
            ))

        return chunks
    
def process_file_content():
        
    loader = PyMuPDFLoader(file_path=settings.PDF_PATH_RAG)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from the PDF.")

        
    file_content_text = [ record.page_content for record in documents ]
    file_content_metadata = [ record.metadata for record in documents ]

    chunks = process_simpler_splitter(
            texts=file_content_text , 
            metadatas= file_content_metadata 
        )

    return chunks


if __name__ == "__main__":
    main()