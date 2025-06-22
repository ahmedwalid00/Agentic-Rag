import os
import logging
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.Helpers.Config import get_settings

# --- Logger Setup ---
logger = logging.getLogger(__name__)


class PdfRetrieverTool:
    """
    A class responsible for creating a LangChain tool that can retrieve
    information from a pre-processed company policy document (PDF).
    
    This tool connects to a Chroma vector store, which should have been
    created by a separate data ingestion script.
    """

    def __init__(self, pdf_path: str):  # The pdf_path is kept for potential future use or validation
        self.app_settings = get_settings()
        
        # --- Pre-computation Check ---
        # It's good practice to fail early if a required resource is missing.
        # This check ensures the data ingestion script has been run.
        if not os.path.isdir(self.app_settings.CHROMA_DB_DIR):
            logger.error(f"Chroma DB directory not found at: {self.app_settings.CHROMA_DB_DIR}")
            raise FileNotFoundError(
                f"Chroma DB directory not found at '{self.app_settings.CHROMA_DB_DIR}'. "
                "Please run the data ingestion script first to create the vector store."
            )
        logger.info("PdfRetrieverTool initialized, Chroma DB directory found.")

    def get_retriever_tool(self):
        """
        Builds and returns the company policy retriever tool.

        This process involves:
        1. Setting up the same embedding model used during data ingestion.
        2. Loading the persistent vector store from disk.
        3. Creating a retriever from the vector store.
        4. Wrapping the retriever in a LangChain Tool with a descriptive name.
        """
        logger.info("Creating company policy retriever tool...")

        # Step 1: Initialize the embedding model. This must match the model
        # used to create the embeddings in the first place.
        embedding_llm = OpenAIEmbeddings(
            api_key=self.app_settings.OPENAI_API_KEY,
            model=self.app_settings.EMBEDDING_MODEL_ID,
        )

        # Step 2: Load the persisted vector store from the specified directory.
        # This operation is fast as it reads pre-computed data from disk.
        vector_store = Chroma(
            persist_directory=self.app_settings.CHROMA_DB_DIR,
            embedding_function=embedding_llm,
        )
        
        # Step 3: Create a retriever from the vector store.
        # `k=3` means it will fetch the top 3 most relevant document chunks.
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Step 4: Use a LangChain helper to create a well-defined tool.
        # The description is critical, as it's what the agent's LLM reads
        # to decide when to use this tool.
        retriever_tool = create_retriever_tool(
            retriever,
            "company_policy_retriever",
            (
                "Use this tool to answer any questions about official company policies, "
                "rules, procedures, and benefits. It is the definitive source for topics "
                "like paid time off (PTO), sick leave, work-from-home rules, "
                "code of conduct, dress code, and HR regulations."
            ),
        )

        logger.info("'company_policy_retriever' tool created successfully.")
        return retriever_tool