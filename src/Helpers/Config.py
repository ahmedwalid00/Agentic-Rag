from pydantic_settings import BaseSettings
from typing import List 

class Settings(BaseSettings):
    APP_NAME : str
    GENERATION_BACKEND : str
    EMBEDDING_BACKEND : str
    OPENAI_API_KEY : str
    OPENAI_API_URL : str = None
    GENERATION_MODEL_ID_LITERAL : List[str] = None
    GENERATION_MODEL_ID : str
    EMBEDDING_MODEL_ID : str
    EMBEDDING_MODEL_SIZE : int
    INPUT_DAFAULT_MAX_CHARACTERS : int
    GENERATION_DAFAULT_MAX_TOKENS : int
    GENERATION_DAFAULT_TEMPERATURE : float
    CHROMA_DB_DIR : str
    PDF_PATH_RAG : str
    FIREBASE_CREDENTIAL_PATH : str

    
    
    class Config:
        env_file = ".env"


def get_settings():
    return Settings()