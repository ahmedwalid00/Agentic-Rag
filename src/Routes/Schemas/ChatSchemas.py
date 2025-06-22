from pydantic import BaseModel , Field
from enum import Enum

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=1400)

class ResponseSignal(str, Enum):
    TEXT_RESPONSE = "TEXT_RESPONSE"
    ERROR = "ERROR" 

class ChatResponse(BaseModel):
    signal: ResponseSignal
    message: str