import logging
from fastapi import APIRouter, Depends, HTTPException, status

from src.Agent.AgentController import AgentController
from src.Authentications.Dependencies import (
    get_async_db_client,
    get_current_user_uid,
    get_sync_db_client,
)
from src.Routes.Schemas.ChatSchemas import ChatRequest, ChatResponse, ResponseSignal

# --- Logger Setup ---
# Use the same logger name as in main.py for consistent logging, or a new one.
logger = logging.getLogger(__name__)

# --- API Router Definition ---
# This router handles all endpoints related to the core chatbot functionality.
chat_router = APIRouter(prefix="/api/v1", tags=["Chatbot"])


@chat_router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(
    chat_request: ChatRequest,
    # --- Dependencies ---
    # FastAPI's dependency injection system provides required services for each request.
    sync_db=Depends(get_sync_db_client),
    async_db=Depends(get_async_db_client),
    user_id: str = Depends(get_current_user_uid),
):
    """
    The main endpoint for handling user chat messages.

    This endpoint receives a user's message, authenticates the user,
    and invokes the AI agent to generate a response. It is fully asynchronous.

    - On success, it returns a 200 OK status with the agent's text response.
    - On failure, it raises an appropriate HTTPException.
    """
    logger.info(f"Received chat request from user_id: {user_id}")
    
    try:
        # --- Agent Initialization ---
        # For each request, a new agent instance is created. This is a stateless
        # approach that ensures thread safety. The heavy components (like embeddings
        # and database clients) are shared, while user-specific memory is
        # loaded on-demand within the controller.
        agent_factory = AgentController(
            user_id=user_id,
            sync_db_client=sync_db,
            async_db_client=async_db,
        )
        agent_executor = agent_factory.get_agent_executor()

        # --- Agent Invocation ---
        # Asynchronously call the agent with the user's input. The `ainvoke`
        # method triggers the agent's reasoning loop.
        agent_output = await agent_executor.ainvoke({"input": chat_request.message})
        
        text_response = agent_output.get(
            "output", "I'm sorry, I couldn't generate a response."
        )

        # --- Successful Response ---
        # The response is structured according to the ChatResponse Pydantic model.
        return ChatResponse(signal=ResponseSignal.TEXT_RESPONSE, message=text_response)

    except Exception as e:
        # --- Critical Error Handling ---
        # This catches any unexpected errors during the agent's execution.
        # Logging the full error with a traceback is crucial for debugging.
        logger.critical(
            f"Unhandled exception while invoking agent for user_id: {user_id}. Error: {e}",
            exc_info=True, # This includes the full stack trace in the log
        )
        
        # Raise a generic 500 error to avoid exposing internal implementation
        # details to the client.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing your request.",
        )