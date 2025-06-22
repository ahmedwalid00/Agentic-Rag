import os
import logging
from fastapi import FastAPI
import firebase_admin
from firebase_admin import credentials
from google.cloud.firestore import Client as SyncFirestoreClient
from google.cloud.firestore_v1.async_client import AsyncClient as AsyncFirestoreClient

from src.Helpers.Config import get_settings
from src.Routes.ChatHandlerRoute import chat_router

# --- Basic Logging Configuration ---
# This sets up a simple logger that prints to the console.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Application Settings and Global Setup ---
# Load settings from the .env file. This is done once when the module is loaded.
settings = get_settings()

# Set the environment variable for Google Cloud libraries BEFORE any services are initialized.
# This is the recommended way to provide credentials programmatically.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.FIREBASE_CREDENTIAL_PATH


# --- Application Lifecycle Events (Startup & Shutdown) ---

async def startup_event():
    """
    Initializes critical services when the FastAPI application starts.
    This function sets up the Firebase Admin SDK and both synchronous and
    asynchronous Firestore database clients.
    """
    try:
        logger.info("Application startup: Initializing external services...")
        
        # Initialize the Firebase Admin SDK using the service account credentials.
        # This is essential for services like Firebase Authentication.
        cred = credentials.Certificate(settings.FIREBASE_CREDENTIAL_PATH)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully.")
        
        project_id = cred.project_id
        
        # Create and store both database clients on the application's state.
        # This makes them accessible throughout the application via the `request` object.
        app.state.db_client_sync = SyncFirestoreClient(project=project_id)
        app.state.db_client_async = AsyncFirestoreClient(project=project_id)
        logger.info("Firestore clients (sync & async) created and ready.")

    except Exception as e:
        # A failure here is critical and likely means the app cannot function.
        logger.critical(f"FATAL ERROR during application startup: {e}", exc_info=True)
        app.state.db_client_sync = None
        app.state.db_client_async = None

def shutdown_event():
    """
    Gracefully cleans up resources when the FastAPI application shuts down.
    This ensures all connections are closed properly.
    """
    try:
        logger.info("Application shutdown: Cleaning up resources...")
        
        # Close database clients if they were successfully created.
        if hasattr(app.state, 'db_client_async') and app.state.db_client_async:
            app.state.db_client_async.close()
            logger.info("Async Firestore client closed.")
            
        if hasattr(app.state, 'db_client_sync') and app.state.db_client_sync:
            app.state.db_client_sync.close()
            logger.info("Sync Firestore client closed.")
            
        # Delete the Firebase app instance to release its resources.
        if firebase_admin._apps:
            firebase_admin.delete_app(firebase_admin.get_app())
            logger.info("Firebase Admin SDK app deleted.")

    except Exception as e:
        logger.error(f"Error during graceful shutdown: {e}", exc_info=True)


# --- FastAPI Application Instance ---
# This is the main entry point for the web server (e.g., Uvicorn).
app = FastAPI(
    title=settings.APP_NAME,
    description="An intelligent chatbot for company interactions, powered by AI.",
    version="1.0.0",
    on_startup=[startup_event],
    on_shutdown=[shutdown_event]
)

# --- API Routers ---
# Include the chat endpoints from the dedicated router file.
app.include_router(chat_router)

# --- Root Endpoint ---
# A simple health check endpoint to confirm the API is running.
@app.get("/", tags=["Health Check"])
def read_root():
    """Provides a simple health check response."""
    return {"status": "ok", "app_name": settings.APP_NAME}