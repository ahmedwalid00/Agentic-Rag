from fastapi import Depends, HTTPException , Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin import auth

# Create a security scheme instance. This tells FastAPI to look for a 'Bearer' token.
security = HTTPBearer()

async def get_current_user_uid(cred: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    A FastAPI dependency to verify a Firebase ID token and return the user's UID.
    
    This function will be run for every request that includes it in its dependencies.
    """
    if not cred:
        raise HTTPException(
            status_code=401,
            detail="Bearer token missing or invalid."
        )
    
    try:
        # Verify the token against the Firebase Admin SDK
        decoded_token = auth.verify_id_token(cred.credentials)
        # Extract and return the user's unique ID (UID)
        return decoded_token['uid']
    except Exception as e:
        # This will catch expired tokens, malformed tokens, etc.
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication credentials: {e}"
        )
    

def get_sync_db_client(request: Request):
    """Dependency to get the SYNCHRONOUS Firestore client."""
    if not hasattr(request.app.state, 'db_client_sync') or not request.app.state.db_client_sync:
        raise HTTPException(status_code=503, detail="Synchronous DB service is not available.")
    return request.app.state.db_client_sync

def get_async_db_client(request: Request):
    """Dependency to get the ASYNCHRONOUS Firestore client."""
    if not hasattr(request.app.state, 'db_client_async') or not request.app.state.db_client_async:
        raise HTTPException(status_code=503, detail="Asynchronous DB service is not available.")
    return request.app.state.db_client_async