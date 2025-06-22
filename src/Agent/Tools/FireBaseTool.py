import logging
import json
from datetime import datetime

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.Agent.Prompts.AgentPrompt import router_llm_prompt
from src.Helpers.Config import get_settings

# --- Logger Setup ---
logger = logging.getLogger(__name__)


# ==============================================================================
# === INTERNAL HELPER FUNCTIONS (Single-responsibility data handlers) ==========
# ==============================================================================

async def _get_user_document(db_client, user_id: str) -> dict | None:
    """Fetches a user's document from Firestore by their unique ID."""
    if not db_client:
        return None
    try:
        user_ref = db_client.collection("users").document(user_id)
        user_doc = await user_ref.get()
        return user_doc.to_dict() if user_doc.exists else None
    except Exception as e:
        logger.error(f"Firestore error fetching user document for {user_id}: {e}")
        return None

def _get_personal_data(user_doc: dict, field: str) -> str:
    """
    Handles data retrieval for ANY user role by intelligently checking for the
    field's existence and using canonical keys provided by the router LLM.
    """
    if not field:
        return "Please specify what personal information you'd like to know."

    name = user_doc.get("name", "The user")
    field_lower = field.lower()

    # Handle the 'all' case to provide a full, structured summary.
    if field_lower == "all":
        details = [f"Here is a summary for {name}:"]
        ordered_keys = [
            'name', 'email', 'role', 'position', 'department',
            'baseSalary', 'bonus', 'annualLeaveDays', 'sickLeaveDays'
        ]
        for key in ordered_keys:
            if key in user_doc:
                value = user_doc[key]
                display_key = key.replace("joinDate", "Start Date").title()
                if key == "baseSalary":
                    details.append(f"- Salary: ${value:,.2f}")
                elif key == "bonus":
                    details.append(f"- Bonus: ${value:,.2f}")
                else:
                    details.append(f"- {display_key}: {value}")
        
        # Add document statuses to the summary if they exist
        if uploaded_docs := user_doc.get("uploadedDocuments"):
            submitted = ", ".join([doc for doc, status in uploaded_docs.items() if status])
            if submitted:
                details.append(f"- Documents Submitted: {submitted}")
        if resubmit_docs := user_doc.get("resubmissionRequested"):
            resubmit = ", ".join([doc for doc, status in resubmit_docs.items() if status])
            if resubmit:
                details.append(f"- Resubmission Required For: {resubmit}")
        
        return "\n".join(details)

    # Handle specific field requests using canonical names.
    # First, try a direct, case-insensitive match.
    for doc_key, doc_value in user_doc.items():
        if doc_key.lower() == field_lower:
            return f"The value for '{doc_key}' for {name} is: {doc_value}"
            
    # Then, handle complex or specially-named fields.
    if field_lower == "salary":
        if "baseSalary" in user_doc:
            base, bonus = user_doc.get("baseSalary", 0), user_doc.get("bonus", 0)
            return f"{name.capitalize()}'s total compensation is ${base + bonus:,.2f}."
        else:
            return f"Salary information is not available for {name}."

    return f"I'm sorry, I couldn't find information about '{field}' in {name}'s records."


def _check_document_status(user_doc: dict, document_name: str) -> str:
    """Handles document status checks for any user with robust, flexible matching."""
    if not document_name:
        return "Please specify which document you'd like to check."
    
    name = user_doc.get("name", "the user")
    
    def find_robust_key(data_map, target_key_from_llm):
        clean_target = target_key_from_llm.lower().replace(" ", "")
        for key_from_db in data_map:
            if clean_target in key_from_db.lower().replace(" ", ""):
                return key_from_db
        return None

    if resub_map := user_doc.get("resubmissionRequested"):
        if resub_key := find_robust_key(resub_map, document_name):
            if resub_map.get(resub_key):
                return f"Action required for {name}: They need to resubmit their '{resub_key}'."

    if up_map := user_doc.get("uploadedDocuments"):
        if up_key := find_robust_key(up_map, document_name):
            if up_map.get(up_key):
                return f"Yes, the '{up_key}' for {name} has been successfully submitted and approved."
            
    return f"There is no information regarding the document '{document_name}' for {name}."


async def _get_new_applicant_count(db_client) -> str:
    """Counts users with the 'new' role."""
    if not db_client: return "Database not available."
    query = db_client.collection("users").where("role", "==", "new")
    count = sum([1 async for _ in query.stream()])
    if count == 0: return "There are currently no new applicants."
    return f"There is currently {count} new applicant." if count == 1 else f"There are currently {count} new applicants."


async def _get_user_by_identifier(db_client, identifier: str):
    """Finds a user by their email first, then falls back to a case-sensitive name."""
    if not identifier: return None
    # Prioritize unique email search
    query_by_email = db_client.collection("users").where("email", "==", identifier).limit(1)
    async for doc in query_by_email.stream(): return doc.to_dict()
    # Fallback to name search (case-sensitive, matching Firestore's capabilities)
    query_by_name = db_client.collection("users").where("name", "==", identifier.title()).limit(1)
    async for doc in query_by_name.stream(): return doc.to_dict()
    return None


def handle_sensitive_or_broad_data_request(query: str) -> str:
    """A guardrail function providing a standard refusal for bulk data requests."""
    return (
        "I'm sorry, I cannot fulfill that request. This is likely because it's a request "
        "for bulk data or sensitive information for which you do not have permission."
    )


# ==============================================================================
# === MAIN DISPATCHER FUNCTION (The single entry point for the Tool) ===========
# ==============================================================================

async def get_user_specific_data_tool(query: str, user_id: str, db_client) -> str:
    """
    This function acts as a smart dispatcher. It uses an internal LLM router to
    analyze the user's query and role, then calls the appropriate helper function
    to fulfill the request securely.
    """
    app_settings = get_settings()
    router_llm = ChatOpenAI(
        api_key=app_settings.OPENAI_API_KEY, model="gpt-4o-mini", temperature=0
    )
    
    if not db_client:
        return "I am sorry, but I cannot connect to our database at the moment."

    user_doc = await _get_user_document(db_client, user_id)
    if not user_doc:
        return "I'm sorry, I couldn't find your user profile."

    user_role = user_doc.get("role", "unknown")
    
    # Dynamically define available actions for the router based on the user's role.
    available_actions = {}
    if user_role in ["employee", "new"]:
        available_actions["get_my_personal_info"] = "Get the logged-in user's own core data."
        available_actions["check_my_document_status"] = "Check the logged-in user's own document submission status."
    elif user_role == "hr":
        available_actions = {
            "get_my_personal_info": "Get the HR user's own core data.",
            "check_my_document_status": "Check the HR user's own document status.",
            "get_information_about_anyone": "Get info (data or document status) about any other user.",
            "get_applicant_count": "Get the total count of new job applicants.",
        }

    if not available_actions:
        return f"I'm sorry, your user role ('{user_role}') is not configured for any actions."

    # --- Internal LLM Routing ---
    prompt = ChatPromptTemplate.from_template(router_llm_prompt)
    chain = prompt | router_llm | JsonOutputParser()

    try:
        response = await chain.ainvoke(
            {"query": query, "user_role": user_role, "actions": json.dumps(available_actions, indent=2)}
        )
        logger.info(f"Internal router response: {response}")
        action_name = response.get("action_name")
        parameters = response.get("parameters", {})

        # --- Dispatch Logic ---
        if action_name == "get_my_personal_info":
            return _get_personal_data(user_doc, parameters.get("field"))
            
        elif action_name == "check_my_document_status":
            return _check_document_status(user_doc, parameters.get("document_name"))
            
        elif action_name == "get_applicant_count" and user_role == "hr":
            return await _get_new_applicant_count(db_client)
            
        elif action_name == "get_information_about_anyone" and user_role == "hr":
            target_id = parameters.get("target_identifier")
            req_details = parameters.get("request_details", "all") # Default to 'all'
            
            if not target_id:
                return "Please specify the person you are asking about."
                
            target_doc = await _get_user_by_identifier(db_client, target_id)
            if not target_doc:
                return f"I could not find a user matching '{target_id}'."
            
            doc_keywords = ["document", "certificate", "id", "bank", "submission", "status", "upload"]
            if any(keyword in req_details.lower() for keyword in doc_keywords):
                return _check_document_status(target_doc, req_details)
            else:
                return _get_personal_data(target_doc, req_details)
                
        else:
            return "I'm sorry, I cannot perform that action due to your permissions or an unrecognized request."
            
    except Exception as e:
        logger.error(f"Error in tool's internal router: {e}", exc_info=True)
        return "I had trouble processing your request. Please try rephrasing."


