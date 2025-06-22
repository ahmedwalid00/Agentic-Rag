import asyncio
import logging
from functools import partial

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_community.chat_message_histories import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI

from src.Agent.Prompts.AgentPrompt import agent_prompt
from src.Agent.Tools.FireBaseTool import get_user_specific_data_tool
from src.Agent.Tools.PdfRetrieverTool import PdfRetrieverTool
from src.Helpers.Config import get_settings

# --- Logger Setup ---
logger = logging.getLogger(__name__)


# --- Local Helper Functions ---

def handle_sensitive_or_broad_data_request(query: str) -> str:
    """
    A simple, synchronous "guardrail" tool that provides a standard refusal
    for overly broad or sensitive data requests.
    """
    return (
        "I'm sorry, I cannot fulfill that request. "
        "This is likely because it's a request for bulk data or sensitive "
        "information for which you do not have permission."
    )


class AgentController:
    """
    The main controller responsible for initializing and assembling the AI agent.
    
    This class brings together all the components: tools, memory, prompts, and the LLM
    to create a runnable AgentExecutor.
    """

    def __init__(self, user_id: str, sync_db_client, async_db_client):
        self.app_settings = get_settings()
        self.user_id = user_id
        self.async_db = async_db_client
        self.sync_db = sync_db_client
        logger.info(f"AgentController initialized for user_id: {self.user_id}")

    def get_agent_memory(self) -> ConversationBufferWindowMemory:
        """
        Configures the agent's memory using Firestore as a persistent backend.
        Each user's conversation history is stored in a separate document,
        scoped by their unique user_id.
        """
        # Note: FirestoreChatMessageHistory is a synchronous class, so it requires
        # the synchronous Firestore client.
        message_history = FirestoreChatMessageHistory(
            collection_name="chat_histories",
            session_id=self.user_id,
            firestore_client=self.sync_db,
            user_id=self.user_id,
        )
        return ConversationBufferWindowMemory(
            k=5,  # Remembers the last 5 conversation turns
            memory_key="chat_history",
            return_messages=True,
            chat_memory=message_history,
        )

    def get_agent_prompt(self) -> ChatPromptTemplate:
        """
        Creates the main system prompt for the agent from a template.
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", agent_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

    def get_agent_executor(self) -> AgentExecutor:
        """
        The core factory method that assembles and returns the complete agent executor.
        """
        logger.info("Assembling agent executor...")

        # --- Tool Assembly ---

        # Tool 1: PDF Retriever for company policies
        pdf_retriever_tool_ins = PdfRetrieverTool(
            pdf_path=self.app_settings.PDF_PATH_RAG
        )
        policy_retriever_tool = pdf_retriever_tool_ins.get_retriever_tool()

        # Tool 2: User-Specific Data Tool (Advanced Setup)
        # We use `functools.partial` to "bake in" contextual arguments (user_id, db_client)
        # that the LLM should not have to reason about.
        firestore_coroutine_with_context = partial(
            get_user_specific_data_tool,
            user_id=self.user_id,
            db_client=self.async_db,
        )
        # Create a synchronous wrapper, required by the Tool constructor.
        def sync_wrapper(query: str):
            return asyncio.run(firestore_coroutine_with_context(query))

        user_data_firestore_tool = Tool(
            name="get_user_specific_data",
            description=(
                "This is the primary tool for retrieving all specific information "
                "about people. It has a built-in permission system.\n"
                "You MUST use this tool for any of these tasks:\n"
                "1. Fetching data about a SPECIFIC person (themselves or others).\n"
                "2. Getting the NUMERICAL COUNT of new applicants."
            ),
            func=sync_wrapper,
            coroutine=firestore_coroutine_with_context,
        )

        # Tool 3: Guardrail for broad data requests
        sensitive_data_tool = Tool(
            name="handle_sensitive_or_broad_data_request",
            description=(
                "This is a security guardrail tool. Use it ONLY for requests for a "
                "list of multiple people or a bulk data dump that is NOT a count. "
                "Examples: 'list all employees', 'give me all CVs'."
            ),
            func=handle_sensitive_or_broad_data_request,
        )

        tools = [policy_retriever_tool, user_data_firestore_tool, sensitive_data_tool]
        logger.info(f"Agent tools configured: {[tool.name for tool in tools]}")

        # --- LLM and Agent Chain Construction ---
        
        memory = self.get_agent_memory()
        prompt = self.get_agent_prompt()
        llm = ChatOpenAI(
            api_key=self.app_settings.OPENAI_API_KEY,
            model=self.app_settings.GENERATION_MODEL_ID,
            temperature=self.app_settings.GENERATION_DAFAULT_TEMPERATURE,
        )

        # This is the main LangChain Expression Language (LCEL) chain.
        # It defines the data flow from input to the final parsed output.
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | llm.bind_tools(tools)
            | OpenAIToolsAgentOutputParser()
        )

        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True, # Set to False in production for cleaner logs
            handle_parsing_errors=True, # Provides resilience against LLM formatting errors
        )