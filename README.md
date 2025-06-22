# REBOTA: The Intelligent Company Chatbot

REBOTA is a sophisticated, AI-powered chatbot designed to streamline interactions within a company's ecosystem. Built with a modern Python backend using FastAPI and LangChain, and powered by a Firebase/Firestore database, REBOTA serves as a centralized, conversational interface for employees, HR personnel, and even new applicants.

The core mission of this project is to provide instant, accurate, and personalized information by leveraging the power of Large Language Models (LLMs) combined with company-specific data.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-339943?style=for-the-badge)](https://www.langchain.com/)
[![Firebase](https://img.shields.io/badge/Firebase-12.0+-FFCA28?style=for-the-badge&logo=firebase)](https://firebase.google.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT_Models-4A90E2?style=for-the-badge&logo=openai)](https://openai.com/)

---

## ✨ Key Features

-   **Personalized Responses:** The chatbot identifies the user and tailors its answers. An employee asking "What is my salary?" receives their specific salary, not a generic answer.
-   **Role-Based Access Control (RBAC):** REBOTA understands user roles. An HR user can ask about applicant data, while a regular employee cannot.
-   **Company Policy Knowledge:** Using Retrieval-Augmented Generation (RAG), the bot can answer detailed questions about company policies by referencing internal PDF documents.
-   **Conversational Memory:** The chatbot remembers the context of the conversation, allowing for natural follow-up questions and a fluid user experience. Chat history is securely stored per-user in Firestore.
-   **Asynchronous & Scalable:** Built on FastAPI's async framework, the application is non-blocking and highly performant, capable of handling numerous concurrent users.
-   **Secure & Robust:** Integrates with Firebase Authentication for secure user identification and features structured logging and tracing for production-grade observability.

---

## 🏛️ Project Architecture

The application is built on a modular and decoupled architecture, primarily consisting of:

1.  **FastAPI Server (`main.py`):** The main entry point of the application. It handles HTTP requests, manages the application lifecycle (startup/shutdown), and routes traffic to the appropriate handlers.
2.  **LangChain Agent (`AgentController.py`):** The "brain" of the chatbot. It uses an LLM to reason about user input and decides which tool to use to find the answer.
3.  **Tools (`Tools/`):** These are specialized functions the agent can use to interact with the outside world.
    -   `FireBaseTool.py`: Connects to the Firestore database to fetch or query user-specific data (e.g., salary, leave days, applicant info). It contains an internal router to map natural language to specific database queries.
    -   `PdfRetrieverTool.py`: Connects to a ChromaDB vector store to find relevant information from company policy documents.
4.  **Firebase (`Firestore` & `Authentication`):** The backend-as-a-service.
    -   **Authentication:** Manages user identity and provides JWT tokens for securing the API.
    -   **Cloud Firestore:** A NoSQL database used to store `users` data (with roles) and persistent `chat_histories` for each user.
5.  **Data Ingestion (`ingest.py`):** A one-time script used to process policy documents, create embeddings, and build the vector store for the RAG tool.


---
