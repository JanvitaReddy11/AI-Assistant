# AI Assistant

## Overview  
This personal assistant streamlines daily tasks by automating processes like email management, meeting scheduling, web searches, and PDF-based information retrieval. It integrates multiple tools and local language models to ensure data privacy and enhance productivity.

## Key Features  

### **Email Assistant**  
- Automates email tasks (write, send, search) using Gmail tools via LangChain.  
- Securely handles sensitive data with the locally downloaded **LLaMA 3 model** (4GB) using **Groq** and **Ollama**.  
- Includes retry mechanisms to handle errors and conditional checks for seamless email operations.  

### **Calendar Assistant**  
- Schedules events using Google Calendar API and extracts natural language inputs (subject, start/end time).  
- Converts times to **CDT** for consistency and checks for conflicts.  
- Sends event links or suggests alternative times if conflicts arise.  

### **Search Assistant**  
- Utilizes the **Tavily API** for unrestricted real-time web searches.  
- Provides relevant results or retries when information is unavailable.  

### **PDF Reader**  
- Implements **Retrieval-Augmented Generation (RAG)** for multi-PDF analysis.  
- Converts PDFs into embeddings using the **Gemini model** and stores them in a **Faiss vector store** for efficient retrieval.  

### **Master Agent**  
- Integrates all functionalities via a custom React-based agent using structured reasoning workflows.  
- Tools include reading PDFs, sending emails, scheduling meetings, and performing web searches.  
- Allows iterative refinement (up to 10 iterations) to optimize user responses.

  ## Setup Instructions
1] Run “ pip install -r requirements.txt” command
2] Create a .env file where the api keys of google, groq, tavily, langchain should be given.
3] Run the command “streamlit run app.py”
