from dotenv import load_dotenv
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langgraph.prebuilt import create_react_agent
from groq import Groq
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from PyPDF2 import PdfReader
import json
import time
import datetime
import os.path
import re
import pytz

load_dotenv()
os.getenv('LANGCHAIN_API_KEY')
api_key = os.getenv('GROQ_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

system_prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop, you output an Answer.
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

1. read_pdfs:
   e.g. read_pdf: 
   eg : summarise about the topic mentioned in the pdfs
   Reads multiple specified PDFs and provides relevant answers based on user queries.

2. search_internet:
   e.g. search_internet: "What are the latest trends in artificial intelligence?"
   Searches the internet for the specified query and returns the most relevant information.

3. send_gmail:
   e.g. send_gmail: "Send an email to Annie (annie@gmail.com) asking about her health update."
   Sends an email with the given subject and body and returns a confirmation.

4. schedule_meeting:
   e.g. schedule_meeting: "Schedule a meeting tomorrow at 10 AM with John."
   Schedules a meeting at the specified time with the mentioned person and returns a confirmation.

Example session:

Question: Can you schedule a meeting with the marketing team for next Monday at 3 PM?
Thought: I need to schedule a meeting with the marketing team for the specified date and time.
Action: schedule_meeting: "Next Monday at 3 PM with the marketing team."
PAUSE 

You will be called again with this:

Observation: Event scheduled successfully!.

Thought: The meeting has been scheduled, and I can confirm this action.
Answer: The meeting with the marketing team has been scheduled for next Monday at 3 PM.


Another session:

Question: Can you schedule a meeting with Josh from 3 pm to 4 pm on 2nd November?
Thought: I need to schedule a meeting with the marketing team for the specified date and time.
Action: schedule_meeting: "Next Monday meeting with Josh from 3 pm to 4 pm on 2nd November"
PAUSE 

You will be called again with this:

Observation: The requested time slot is not available. Please choose a different time.

Thought: The meeting could not be scheduled. User as to enter another time slot
Answer: Please choose another time since the requested time slot is not available.


Another session:

Question: Can you summarize the key findings from the pdfs?
Thought: I need to call read_pdfs tool to extract and summarize the key findings.
Action: read_pdfs: ["Summarise key findings"]
PAUSE 

You will be called again with this:

Observation: Key findings from Q1: "Sales increased by 20%... Major challenges included..." Key findings from Q2: "Profit margins improved... New product launch was successful..."

Thought: I have extracted the key findings from both reports.
Answer: The key findings are: Q1 - Sales increased by 20% with major challenges; Q2 - Profit margins improved and the new product launch was successful.

Another session:

Question: Send an email to Annie (annie@gmail.com) asking about her health update.
Thought: I need to draft and send an email to Annie asking about her health.
Action: send_gmail: "Send an email to Annie (annie@gmail.com) asking about her health update,send = True"
PAUSE 

You will be called again with this:

Observation: Email has been sent sucessfully. 

{'subject': 'Your Health Update',
 'message': 'Hi Janvita, I was just thinking about you and wanted to check in on your health. How are you feeling lately?',
 'to': ['janvita.reddy@example.com']}

Thought: The email has been sent, and I can confirm this action.
Answer: The email to Annie asking about her health update has been sent successfully with the message
{'subject': 'Your Health Update',
 'message': 'Hi Janvita, I was just thinking about you and wanted to check in on your health. How are you feeling lately?',
 'to': ['janvita.reddy@example.com']}.

Another session:

Question: What is the weather like today in College Station?
Thought: I need to search the internet to get real-time information for this query.
Action: search_internet: "Weather in College Station."
PAUSE 

You will be called again with this:

Observation: "The high is 85°F and the low is 68°F. There is a 6 mph wind, 86% humidity, and a dew point of 74°F."

Thought: I have obtained the required weather details.
Answer: The weather today in College Station is a high of 85°F and a low of 68°F with a 6 mph wind, 86% humidity, and a dew point of 74°F.

Now it's your turn:
""".strip()


def send_gmail(user_prompt, send=True):
    load_dotenv()
    credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/"],
        client_secrets_file="credentials.json",
    )
    api_resource = build_resource_service(credentials=credentials)
    toolkit = GmailToolkit(api_resource=api_resource)
    tools = toolkit.get_tools()
    model = ChatOllama(model='llama3-groq-tool-use')
    agent_executor = create_react_agent(
        model,
        tools,
        state_modifier=("You are an expert in writing and sending emails.")
    )
    instructions = (
        "Please draft the email in the following format:\n\n"
        "**Subject:** Create a clear and engaging subject line.\n\n"
        "**Greeting:** Start with a proper greeting (e.g., 'Hello' or 'Hi').\n\n"
        "**Body:** Write concise, relevant, and brief content with a professional tone. Ensure the body is separate from the greeting and signature by a new line.\n\n"
        "**Signature:** End with:\n\n"
        "Best regards,\n"
        "Janvita Reddy\n\n"
    )


    def prepare_prompt(user_prompt):
        return instructions + user_prompt

    user_input = HumanMessage(content=prepare_prompt(user_prompt))
    max_trials = 1
    attempts = 0
    message_dict = None
    draft_id = None

    while attempts < max_trials:
        try:
            response = agent_executor.invoke({"messages": [user_input]})
            ai_response = response['messages'][1]
            message_dict = ai_response.response_metadata['message']['tool_calls'][0]['function']['arguments']
            
            if message_dict:
                last_tool_message_content = response['messages'][2].content
                draft_id_prefix = "Draft Id: "

                if send:
                    if draft_id_prefix in last_tool_message_content:
                        draft_id = last_tool_message_content.split(draft_id_prefix)[1].strip()

                    if draft_id:
                        try:
                            api_resource.users().drafts().send(userId="me", body={"id": draft_id}).execute()
                            return f'Email has been sent successfully: {message_dict}'
                        except Exception as e:
                        # Handle any errors that occur during sending
                            return f'Failed to send email. Please check the recipient address and try again. Error: {str(e)}'
                    else:
                        return f'Email has been drafted, but draft ID was not found: {message_dict}'
                else:
                    return f'Email has been drafted {message_dict}'

        except Exception as e:
            attempts += 1
            print(f"Attempt {attempts} failed: {e}")

    raise Exception("Failed to generate email draft after multiple attempts.")


def search_internet(query):
   
    load_dotenv()

    # Set environment variables for LangChain
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
    os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')
    os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

    # Initialize tools and model
    search = TavilySearchResults(max_results=5)
    tools = [search]
    model = ChatGroq(model="llama3-8b-8192")

    # Create the agent with a system prompt for relevance and accuracy
    system_prompt = (
        "You are an expert in retrieving accurate, real-time information on diverse topics by searching the web. "
        "Your task is to return only relevant, precise information, discarding any irrelevant details. "
        "Focus on providing the correct answer with confidence, using only the best and most reliable sources available."
    )

    agent_executor = create_react_agent(model, tools, state_modifier=system_prompt)

    # Try querying with retries in case of failures
    max_retries = 5
    attempt = 0
    final_answer = None

    while attempt < max_retries and final_answer is None:
        try:
            # Process the query through the agent
            response = agent_executor.invoke({"messages": [HumanMessage(content=query)]})

            # Extract the response content
            for i in range(1, len(response["messages"])):
                if 'tool_calls' not in response['messages'][i].additional_kwargs:
                    final_answer = response["messages"][i].content.strip()
                    break

        except Exception as e:
            attempt += 1
            time.sleep(2)  # Brief pause before retrying

    if final_answer:
        return final_answer
    else:
        return "Failed to retrieve an answer after multiple attempts. Please try again later or rephrase your query."




load_dotenv()
model = ChatOllama(model='llama3-groq-tool-use')
SCOPES = ["https://www.googleapis.com/auth/calendar"]


def authenticate_google_calendar():
    creds = None
    if os.path.exists("token_cal.json"):
        creds = Credentials.from_authorized_user_file("token_cal.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token_cal.json", "w") as token:
            token.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)

def check_availability(service, start_time, end_time):
    events_result = service.events().list(
        calendarId="primary",
        timeMin=start_time,
        timeMax=end_time,
        singleEvents=True,
        orderBy="startTime",
    ).execute()
    events = events_result.get("items", [])
    return len(events) == 0

def create_event(service, title, start_time, end_time):
    event = {
        "summary": title,
        "start": {"dateTime": start_time, "timeZone": "UTC"},
        "end": {"dateTime": end_time, "timeZone": "UTC"},
    }
    event = service.events().insert(calendarId="primary", body=event).execute()
    return event["htmlLink"]

def get_llm_response(prompt, model):
    return model.invoke(prompt).content.strip()


def parse_schedule_request(request_text, model):
    prompt = f"""
    Interpret this scheduling request and return details in the following format:
    Title: <meeting title>
    Start Time: <start time in ISO 8601 format, e.g., 2024-07-27T14:00:00>
    End Time: <end time in ISO 8601 format, e.g., 2024-07-27T15:00:00>

    Request: {request_text}
    """
    response = get_llm_response(prompt, model)
    title_pattern = r"Title: (.+)"
    start_time_pattern = r"Start Time: (.+)"
    end_time_pattern = r"End Time: (.+)"
    
    title_match = re.search(title_pattern, response)
    start_time_match = re.search(start_time_pattern, response)
    end_time_match = re.search(end_time_pattern, response)
    
    title = title_match.group(1) if title_match else None
    start_time = start_time_match.group(1) if start_time_match else None
    end_time = end_time_match.group(1) if end_time_match else None

    ct_zone = pytz.timezone("America/Chicago")
    start_time_iso = ct_zone.localize(datetime.datetime.fromisoformat(start_time)).isoformat() if start_time else None
    end_time_iso = ct_zone.localize(datetime.datetime.fromisoformat(end_time)).isoformat() if end_time else None

    return title, start_time_iso, end_time_iso

def schedule_meeting(request_text):
    # Authenticate with Google Calendar
    service = authenticate_google_calendar()

    title, start_time, end_time = parse_schedule_request(request_text, model)
    
    # Check if all necessary data is parsed
    if not (title and start_time and end_time):
        return "Unable to parse the request. Please check the format of your input."
    
    # Check calendar availability and create event
    if check_availability(service, start_time, end_time):
        event_link = create_event(service, title, start_time, end_time)
        return "Event scheduled successfully!"
    else:
        return "The requested time slot is not available. Please choose a different time."
    



def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ''
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local('faiss_index')
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0)
    #model  = Groq(api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def read_pdfs(user_question):
    print(user_question)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response['output_text']

class Agent:
    def __init__(self, client, system):
        self.client = client
        self.system = system
        self.messages = []
        if self.system is not None:
            self.messages.append({'role': 'system', 'content': self.system})

    def __call__(self, message=""):
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({'role': 'assistant', 'content': result})
        return result

    def execute(self):
        completion = self.client.chat.completions.create(
            model="llama3-70b-8192", messages=self.messages
        )
        return completion.choices[0].message.content



def loop(max_iterations=10, query: str = ""):
    client = Groq(api_key=api_key)
    agent = Agent(client=client, system=system_prompt)

    # Tools to be used by the agent
    tools = ["schedule_meeting", "search_internet", "send_email", "read_pdfs"]
    next_prompt = query
    i = 0

    while i < max_iterations:
        i += 1
        result = agent(next_prompt)
        print(result)
        if "PAUSE" in result and "Action" in result:
            # Detect tool and arguments in the response
            action = re.findall(r"Action: ([a-z_]+): (.+)", result, re.IGNORECASE)
            if not action:
                next_prompt = "Observation: Action not recognized."
                continue
            
            chosen_tool, args = action[0]
            if chosen_tool == "schedule_meeting":
                result_tool = schedule_meeting(args)

            elif chosen_tool == "search_internet":
                result_tool = search_internet(args)

            elif chosen_tool == "send_gmail":
                email_args = args.strip()
                if ',' in email_args:
                    query, send_val = email_args.split(',')
                    query = query.strip()
                    send_val = send_val.strip()
                    print(send_val)
                    if "=" in send_val:
                        key, value = send_val.split("=")
                        value = value[:-1]
                        
                        send_value = value.strip() == "True"
                        print(send_value)
                        result_tool = send_gmail(query, send=send_value)
                    else:
                        result_tool = send_gmail(query)
                else:
                    result_tool = send_gmail(email_args)

            elif chosen_tool == "read_pdfs":

                result_tool = read_pdfs(args[1:-1])
                print(args[1:-1])
                print(result_tool)

            else:
                result_tool = "Observation: Tool not found"
            next_prompt = f"Observation: {result_tool}"
            continue
        if "Answer" in result:
            break

    return result

def main():
    st.header('Hi Janvita')
    with st.sidebar:
        st.title('Menu')
        pdf_docs = st.file_uploader('Upload PDF files', accept_multiple_files=True, type=['pdf'])

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Text processed and vector store created.")
                    else:
                        st.error("No text could be extracted from the PDFs.")
            else:
                st.error("Please upload at least one PDF.")

    user_question = st.text_input('How can I help you?')
    if user_question:
        result = loop(query=user_question) 
        st.write(result)

# Run the app
if __name__ == "__main__":
    main()


