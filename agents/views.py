import os
import json
import traceback
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv
from django.shortcuts import render
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from .helpers.stream_helper import stream_generator
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from langchain.agents import AgentState
from django.http import HttpRequest, JsonResponse,StreamingHttpResponse
from langchain_core.messages import HumanMessage

from .services.message_agent import create_message_agent
from .services.email_agent import create_email_agent
from .services.supervisor import create_supervisor_agent

# Initialize langchain short memory
if os.getenv("ENV_TYPE") == 'localhost':
    DB_URI = os.getenv("POSTGRES_URL")
else:
    DB_URI = os.getenv("POSTGRES_URL_PROD")

# Initialize connection pool
connection_pool = ConnectionPool(DB_URI, min_size=1, max_size=5)

def init_checkpointer():
    with connection_pool.connection() as conn:
        conn.autocommit = True
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        print("INIT CHECKPOINTER #####")


# Call once at startup
try:
    init_checkpointer()
except Exception as e:
    print(f"⚠ Warning during checkpointer init: {e}")
    # Don't crash the app - tables might already exist




# LOAD ENV VARIABLE
load_dotenv()

class CustomAgentState(AgentState):
    """Custom state with messages + custom fields"""
    categories: list = []
    context_snippets: list = []
    base64_string: str = ""
    file_name: str | None = None

def index(request: HttpRequest):
    """Render home page"""
    return render(request, "index.html")


def chatbot_view(request: HttpRequest):
    """Render chatbot page"""
    return render(request, "chat.html")


@csrf_exempt
@require_POST
def chat_asistance(request: HttpRequest):
    """Handle chat with dual output: JSON structure + formatted response."""
    
    # ---- Parse request ----
    session_id = request.POST.get("session_id")
    user_message = request.POST.get("user_message")

    if not session_id:
        return JsonResponse({"error": "session_id is required"}, status=400)

    config = {"configurable": {"thread_id": session_id}}

    # ---- Create model ----
    model = ChatOpenAI(model="gpt-4.1", temperature=0.1)

    
    # Create agent with checkpointer
    with connection_pool.connection() as conn:
        checkpointer = PostgresSaver(conn)
        # Create specialist agents
        message_agent = create_message_agent(model, checkpointer)
        # email_agent = create_email_agent(model, checkpointer)

        # Create supervisor
        supervisor_agent = create_supervisor_agent(model, message_agent, checkpointer)

        agent_input = {
            "messages": [HumanMessage(content=user_message)],
            "user_message": user_message
        }
        
        try:
            response = StreamingHttpResponse(
                stream_generator(
                    agent=supervisor_agent,
                    agent_input=agent_input,
                    config=config,
                ),
                content_type="text/event-stream",
                charset="utf-8",
            )
            response['Cache-Control'] = 'no-cache'
            return response
        except Exception as e:
            print(f"Error in agent: {e}")
            error_detail = traceback.format_exc()
            return JsonResponse(
                {"error": str(e), "detail": error_detail}, 
                status=400
            )