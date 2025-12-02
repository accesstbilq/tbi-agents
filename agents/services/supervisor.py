from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.retrievers import MultiQueryRetriever

from typing import Optional, Annotated
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from pathlib import Path
import operator
from dataclasses import dataclass


BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_DB_PATH = BASE_DIR / "agents/chroma_db"
COLLECTION_NAME = 'project_portfolio'


# Use TypedDict for state schema (not BaseModel) - LangChain v1 requirement
class SupervisorState(TypedDict):
    """State for supervisor workflow"""
    messages: Annotated[list, operator.add]
    intent: str
    confidence: Optional[float]
    next_agent: str
    task_description: str
    rag_data: Optional[list]  # Added for storing retrieval results
    user_message: str  # Added to track current user message

@dataclass
class AgentContext:
    rag_context: str      # Actual knowledge content
    has_rag_data: bool

class RouteQuery(BaseModel):
    """Route user query to the correct knowledge bucket."""
    intent: str = Field(
        ...,
        description="Select one: 'technical_capability', 'domain_expertise', 'business_trust', 'engagement_hiring', 'process_communication', 'general_chat'"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Optional confidence score from 0.0 to 1.0"
    )


def get_vectorstore() -> Chroma:
    """
    Re-open the existing Chroma collection using the same settings
    used when creating the embeddings.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_PATH),
    )
    return vectorstore


def create_supervisor_agent(llm, message_agent, checkpointer):
    """
    Create a supervisor agent that routes tasks to specialist agents.
    
    Args:
        llm: Language model to use for routing decisions
        message_agent: Agent for message generation
        checkpointer: Checkpoint saver for graph persistence
    
    Returns:
        Compiled supervisor graph
    """

    def intent_classifier_node(state: SupervisorState) -> SupervisorState:
        """Classify user intent using structured output"""
        system_prompt = """
        You are a sophisticated intent classifier for Brihaspati Infotech.
        Map the user's question to the correct category:

        1. 'engagement_hiring': hire, developers, quote, cost, rate, hourly, team.
        2. 'process_communication': communication, agile, sprint, NDA, support, maintenance, how do you work.
        3. 'technical_capability': React, Node, AWS, scaling, security, integrations.
        4. 'business_trust': startups, success stories, ratings, reviews, why choose you.
        5. 'domain_expertise': dashboard, LMS, cart, fintech, healthcare.
        6. 'general_chat': greetings and small talk.
        """

        print("[CALLED INTENT CLASSIFIER NODE].................", state.get("user_message", ""))

        # Use with_structured_output for structured output
        llm_with_structure = llm.with_structured_output(RouteQuery)
        
        llm_response = llm_with_structure.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=state.get("user_message", ""))
        ])

        print("[RESPONSE INETNT CATGH].................", llm_response)

        return {
            "intent": llm_response.intent,
            "confidence": llm_response.confidence or 0.0,
        }

    def rag_executor_node(state: SupervisorState) -> SupervisorState:
        """Execute RAG retrieval based on intent"""
        intent = state.get("intent", "")
        confidence = state.get("confidence", 0.0)
        user_message = state.get("user_message", "")

        print("[CALLED RAG CLASSIFIER NODE].................")

        if not intent:
            # Low confidence - return empty results
            return {"rag_data": []}
        
        try:
            vectorstore = get_vectorstore()
            
            base_retriever = vectorstore.as_retriever(
                search_type="mmr", 
                search_kwargs={
                    "k": 6,          # How many unique docs to return per query
                    "fetch_k": 20,   # Pool of docs to select from
                }
            )

            retrieval_llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

            # MultiQueryRetriever
            advanced_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=retrieval_llm
            )

            # Invoke returns List[Document]
            unique_docs = advanced_retriever.invoke(user_message)

            print(f"[RAG] Retrieved {len(unique_docs)} documents")

            return {"rag_data": unique_docs}
        
        except Exception as e:
            print(f"[RAG Error] {str(e)}")
            return {"rag_data": []}

    def route_task(state: SupervisorState) -> SupervisorState:
        """Analyze task and decide which agent to route to"""
        messages = state.get("messages", [])

        print("[CALLED ROUTE TASK].................")

        if not messages:
            return {
                "next_agent": "end",
                "task_description": "No messages provided"
            }
        
        # Get the last user message
        last_message = messages[-1]
        if isinstance(last_message, dict):
            task_text = last_message.get("content", "")
        else:
            task_text = last_message.content if hasattr(last_message, "content") else str(last_message)

        return {
            "next_agent": "message_agent",
            "task_description": task_text
        }

    def message_generator_node(state: SupervisorState) -> SupervisorState:
        """
        Generate response using message agent with RAG context grounding.
        
        Flow:
        1. Extract RAG data from state
        2. Build context string from RAG documents
        3. Create AgentContext with grounding info
        4. Invoke agent with context
        5. Extract and return response
        """
        messages = state.get("messages", [])
        rag_data = state.get("rag_data", [])

        print("[MESSAGE GENERATOR NODE].................")
        print(f"[MESSAGE GENERATOR] Processing {len(messages)} messages with {len(rag_data)} RAG docs")
        
        try:
            # Build RAG context from documents
            rag_context = ""
            has_rag_data = len(rag_data) > 0
            
            if has_rag_data:
                # Combine document content with metadata for richer context
                context_parts = []
                for doc in rag_data[:5]:  # Use top 5 docs
                    category = doc.metadata.get("category", "General")
                    content = doc.page_content
                    context_parts.append(f"[{category}] {content}")
                
                rag_context = "\n\n".join(context_parts)
                print(f"[MESSAGE GENERATOR] RAG Context (length: {len(rag_context)}):")
                print(rag_context[:200] + "...\n")
            else:
                print("[MESSAGE GENERATOR] ⚠️ NO RAG DATA - Agent will not hallucinate")
                rag_context = ""

            # Invoke agent with context - THIS IS WHERE GROUNDING HAPPENS
            result = message_agent.invoke(
                {"messages": messages},
                context=AgentContext(
                    rag_context=rag_context,
                    has_rag_data=has_rag_data
                )
            )
            
            # Extract the response
            if hasattr(result, "messages") and result.messages:
                # Agent returns state with messages
                last_message = result.messages[-1]
                agent_response = last_message.content if hasattr(last_message, "content") else str(last_message)
            elif hasattr(result, "content"):
                agent_response = result.content
            elif isinstance(result, dict):
                agent_response = result.get("messages", [])
                if agent_response and hasattr(agent_response[-1], "content"):
                    agent_response = agent_response[-1].content
            else:
                agent_response = str(result)
            
            print(f"[MESSAGE GENERATOR] Generated response: {agent_response[:100]}...")
            
            return {
                "messages": [AIMessage(content=agent_response)]
            }
            
        except Exception as e:
            print(f"[MESSAGE GENERATOR ERROR] {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "messages": [AIMessage(content=f"Error processing request: {str(e)}")]
            }
    
    def should_route(state: SupervisorState) -> str:
        """Determine next step based on routing decision"""
        next_agent = state.get("next_agent", "end")
        print("[CALLED SHOULD ROUTE].................")

        return next_agent if next_agent != "end" else END

    # Build the graph using StateGraph
    workflow = StateGraph(SupervisorState)
    
    # Add nodes
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("rag_executor", rag_executor_node)
    workflow.add_node("router", route_task)
    workflow.add_node("message_agent", message_generator_node)

    # Add edges
    workflow.add_edge(START, "intent_classifier")
    workflow.add_edge("intent_classifier", "rag_executor")
    workflow.add_edge("rag_executor", "router")
    
    # Conditional routing based on classification
    workflow.add_conditional_edges(
        "router",
        should_route,
        {
            "message_agent": "message_agent",
            "end": END
        }
    )
    
    # Message agent goes to end
    workflow.add_edge("message_agent", END)
    
    # Compile the graph with checkpointer
    supervisor_graph = workflow.compile(checkpointer=checkpointer)
    
    return supervisor_graph