from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict, Annotated
from langchain_core.agents import AgentAction, AgentFinish
import operator

class SupervisorState(TypedDict):
    """State for supervisor workflow"""
    messages: Annotated[list, operator.add]
    next_agent: str
    task_description: str

def create_supervisor_agent(model, message_agent, email_agent, checkpointer):
    """
    Create a supervisor agent that routes tasks to specialist agents.
    
    Args:
        model: Language model to use for routing decisions
        message_agent: Agent for message generation
        email_agent: Agent for email sending
    
    Returns:
        Compiled supervisor graph
    """
    
    # Define the routing logic node
    def route_task(state: SupervisorState):
        """Analyze task and decide which agent to route to"""
        messages = state.get("messages", [])
        
        if not messages:
            return {"next_agent": "error", "task_description": "No messages provided"}
        
        # Get the last user message
        last_message = messages[-1]
        if isinstance(last_message, dict):
            task_text = last_message.get("content", "")
        else:
            task_text = last_message.content if hasattr(last_message, "content") else str(last_message)
        
        # Use the model to classify the task
        classification_prompt = f"""
        You are a task router. Analyze this request and determine if it should be handled by:
        1. message_generator - for creating, writing, or generating messages/content
        2. email_sender - for sending emails or email-related tasks

        Request: {task_text}

        Respond with ONLY one word: either '[msg]' or '[eml]'
        """
        
        routing_response = model.invoke([HumanMessage(content=classification_prompt)])
        routing_text = routing_response.content.strip().lower()
        
        # Determine which agent to route to
        if "email" in routing_text or "email_sender" in routing_text:
            next_agent = "email_agent"
        elif "message" in routing_text or "message_generator" in routing_text:
            next_agent = "message_agent"
        else:
            # Default routing based on keywords in original message
            if any(word in task_text.lower() for word in ["email", "send", "recipient", "smtp"]):
                next_agent = "email_agent"
            elif any(word in task_text.lower() for word in ["generate", "write", "create", "message", "text"]):
                next_agent = "message_agent"
            else:
                next_agent = "message_agent"  # Default
        
        return {
            "next_agent": next_agent,
            "task_description": task_text
        }
    
    # Message generation node
    def message_generator_node(state: SupervisorState):
        """Route to message generation agent"""
        messages = state.get("messages", [])
        
        try:
            # Invoke message agent
            result = message_agent.invoke({
                "messages": messages
            })
            
            # Extract the response
            if hasattr(result, "content"):
                agent_response = result.content
            elif isinstance(result, dict):
                agent_response = result.get("messages", [])
                if agent_response and hasattr(agent_response[-1], "content"):
                    agent_response = agent_response[-1].content
            else:
                agent_response = str(result)
            
            return {
                "messages": [
                    AIMessage(content=agent_response)
                ]
            }
        except Exception as e:
            return {
                "messages": [
                    AIMessage(content=f"[Message Agent Error] {str(e)}")
                ]
            }
    
    # Email sending node
    def email_sender_node(state: SupervisorState):
        """Route to email sending agent"""
        messages = state.get("messages", [])
        
        try:
            # Invoke email agent
            result = email_agent.invoke({
                "messages": messages
            })
            
            # Extract the response
            if hasattr(result, "content"):
                agent_response = result.content
            elif isinstance(result, dict):
                agent_response = result.get("messages", [])
                if agent_response and hasattr(agent_response[-1], "content"):
                    agent_response = agent_response[-1].content
            else:
                agent_response = str(result)
            
            return {
                "messages": [
                    AIMessage(content=agent_response)
                ]
            }
        except Exception as e:
            return {
                "messages": [
                    AIMessage(content=f"[Email Agent Error] {str(e)}")
                ]
            }
    
    # Conditional routing function
    def should_route(state: SupervisorState):
        """Determine next step based on routing decision"""
        next_agent = state.get("next_agent", "message_agent")
        
        if next_agent == "email_agent":
            return "email_agent"
        elif next_agent == "message_agent":
            return "message_agent"
        else:
            return "end"
    
    # Build the graph
    workflow = StateGraph(SupervisorState)
    
    # Add nodes
    workflow.add_node("router", route_task)
    workflow.add_node("message_agent", message_generator_node)
    workflow.add_node("email_agent", email_sender_node)
    
    # Add edges
    workflow.add_edge(START, "router")
    
    # Conditional routing based on classification
    workflow.add_conditional_edges(
        "router",
        should_route,
        {
            "message_agent": "message_agent",
            "email_agent": "email_agent",
            "end": END
        }
    )
    
    # Both agents go to end
    workflow.add_edge("message_agent", END)
    workflow.add_edge("email_agent", END)
    
    # Compile the graph
    supervisor_graph = workflow.compile(checkpointer=checkpointer)
    
    return supervisor_graph