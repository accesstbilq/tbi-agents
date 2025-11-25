from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def generate_professional_message(topic: str, recipient: str = "General", tone: str = "professional") -> str:
    """
    Generate a professional message for a given topic.
    
    Args:
        topic: The subject matter of the message
        recipient: Who the message is for (e.g., "client", "manager", "team")
        tone: The tone of the message (professional, friendly, formal, casual)
    
    Returns:
        A well-written message
    """
    return f"""
Generated {tone} message for {recipient}:

Subject: {topic}

Dear {recipient},

[Message content about {topic} in {tone} tone]

Thank you for your attention to this matter.

Best regards
"""

@tool
def generate_greeting_message(name: str, context: str = "") -> str:
    """
    Generate a greeting or welcome message.
    
    Args:
        name: The name of the person to greet
        context: Additional context for the greeting
    
    Returns:
        A personalized greeting message
    """
    return f"Hello {name},\n\nWelcome! {context}\n\nWe're excited to work with you."

@tool
def generate_follow_up_message(previous_topic: str, action_items: str = "") -> str:
    """
    Generate a follow-up message based on a previous conversation.
    
    Args:
        previous_topic: What was discussed previously
        action_items: Any action items to mention
    
    Returns:
        A professional follow-up message
    """
    return f"""
Follow-up Message:

I wanted to follow up on our discussion about {previous_topic}.

As we discussed, here are the action items:
{action_items}

Please let me know if you have any questions.

Best regards
"""

@tool
def generate_summary_message(meeting_points: str) -> str:
    """
    Generate a summary message from meeting points.
    
    Args:
        meeting_points: Key points from a meeting
    
    Returns:
        A formatted summary message
    """
    return f"""
Meeting Summary:

Key Discussion Points:
{meeting_points}

Next Steps:
- Review the action items assigned
- Provide feedback by next meeting
- Contact me with any questions

Thank you for attending.
"""

def create_message_agent(model, checkpointer):
    """
    Create a message generation agent.
    
    Args:
        model: Language model to use (ChatOpenAI, ChatAnthropic, etc.)
        checkpointer: Optional checkpointer for persistence
    
    Returns:
        Compiled agent ready for message generation tasks
    """
    
    tools = [
        generate_professional_message,
        generate_greeting_message,
        generate_follow_up_message,
        generate_summary_message,
    ]
    
    system_prompt = """
You are an expert message writing assistant. Your job is to help users generate well-written, 
professional messages for various purposes.

When a user asks you to generate a message:
1. Understand the context and purpose
2. Determine the appropriate tone (professional, friendly, formal, etc.)
3. Identify the recipient or audience
4. Use the appropriate tool to generate the message
5. If needed, refine or customize the output

You have access to tools for:
- Professional messages for clients and managers
- Greeting and welcome messages
- Follow-up messages after meetings or conversations
- Summary messages from meetings

Always ensure messages are:
- Clear and concise
- Appropriately formal or casual based on context
- Free of spelling and grammar errors
- Well-structured and easy to read
"""
    
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer
    )
    
    return agent