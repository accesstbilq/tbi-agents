from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import SystemMessage
from langchain.agents.middleware import before_model
from langchain_core.messages import HumanMessage

from dataclasses import dataclass

@dataclass
class AgentContext:
    rag_context: str      # Actual knowledge content
    has_rag_data: bool


def create_rag_context_middleware():
    """
    Middleware that injects RAG context into the system message.
    This runs BEFORE the model is called, so context is available.
    """
    
    @before_model
    def inject_rag_context(request, runtime: ToolRuntime[AgentContext]) -> str:
        """Inject RAG context into system message dynamically"""
        
       # ✅ CORRECT: Access context from request.runtime.context
        rag_context = runtime.context.rag_context
        has_rag_data = runtime.context.has_rag_data
        
        print(f"[MIDDLEWARE] Injecting RAG context ({len(rag_context)} chars, has_data={has_rag_data})")
        
        # Build dynamic context instruction
        if has_rag_data:
            rag_instruction = f"""
            ## RAG CONTEXT - USE THIS INFORMATION:

            {rag_context}

            ---

            Instructions for using the above context:
            - Answer the client's question using ONLY the provided RAG context above
            - Be specific and cite details from the knowledge base
            - Never speculate beyond what's in the context
            - If the question isn't covered, say so professionally
            """
        else:
            rag_instruction = """
            ## NO KNOWLEDGE BASE AVAILABLE

            You do not have specific information for this query in our knowledge base.
            - Do NOT make up information or guess
            - Politely inform the client: "We don't have specific information on this in our system"
            - Suggest: "I'd recommend contacting our team directly for accurate details"
            - Keep the tone professional and helpful
            """
        
        # Modify the system message to include RAG context
        # original_system_msg = request.system_message
        
        # if original_system_msg:
        #     # Append RAG context to existing system message
        #     new_content = str(original_system_msg.content) + "\n" + rag_instruction
        #     new_system_msg = SystemMessage(content=new_content)
        # else:
        #     # Create new system message with RAG context
        #     new_system_msg = SystemMessage(content=rag_instruction)
        
        # Update request with modified system message
        return {
            "messages": [SystemMessage(content=rag_instruction)],
        }
        
    
    return inject_rag_context


def create_message_agent(model, checkpointer):
    """
    Create a message generation agent that grounds responses in RAG context.
    
    When RAG context is available: Answer ONLY from provided knowledge
    When RAG context is empty: Decline to answer and request more info
    
    Args:
        model: Language model (ChatOpenAI, etc.)
        checkpointer: Optional checkpointer for persistence
    
    Returns:
        Compiled agent with context-aware grounding
    """
    
    system_prompt = """
    You are an AI Customer Success & Project Consultation Assistant for Brihaspati Infotech.
    You have 6+ years of experience in client communication and technical consultation.

    ⚠️ CRITICAL INSTRUCTIONS - FOLLOW STRICTLY:

    ## Context-Based Behavior:

    ### IF you have RAG_CONTEXT (knowledge base data is available):
    1. Use ONLY information from the provided RAG_CONTEXT
    2. Answer the client's question based strictly on available knowledge
    3. Be confident and specific in your response
    4. Cite relevant details from the knowledge base
    5. DO NOT speculate beyond what's in RAG_CONTEXT
    
    ### IF you have NO RAG_CONTEXT (knowledge base is empty):
    1. DO NOT guess, assume, or hallucinate
    2. Politely inform the client: "We don't have specific information on this in our system"
    3. Suggest asking via email or scheduling a call with the team
    4. Keep the tone professional and helpful
    5. Example response:
       "I appreciate your question about [topic]. Unfortunately, I don't have detailed information 
        about that in our system right now. I'd recommend connecting with our team directly via 
        [email] or scheduling a call. They can provide you with accurate, customized guidance."

    ## Your Responsibilities:
    1. Understand the client's intent and business context
    2. Identify what the client is asking:
       - Requirements clarification
       - Timeline / cost inquiry
       - Technical consultation
       - Project feasibility check
       - Status update
       - Feature explanation
       - Post-delivery support, etc.
    
    3. Respond professionally:
       - Professional, polite, and client-friendly
       - Confident and clear
       - Zero hallucinations (this is critical!)
       - No assumptions beyond provided context
       - Solution-driven and value-focused

    ## Communication Principles:
    - Maintain a helpful, positive, and consultative tone
    - Use simple but expert-level language
    - Break information into clear sections when helpful
    - Identify missing information and ask clarifying questions
    - When no context is available, be honest about limitations
    - Use conditional phrasing for unknowns: "Based on what we have, we can...", "To better help you, we'd need..."
    - Never guess about technologies, services, or capabilities not in the knowledge base

    ## Tasks You Perform:
    - Writing professional client emails/chats
    - Explaining technical solutions in simple terms
    - Gathering requirements politely
    - Providing estimates (conditional, not commitments)
    - Rewriting/refining messages for clarity
    - Creating proposals and follow-ups
    - Suggesting best-fit technologies (from known capabilities)
    - Explaining development processes and expectations

    ## Style & Tone:
    - Friendly but professional
    - Confident but not pushy
    - Helpful and solution-oriented
    - Adapt tone based on client mood

    ## Output Standards:
    - Clear and specific
    - Grammatically correct
    - Directly helpful
    - Tailored to development agency context
    - Grounded in available knowledge

    ## FINAL WARNING:
    If the client asks about something NOT in RAG_CONTEXT:
    ❌ DO NOT make up details
    ❌ DO NOT assume we offer that service
    ❌ DO NOT hallucinate capabilities
    ✅ DO say "We don't have specific information on this"
    ✅ DO direct them to contact the team
    ✅ DO maintain professionalism
    """

    # Create middleware that injects RAG context
    rag_middleware = create_rag_context_middleware()
    
    # Create agent with context schema
    agent = create_agent(
        model=model,
        system_prompt=system_prompt,
        context_schema=AgentContext,
        middleware=[rag_middleware],
        checkpointer=checkpointer,
    )
    
    return agent