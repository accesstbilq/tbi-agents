import json
import time
import traceback
from typing import Dict, Any, Generator
import re
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

def stream_generator(
    agent,
    agent_input: Dict[str, Any],
    config: Dict[str, Any],
) -> Generator[str, None, None]:
    """
    Streams SSE-like events from a multi-step agent with real-time token streaming.
    
    Emits events:
    - token: Individual tokens as they arrive (for real-time display)
    - message: Complete message when done
    - usage: Token usage statistics
    - error: Error messages if something fails
    """
    def emit_sse(obj: dict) -> str:
        """Format SSE data line"""
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    # Token usage tracking
    prompt_tokens = 0
    completion_tokens = 0

    try:
        print(f"[DEBUG] Starting agent stream with config: {config}")
        
        # ============================================
        # STREAM AGENT EXECUTION - USE stream_mode="messages"
        # ============================================
        for step in agent.stream(
            agent_input, 
            config=config, 
            stream_mode="messages"  # ✅ Changed to "messages" for token streaming
        ):
            last_message = step[0]
            content = getattr(last_message, "content", None)

            yield emit_sse({
                "type": "streaming",
                "message": content
            })

            # ============================================
            # TOKEN USAGE (Extract from any message)
            # ============================================
            if hasattr(last_message, "usage_metadata") and last_message.usage_metadata:
                usage = last_message.usage_metadata
                in_tokens = usage.get("input_tokens")
                out_tokens = usage.get("output_tokens")

                if in_tokens is not None:
                    prompt_tokens = in_tokens
                if out_tokens is not None:
                    completion_tokens = out_tokens

                print(f"[DEBUG] Token usage found: input={in_tokens}, output={out_tokens}")
        
        total_tokens = prompt_tokens + completion_tokens
        yield emit_sse({
            "type": "usage",
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens
        })

        print(f"[DEBUG] Stream complete")
        
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"Stream error: {error_detail}")

        yield emit_sse({
            "type": "error",
            "message": str(e),
            "detail": error_detail,
            "timestamp": time.time()
        })