# --- START OF REFACTORED FILE main.py (with token counting & caching) ---

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, model_validator
import logging
from typing import Any, Dict, List, Union, Iterable
import os
# Added anthropic imports
from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    DocumentBlockParam,
    ThinkingConfigParam,
    CacheControlEphemeralParam, # Import for caching
    MessageTokensCount, # Import for token counting response
)
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming # Needed for token count structure check
from anthropic.types.message_count_tokens_params import MessageCountTokensParams # Params for token count
from anthropic import RateLimitError, APIStatusError, APIConnectionError, APITimeoutError # Anthropic errors

import aiohttp
import datetime
from functools import lru_cache
from dotenv import load_dotenv
import json
from pathlib import Path

load_dotenv()

# --- Logging Configuration (Unchanged) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- FastAPI App and CORS (Unchanged) ---
anthropic_app = FastAPI(title="pyqchat-anthropic-pro")
webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
anthropic_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Allowed Subjects (Unchanged) ---
ALLOWED_SUBJECTS = {"DAA", "DBMS", "TOC", "WP", "OOPJ", "MPMC", "CVT", "COA"}

# --- Pydantic Models (Unchanged) ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    subject: str
    messages: List[ChatMessage]

    @model_validator(mode="before")
    def validate_subject(cls, values):
        subject = values.get('subject')
        if subject not in ALLOWED_SUBJECTS and subject != "TEST":
            raise ValueError(f"Subject '{subject}' is not allowed. Allowed subjects are: {', '.join(ALLOWED_SUBJECTS)}")
        return values

# --- Anthropic Client Initialization (Unchanged) ---
def get_client() -> AsyncAnthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    client = AsyncAnthropic(api_key=api_key)
    return client

# --- PDF Path Caching (Unchanged) ---
@lru_cache(maxsize=None)
def get_pdf_paths(subject: str) -> List[Path]:
    # ... (keep the existing implementation)
    logger.info(f"Getting PDF file paths for {subject}")
    paths = []
    subject_dir = Path(f"v1/{subject}")

    try:
        if not subject_dir.is_dir():
             logger.warning(f"Directory not found for subject: {subject}")
             return []

        for item in subject_dir.iterdir():
            if item.is_file() and item.suffix.lower() == ".pdf":
                logger.info(f"Found PDF file: {item}")
                paths.append(item)
        return paths
    except Exception as e:
        logger.error(f"Error getting PDF paths for {subject}: {str(e)}")
        raise

# --- Generate Response Function (Updated) ---
async def generate_response(subject: str, history: List[ChatMessage]):
    async_client = get_client()
    pdf_paths = get_pdf_paths(subject)

    if not pdf_paths:
        logger.warning(f"No PDF files found for subject: {subject}")
        content = f"No reference materials (PDFs) found for the subject '{subject}'. Cannot generate response."
        payload = {"type": "error", "content": content}
        yield f"data: {json.dumps(payload)}\n\n"
        return

    model_name = "claude-3-7-sonnet-20250219"
    max_tokens_to_generate = 2048
    # Define system prompt here to use in both count and generation
    system_prompt = "You are a helpful assistant that can answer questions about the subject and provide references to the relevant papers. You will be provided context of different question papers of a specific subject. The user, i.e., the students will ask you questions about the subject and you will answer them based on the context provided. When the user asks questions about the papers, you will be specific as to what paper and what date the paper is from."

    # 1. Construct Anthropic messages list (needed for both count and generation)
    anthropic_messages: List[MessageParam] = []
    first_user_message_processed = False
    for message in history:
        role = message.role.lower()
        if role == 'model':
            api_role = 'assistant'
        elif role == 'user':
            api_role = 'user'
        elif role == 'assistant':
            api_role = 'assistant'
        else:
            logger.warning(f"Skipping message with unknown role: {message.role}")
            continue

        content: Union[str, List[Union[TextBlockParam, DocumentBlockParam]]]

        if api_role == 'user' and not first_user_message_processed:
            content_blocks: List[Union[TextBlockParam, DocumentBlockParam]] = []
            content_blocks.append({"type": "text", "text": message.content})
            for pdf_path in pdf_paths:
                # Add cache_control here for prompt caching
                cache_control: CacheControlEphemeralParam = {"type": "ephemeral"}
                content_blocks.append(
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_path
                        },
                        "cache_control": cache_control # Add cache control
                    }
                )
            content = content_blocks
            first_user_message_processed = True
        else:
            content = message.content

        anthropic_messages.append({"role": api_role, "content": content}) # type: ignore

    if not anthropic_messages:
        logger.error(f"Cannot generate response for {subject}: No valid messages constructed.")
        payload = {"type": "error", "content": "No valid messages found in the provided chat history."}
        yield f"data: {json.dumps(payload)}\n\n"
        return

    # 2. Count tokens before proceeding
    try:
        logger.info(f"Counting tokens for {subject} request...")
        token_count_params: MessageCountTokensParams = {
            "messages": anthropic_messages,
            "model": model_name,
            "system": system_prompt,
            "thinking": {"type": "enabled", "budget_tokens": 1024}, # Add thinking to token count
            # Include tools here if you plan to use them in the generation step
            # "tools": [...]
        }
        token_count_result: MessageTokensCount = await async_client.messages.count_tokens(**token_count_params)
        input_tokens = token_count_result.input_tokens
        logger.info(f"Token count for {subject}: {input_tokens}")

        # 3. Check token limit
        MAX_INPUT_TOKENS = 128000
        if input_tokens > MAX_INPUT_TOKENS:
            logger.error(f"Token count {input_tokens} exceeds limit of {MAX_INPUT_TOKENS} for {subject}.")
            # Raise HTTPException here as we haven't started streaming yet.
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Input prompt and history are too long ({input_tokens} tokens). Maximum allowed is {MAX_INPUT_TOKENS} tokens."
            )

    except (APIStatusError, RateLimitError, APIConnectionError, APITimeoutError) as e:
        logger.error(f"API error during token count for {subject}: {str(e)}")
        # Raise an appropriate HTTP error before streaming starts
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to communicate with the AI service for token counting: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during token count for {subject}: {str(e)}")
        # Raise an appropriate HTTP error before streaming starts
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred during token counting: {str(e)}"
        )

    # 4. Proceed with generation if token count is okay
    try:
        logger.info(f"Streaming content for {subject} using model {model_name}.")
        full_response_for_log = ""
        # Optional: Configure thinking (ensure budget < max_tokens)
        thinking_config: ThinkingConfigParam = {"type": "enabled", "budget_tokens": 1024} # Enable thinking here

        async with async_client.messages.stream(
            model=model_name,
            messages=anthropic_messages, # Reuse the constructed messages
            system=system_prompt,        # Reuse the system prompt
            max_tokens=max_tokens_to_generate,
            thinking=thinking_config, # Pass thinking config
            # tools=... # Pass tools if defined and counted above
        ) as stream:
            async for event in stream:
                logger.info(f"Received Stream Event: {event.type}")

                if event.type == "text":
                    logger.debug(f"Response chunk: {event.text}")
                    payload = {"type": "response", "chunk": event.text}
                    yield f"data: {json.dumps(payload)}\n\n"
                    full_response_for_log += event.text
                elif event.type == "thinking":
                    logger.info(f"Model thought chunk: {event.thinking}")
                    thought_payload = {"type": "thought", "content": event.thinking}
                    yield f"data: {json.dumps(thought_payload)}\n\n"
                elif event.type == "content_block_delta":
                     if event.delta.type == "text_delta":
                         logger.debug(f"Response chunk (delta): {event.delta.text}")
                         payload = {"type": "response", "chunk": event.delta.text}
                         yield f"data: {json.dumps(payload)}\n\n"
                         full_response_for_log += event.delta.text
                     elif event.delta.type == "thinking_delta":
                         logger.info(f"Model thought chunk (delta): {event.delta.thinking}")
                         thought_payload = {"type": "thought", "content": event.delta.thinking}
                         yield f"data: {json.dumps(thought_payload)}\n\n"
                elif event.type == "message_stop":
                    logger.info("Message stream finished.")
                    final_usage = (await stream.get_final_message()).usage
                    logger.info(f"Final Usage: Input={final_usage.input_tokens}, Output={final_usage.output_tokens}")
                    # Optionally send a final "done" message
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"


        # Log the full response after streaming is complete
        try:
            last_user_message = next((msg.content for msg in reversed(history) if msg.role == 'user'), "N/A")
            await log_to_discord(last_user_message, full_response_for_log if full_response_for_log else "[No text content received]")
        except Exception as e:
            logger.error(f"Failed to log to Discord after streaming: {e}")


    except (APIStatusError, RateLimitError, APIConnectionError, APITimeoutError) as e:
        logger.error(f"API error during generation for {subject}: {str(e)}")
        content = f"Failed to generate response due to API error: {str(e)}"
        payload = {"type": "error", "content": content}
        yield f"data: {json.dumps(payload)}\n\n"
    except Exception as e:
        logger.error(f"Error generating streaming response for {subject}: {str(e)}")
        content = f"Failed to generate response: {str(e)}"
        payload = {"type": "error", "content": content}
        yield f"data: {json.dumps(payload)}\n\n"

# --- Discord Logging (Unchanged) ---
async def log_to_discord(user_query: str, bot_response: str):
    # ... (keep the existing implementation)
    log_entry = (
        f"**[{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).strftime('%Y-%m-%d %H:%M:%S %Z')}]**\n"
        f"**User Query:** {user_query}\n"
        f"**Bot Response:** {bot_response}\n"
        "―――――――――――――――――――――――――――"
    )
    if not webhook_url:
        logger.warning("DISCORD_WEBHOOK_URL not set, skipping logging.")
        return
    async with aiohttp.ClientSession() as session:
        try:
            await session.post(
                webhook_url,
                json={"content": log_entry},
                timeout=aiohttp.ClientTimeout(total=3)
            )
            logger.info("Successfully logged to Discord.")
        except Exception as e:
            logger.error(f"Failed to log to Discord: {e}")

# --- Status Endpoint (Unchanged) ---
@anthropic_app.get("/status")
async def get_status():
    # ... (keep the existing implementation)
     try:
        return {"status": "OK", "library": "anthropic-python"}
     except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service health check failed"
        )

# --- Dynamic Subject Endpoint (Modified to handle token limit exception) ---
@anthropic_app.post("/{subject}")
async def subject_endpoint(subject: str, request_data: ChatRequest):
    try:
        current_subject = request_data.subject
        if subject != current_subject and subject != "TEST":
             raise HTTPException(
                 status_code=status.HTTP_400_BAD_REQUEST,
                 detail=f"Path subject '{subject}' does not match request body subject '{current_subject}'."
             )

        logger.info(f"Processing {current_subject} request for SSE. History length: {len(request_data.messages)}")

        # The generate_response function now handles the token limit check internally
        # before starting the stream. If it raises HTTPException, FastAPI handles it.
        return StreamingResponse(
            generate_response(current_subject, request_data.messages),
            media_type="text/event-stream"
        )

    except ValueError as ve: # Catches Pydantic validation errors
         logger.error(f"Validation error for {subject}: {str(ve)}")
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request data: {str(ve)}"
         )
    except HTTPException as http_exc: # Re-raise HTTPExceptions from generate_response
        logger.error(f"HTTP Exception during setup for {subject}: {http_exc.detail} (Status: {http_exc.status_code})")
        raise http_exc # FastAPI will handle this
    except Exception as e: # Catch other unexpected errors during setup
        logger.error(f"Error in {subject} endpoint before streaming: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate {subject} request stream: {str(e)}"
        )

# --- END OF REFACTORED FILE main.py (with token counting & caching) ---