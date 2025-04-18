from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, model_validator
import logging
from typing import Any, Dict, List
import os
from google import genai
from google.genai import types
import aiohttp
import datetime
from functools import lru_cache
from dotenv import load_dotenv
import json

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="pyqchat")

webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define allowed subjects
ALLOWED_SUBJECTS = {"DAA", "DBMS", "TOC", "WP", "OOPJ", "MPMC", "CVT", "COA"}

# Pydantic model for individual messages in the history
class ChatMessage(BaseModel):
    role: str
    content: str

# Pydantic model for the incoming chat request body
class ChatRequest(BaseModel):
    message: str # The latest message from the user
    subject: str
    messages: List[ChatMessage] # The history of the conversation

    @model_validator(mode="before")
    def validate_subject(cls, values):
        subject = values.get('subject')
        if subject not in ALLOWED_SUBJECTS and subject != "TEST":
            raise ValueError(f"Subject '{subject}' is not allowed. Allowed subjects are: {', '.join(ALLOWED_SUBJECTS)}")
        return values

# Create a Gemini client
def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        raise ValueError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    return client

# Get PDF files for a subject with caching
@lru_cache(maxsize=None) # Use lru_cache for simplicity
def get_pdf_files(subject):
    logger.info(f"Loading PDF files for {subject}")
    files = []

    client = get_client() 
    subject_dir = f"v1/{subject}"

    try:
        if not os.path.isdir(subject_dir):
             logger.warning(f"Directory not found for subject: {subject}")
             return [] # Return empty list if directory doesn't exist

        for filename in os.listdir(subject_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(subject_dir, filename) 
                logger.info(f"Uploading file: {file_path}")
                uploaded_file = client.files.upload(file=file_path)
                files.append(uploaded_file)
        return files
    except Exception as e:
        logger.error(f"Error loading PDF files for {subject}: {str(e)}")
        raise 

# Generate function for subject-specific responses, now accepting history
async def generate_response(subject: str, history: List[ChatMessage]):
    # client_module = get_client(subject) # Get the configured genai module
    client = get_client() # Get the client instance
    files = get_pdf_files(subject) # This will use the cached result if available

    if not files:
        logger.warning(f"No PDF files found for subject: {subject}")
        # Yield a message indicating no files were found
        content = f"No reference materials (PDFs) found for the subject '{subject}'. Cannot generate response."
        payload = {"type": "message", "content": content}
        yield f"data: {json.dumps(payload)}\\n\\n"
        return # Stop execution for this request

    model_name = "models/gemini-2.0-flash-thinking-exp-01-21" 

    try:
        # Construct the prompt contents including files and chat history
        # The first message should include the file context
        content_parts = [
            *[types.Part.from_uri(
                mime_type=file.mime_type,
                file_uri=file.uri
            ) for file in files]
        ]

        # Process history: Ensure roles are correct ('user' or 'model')
        contents = []
        first_user_message = True
        for message in history:
            role = message.role.lower() # Normalize role
            if role not in ['user', 'model', 'assistant']: # Allow 'assistant' as alias for 'model'
                logger.warning(f"Skipping message with unknown role: {message.role}")
                continue

            # Map 'assistant' to 'model' for the API
            api_role = 'model' if role == 'assistant' else role

            if api_role == 'user' and first_user_message:
                 # Add file parts only to the first user message parts
                 current_parts = content_parts + [types.Part.from_text(text=message.content)]
                 first_user_message = False
            else:
                 current_parts = [types.Part.from_text(text=message.content)]

            contents.append(types.Content(role=api_role, parts=current_parts))

        if not contents:
             logger.error(f"Cannot generate response for {subject}: No valid messages in history.")
             # Yield an error message for SSE
             payload = {"type": "error", "content": "No valid messages found in the provided chat history."}
             yield f"data: {json.dumps(payload)}\\n\\n"
             # Optionally raise an exception if you want server logs, but yielding is better for client feedback
             # raise HTTPException(
             #      status_code=status.HTTP_400_BAD_REQUEST,
             #      detail="No valid messages found in the provided chat history."
             # )
             return


        # Use the async client and its models module to generate content
        aclient = client.aio
        logger.info(f"Generating content for {subject} with {len(contents)} history entries using model {model_name}.")

        # Stream the response
        response_stream = await aclient.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful assistant that can answer questions about the subject and provide references to the relevant papers. You will be provided context of different question papers of a specficic subject. The user, ie the students will ask you questions about the subject and you will answer them based on the context provided. When the user asks questions about the papers, you will be specific as to what paper and what date the paper is from.",
                thinking_config=types.ThinkingConfig(include_thoughts=True)
            )
        )

        full_response_for_log = ""
        async for chunk in response_stream:
            # Process parts within the chunk
            if chunk.candidates:
                for part in chunk.candidates[0].content.parts:
                    # --- Added Detailed Logging ---
                    logger.info(f"Received Part: {part!r}") # Log the repr of the part object
                    # --- End Added Logging ---

                    if part.thought:
                        # This is a thought part
                        logger.info(f"Model thought: {part.thought}")
                        # Optionally yield this to the client if needed, e.g.:
                        thought_payload = {"type": "thought", "content": part.thought}
                        yield f"data: {json.dumps(thought_payload)}\\n\\n"
                    elif part.text:
                        # This is a regular response part
                        logger.debug(f"Response chunk: {part.text}") # Log debug level if too verbose
                        payload = {"type": "response", "chunk": part.text}
                        json_payload = json.dumps(payload)
                        yield f"data: {json_payload}\n\n"
                        full_response_for_log += part.text
            elif chunk.text:
                 # Fallback for simpler chunks? Less likely with thoughts enabled, but safer.
                 logger.debug(f"Response chunk (direct text): {chunk.text}")
                 payload = {"type": "response", "chunk": chunk.text}
                 json_payload = json.dumps(payload)
                 yield f"data: {json_payload}\n\n"
                 full_response_for_log += chunk.text


        # Log the full response after streaming is complete
        try:
            # Find the last user message in the history for logging
            last_user_message = next((msg.content for msg in reversed(history) if msg.role == 'user'), "N/A")
            await log_to_discord(last_user_message, full_response_for_log)
        except Exception as e:
            logger.error(f"Failed to log to Discord after streaming: {e}")


    except Exception as e:
        logger.error(f"Error generating response for {subject} using model {model_name}: {str(e)}")
        # Yield an error message back to the client via SSE
        content = f"Failed to generate response: {str(e)}"
        payload = {"type": "error", "content": content}
        yield f"data: {json.dumps(payload)}\\n\\n"
        # Optionally raise an exception if you want the server to also log/handle it
        # raise HTTPException(
        #     status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        #     detail=f"Failed to generate response: {str(e)}"
        # )

async def log_to_discord(user_query: str, bot_response: str):    
    log_entry = (
        f"**[{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).strftime('%Y-%m-%d %H:%M:%S %Z')}]**\n"
        f"**User Query:** {user_query}\n"
        f"**Bot Response:** {bot_response}\n"
        "―――――――――――――――――――――――――――"
    )

    async with aiohttp.ClientSession() as session:
        try:
            await session.post(
                webhook_url,
                json={"content": log_entry},
                timeout=aiohttp.ClientTimeout(total=3)
            )
        except Exception as e:
            print(f"Failed to log to Discord: {e}")

@app.get("/status")
async def get_status(): # Renamed for clarity
    try:
        # Simple health check
        return {"status": "OK"}
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        # Raise standard HTTPException for service unavailability
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service health check failed"
        )

# Single dynamic endpoint for all subjects
@app.post("/{subject}")
async def subject_endpoint(subject: str, request_data: ChatRequest):
    try:
        # Use the subject from the validated request data
        current_subject = subject
        logger.info(f"Processing {current_subject} request for SSE. History length: {len(request_data.messages)}")

        # Return a StreamingResponse with the async generator
        return StreamingResponse(
            generate_response(current_subject, request_data.messages),
            media_type="text/event-stream"
        )

    except ValueError as ve: # Catch Pydantic validation errors or other ValueErrors
         logger.error(f"Validation or configuration error for {current_subject}: {str(ve)}")
         # For SSE, we can't easily return an HTTP error *after* starting the stream.
         # Validation happens before streaming starts, so this HTTPException is okay.
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, # Use 400 for validation errors
            detail=f"Invalid request data: {str(ve)}"
         )
    except Exception as e:
        # Catch any other unexpected errors during setup before streaming starts
        logger.error(f"Error in {current_subject} endpoint before streaming: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate {current_subject} request stream: {str(e)}"
        )


if __name__ == "__main__":
    try:
        import uvicorn
        # Ensure API key is set before starting
        if not os.environ.get("GEMINI_API_KEY"):
            logger.critical("GEMINI_API_KEY environment variable not set. Service cannot start.")
            exit(1)  # Exit if API key is missing

        logger.info("Starting pyqchat Service")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.critical(f"Failed to start service: {str(e)}")
        exit(1)  # Exit on any other startup error