from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import logging
from typing import Any, Dict, List
import os
from google import genai
from google.genai import types
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Agent Chatbot Service")

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
    # attachments: list = Field(default_factory=list) # Add if needed later

# Pydantic model for the incoming chat request body
class ChatRequest(BaseModel):
    message: str # The latest message from the user
    subject: str
    messages: List[ChatMessage] # The history of the conversation

    @validator('subject')
    def subject_must_be_allowed(cls, v):
        if v.upper() not in ALLOWED_SUBJECTS:
            raise ValueError(f"Subject '{v}' is not allowed. Allowed subjects are: {', '.join(ALLOWED_SUBJECTS)}")
        return v

# Create a Gemini client
def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        raise ValueError("GEMINI_API_KEY environment variable not set")
    # Use environment variables, Client() will pick it up automatically
    # genai.configure(api_key=api_key) # Configure the library with the API key
    # return genai # Return the configured library module
    # According to docs, Client() reads GOOGLE_API_KEY env var
    client = genai.Client(api_key=api_key)
    return client

# Cache PDF files for each subject
subject_files_cache = {}

# Get PDF files for a subject with caching
@lru_cache(maxsize=None) # Use lru_cache for simplicity
def get_pdf_files(subject):
    logger.info(f"Loading PDF files for {subject}")
    files = []
    # client_module = get_client(subject) # Get the configured genai module
    client = get_client() # Get the client instance
    subject_dir = f"v1/{subject}"

    try:
        if not os.path.isdir(subject_dir):
             logger.warning(f"Directory not found for subject: {subject}")
             return [] # Return empty list if directory doesn't exist

        for filename in os.listdir(subject_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(subject_dir, filename) # Use os.path.join for cross-platform compatibility
                logger.info(f"Uploading file: {file_path}")
                # Use client.files.upload with updated API structure
                uploaded_file = client.files.upload(file=file_path)
                files.append(uploaded_file)

        # subject_files_cache[subject] = files # Caching handled by lru_cache
        return files
    except Exception as e:
        logger.error(f"Error loading PDF files for {subject}: {str(e)}")
        raise # Re-raise the exception after logging

# Generate function for subject-specific responses, now accepting history
async def generate_response(subject: str, history: List[ChatMessage]):
    # client_module = get_client(subject) # Get the configured genai module
    client = get_client() # Get the client instance
    files = get_pdf_files(subject) # This will use the cached result if available

    if not files:
        logger.warning(f"No PDF files found for subject: {subject}")
        # Return a more informative message if no files are found
        return f"No reference materials (PDFs) found for the subject '{subject}'. Cannot generate response."

    model_name = "models/gemini-2.0-flash" # Using a new model, more reliable responses and better support.

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
             raise HTTPException(
                  status_code=status.HTTP_400_BAD_REQUEST,
                  detail="No valid messages found in the provided chat history."
             )

        # Get the model instance from the client
        # model = client_module.GenerativeModel(model_name=model_name)
        # model = client.get_model(model_name=model_name) # Incorrect: Client has no get_model

        # Use the async client and its models module to generate content
        aclient = client.aio
        logger.info(f"Generating content for {subject} with {len(contents)} history entries using model {model_name}.")
        # response = await model.generate_content_async(contents) # Old way using GenerativeModel instance
        response = await aclient.models.generate_content(
            model=model_name,
            system_instruction="You are a helpful assistant that can answer questions about the subject and provide references to the relevant papers. You will be provided context of different question papers of a specficic subject. The user, ie the students will ask you questions about the subject and you will answer them based on the context provided.",
            contents=contents # Pass the constructed contents
        )

        return response.text # Access the text part of the response
    except Exception as e:
        logger.error(f"Error generating response for {subject} using model {model_name}: {str(e)}")
        # Consider returning a more specific error or re-raising
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )

@app.get("/status")
async def get_status(): # Renamed for clarity
    try:
        # Simple health check
        return {"status": "OK", "service": "AI Agent Chatbot", "version": "1.0.0"}
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        # Raise standard HTTPException for service unavailability
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service health check failed"
        )

# Single dynamic endpoint for all subjects
@app.post("/{subject}")
async def subject_endpoint(subject: str, request_data: ChatRequest): # Changed model to ChatRequest
    # Validate subject consistency
    path_subject_upper = subject.upper()
    body_subject_upper = request_data.subject.upper() # Already validated by Pydantic

    if path_subject_upper != body_subject_upper:
         logger.warning(f"Path subject '{path_subject_upper}' does not match body subject '{body_subject_upper}'.")
         raise HTTPException(
             status_code=status.HTTP_400_BAD_REQUEST,
             detail=f"Path subject '{path_subject_upper}' does not match subject '{body_subject_upper}' in request body."
         )

    # Check against allowed subjects (redundant due to Pydantic validator, but safe)
    if path_subject_upper not in ALLOWED_SUBJECTS:
        logger.warning(f"Invalid subject requested via path: {subject}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, # Keep 404 for path-based lookup failure
            detail=f"Subject '{subject}' not found. Allowed subjects are: {', '.join(ALLOWED_SUBJECTS)}"
        )

    try:
        # Use the subject from the validated request data
        current_subject = body_subject_upper
        logger.info(f"Processing {current_subject} request. History length: {len(request_data.messages)}")
        # We now need to pass the history to generate_response
        # The 'message' field in ChatRequest is the latest user message, already part of 'messages' history
        response_text = await generate_response(current_subject, request_data.messages) # Pass history
        return {"message": f"Success from {current_subject}!", "response": response_text}
    except HTTPException as he:
        # Re-raise HTTPExceptions directly (e.g., from generate_response)
        raise he
    except ValueError as ve: # Catch Pydantic validation errors or other ValueErrors
         logger.error(f"Validation or configuration error for {current_subject}: {str(ve)}")
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, # Use 400 for validation errors
            detail=f"Invalid request data: {str(ve)}"
         )
    except Exception as e:
        # Catch any other unexpected errors during processing
        logger.error(f"Error in {current_subject} endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process {current_subject} request: {str(e)}"
        )


if __name__ == "__main__":
    try:
        import uvicorn
        # Ensure API key is set before starting
        if not os.environ.get("GEMINI_API_KEY"):
            logger.critical("GEMINI_API_KEY environment variable not set. Service cannot start.")
            exit(1)  # Exit if API key is missing

        logger.info("Starting AI Agent Chatbot Service")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.critical(f"Failed to start service: {str(e)}")
        exit(1)  # Exit on any other startup error