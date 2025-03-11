from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
import uvicorn
import logging
from typing import Any, Dict
import json

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

# Custom exception handler for all exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )

# Exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation error", "detail": str(exc)},
    )

# Helper function to safely parse request body
async def parse_request_body(request: Request) -> Dict[str, Any]:
    try:
        body = await request.body()
        if not body:
            return {}
        
        text = body.decode("utf-8")
        if not text:
            return {}
            
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in request body")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON in request body"
        )
    except Exception as e:
        logger.error(f"Error parsing request body: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error parsing request body: {str(e)}"
        )

@app.get("/status")
async def status():
    try:
        # Simulate checking some dependencies that might fail
        # In a real app, you might check database connections, etc.
        return {"status": "OK", "service": "AI Agent Chatbot", "version": "1.0.0"}
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service health check failed"
        )

@app.post("/subject1")
async def subject1(request: Request):
    try:
        data = await parse_request_body(request)
        logger.info(f"Processing subject1 request with data: {data}")
        # Process the request (placeholder for your actual logic)
        return {"message": "Success from subject1 call!", "received_data": data}
    except HTTPException:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise
    except Exception as e:
        logger.error(f"Error in subject1: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process subject1 request: {str(e)}"
        )

@app.post("/subject2")
async def subject2(request: Request):
    try:
        data = await parse_request_body(request)
        logger.info(f"Processing subject2 request with data: {data}")
        return {"message": "Success from subject2 call!", "received_data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in subject2: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process subject2 request: {str(e)}"
        )

@app.post("/subject3")
async def subject3(request: Request):
    try:
        data = await parse_request_body(request)
        logger.info(f"Processing subject3 request with data: {data}")
        return {"message": "Success from subject3 call!", "received_data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in subject3: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process subject3 request: {str(e)}"
        )

@app.post("/subject4")
async def subject4(request: Request):
    try:
        data = await parse_request_body(request)
        logger.info(f"Processing subject4 request with data: {data}")
        return {"message": "Success from subject4 call!", "received_data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in subject4: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process subject4 request: {str(e)}"
        )

@app.post("/subject5")
async def subject5(request: Request):
    try:
        data = await parse_request_body(request)
        logger.info(f"Processing subject5 request with data: {data}")
        return {"message": "Success from subject5 call!", "received_data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in subject5: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process subject5 request: {str(e)}"
        )

@app.post("/subject6")
async def subject6(request: Request):
    try:
        data = await parse_request_body(request)
        logger.info(f"Processing subject6 request with data: {data}")
        return {"message": "Success from subject6 call!", "received_data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in subject6: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process subject6 request: {str(e)}"
        )

@app.post("/subject7")
async def subject7(request: Request):
    try:
        data = await parse_request_body(request)
        logger.info(f"Processing subject7 request with data: {data}")
        return {"message": "Success from subject7 call!", "received_data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in subject7: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process subject7 request: {str(e)}"
        )

@app.post("/subject8")
async def subject8(request: Request):
    try:
        data = await parse_request_body(request)
        logger.info(f"Processing subject8 request with data: {data}")
        return {"message": "Success from subject8 call!", "received_data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in subject8: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process subject8 request: {str(e)}"
        )

if __name__ == "__main__":
    try:
        logger.info("Starting AI Agent Chatbot Service")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.critical(f"Failed to start service: {str(e)}")