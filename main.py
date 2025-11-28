from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from fastapi.middleware.cors import CORSMiddleware

# Initialize the app
app = FastAPI()

origins = [
    "https://pis3.aempro.ca",  # Your main Node app
    "http://localhost:3000", # Good to add for local testing
]

# 4. ADD THE MIDDLEWARE TO YOUR APP
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"], # Allow both GET and POST
    allow_headers=["*"],
)

# Define the root endpoint (health check)
@app.get("/")
def read_root():
    """
    This is the health check endpoint.
    If you can see this, the API is working.
    """
    return {"status": "ok", "message": "Python API is running!"}

@app.get("/health")
def check_health():
    """
    A simple GET endpoint to confirm the API is live
    and reachable.
    """
    return {"status": "ok", "message": "Python API is healthy!"}
