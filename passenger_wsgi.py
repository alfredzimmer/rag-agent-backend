
import sys
import os

# --- START OF THE FIX ---
# This is the path to your new venv's libraries.
# We are manually adding it to the list of paths Python searches.
VENV_PATH = "/home/ziyutecc/public_html/pyapi/venv_py39/lib/python3.9/site-packages"
if VENV_PATH not in sys.path:
    sys.path.insert(0, VENV_PATH)
# --- END OF THE FIX ---

# Import the new adapter
from a2wsgi import ASGIMiddleware

# Import your FastAPI app
from main import app

# This is the correct way to wrap your ASGI app for a WSGI server
# Passenger will now talk to 'application', which translates
# the request for your FastAPI 'app'.
application = ASGIMiddleware(app)