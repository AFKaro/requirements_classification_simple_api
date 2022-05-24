import logging
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()


def api_storage_requirements(data: dict):
    endpoint = os.getenv("SOFTLAB_API_STORAGE_REQUIREMENTS")
    data = json.dumps(data)
    response = requests.post(endpoint, json=data)
    return response
