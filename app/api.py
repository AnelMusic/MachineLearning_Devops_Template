# app/api.py
# FastAPI application endpoints.


from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import Dict, Optional

from fastapi import FastAPI, Request

from app.config import logger
from titanic_classification import utils

# Define application
app = FastAPI(
    title="Titanic Classification Api",
    description="Predict surival based on passenger data",
    version="0.1",
)


@app.on_event("startup")
def load_artifacts():
    logger.info("Startup done")


# Decorator for additional information
def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


@app.get("/")
def index():
    """Health check."""
    response = {
        "message": "Use: URL/docs to access API documentation",
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.get("/model_params")
@construct_response
def _params(request: Request) -> Dict:
    """Get parameter values used for a run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "model_params": utils.load_model_params(),
        },
    }
    return response


@app.get(
    "/performance",
)
@construct_response
def _performance(request: Request, filter: Optional[str] = None) -> Dict:
    """Get best model performance"""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": utils.load_model_performance(),
    }
    return response
