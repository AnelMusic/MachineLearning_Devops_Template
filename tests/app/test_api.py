#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 23:08:19 2021

@author: anelmusic
"""


from http import HTTPStatus

from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_construct_response():
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
    assert response.request.method == "GET"


def test_index():
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
    assert response.json()["message"] == "Use: URL/docs to access API documentation"


def test_params():
    with TestClient(app) as client:
        response = client.get("/model_params")
        assert response.status_code == HTTPStatus.OK
        assert response.request.method == "GET"
        print(response.json())
        assert isinstance(response.json()["data"]["model_params"]["params"], dict)
        assert isinstance(response.json()["data"]["model_params"]["run_id"], str)


def test_performance():
    with TestClient(app) as client:
        response = client.get("/performance")
        assert response.status_code == HTTPStatus.OK
        assert response.request.method == "GET"
        print(response.json())
        assert isinstance(response.json()["data"]["overall"]["precision"], float)
        assert isinstance(response.json()["data"]["overall"]["recall"], float)
        assert isinstance(response.json()["data"]["overall"]["f1"], float)
