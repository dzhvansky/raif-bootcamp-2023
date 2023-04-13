#!/usr/bin/env bash

uvicorn painting_estimation.api.main:APP --host 0.0.0.0 --port 8000 --workers 4 --reload