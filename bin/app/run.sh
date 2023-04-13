#!/usr/bin/env bash

gunicorn painting_estimation.api.main:APP --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:80