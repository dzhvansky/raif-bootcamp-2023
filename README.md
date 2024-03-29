[![build](https://github.com/dzhvansky/raif-bootcamp-2023/actions/workflows/fly.yml/badge.svg)](https://github.com/dzhvansky/raif-bootcamp-2023/actions/workflows/fly.yml) [![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)

# Raiffeisen bootcamp 2023 - Telegram bot for painting value estimation

Deployed API (docs): https://velvet-wolves-art-expert-api.fly.dev/docs

Service dashboard: https://fly-metrics.net/d/_eX4mpl3/bootcamp-dashboard?orgId=205759&refresh=5s

# Quickstart

Run `poetry install`

Then for API:
1. run `poetry run ./bin/scripts/run-api-dev.sh` for development with autoreload or `poetry run ./bin/app/run-api.sh` for production
1. open http://localhost:8000/docs — FastAPI autogenerated documentation

for BOT:
1. run `TELEGRAM_TOKEN=<your-token-here> poetry run ./bin/app/run-bot.sh`

# Docker

1. Install docker and docker-compose (if not yet)
1. `export TELEGRAM_TOKEN=<your-token-here>`
1. run `docker-compose build`
1. run `docker-compose up`
1. open http://localhost:8000/docs — FastAPI autogenerated documentation
1. bot is also run

# Fly.io

To deploy manually:
1. install flyctl: `brew install flyctl`
2. create apps: `flyctl apps create <velvet-wolves-art-expert-api or velvet-wolves-art-expert-bot>`
3. increase API app memory: `flyctl scale memory 1024 -a velvet-wolves-art-expert-api`
4. set telegram token for BOT app: `fly secrets set TELEGRAM_TOKEN=<your-token-here> -a velvet-wolves-art-expert-bot`
5. run `fly deploy --config <fly.api.toml or fly.bot.toml>`
6. generate deploy token `flyctl tokens create deploy -a <velvet-wolves-art-expert-api or velvet-wolves-art-expert-bot>` and put it in Github action secrets

or tag a commit with some version in format `v*.*.*` (for ex. `v1.0.1`) and let CI/CD do its work:
1. `git tag v1.0.1`
2. `git push --tags`

or force deployment manually:
1. go to https://github.com/dzhvansky/raif-bootcamp-2023/actions/workflows/fly.yml
2. Press "Run workflow" button and actually run it


# Load testing

Run `poetry run locust -f painting_estimation/load_test.py` and open browser with suggested link

# Memory profiling

1. Run `poetry run mpref run pytest tests/test_api.py`
2. run `poetry run mpref plot -o mem.py --title "Prediction API RAM consumption"`