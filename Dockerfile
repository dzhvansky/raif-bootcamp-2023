FROM python:3.10 as compile-image

ARG EXECUTABLE="run-api.sh"

RUN groupadd --gid 2000 python
RUN useradd --uid 2000 --gid python --shell /usr/sbin/nologin --create-home python

RUN apt-get update

COPY pyproject* ./
# Uncomment before review
# COPY poetry.lock ./

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install

RUN apt-get remove -y gcc cmake make
RUN rm -rf /var/lib/apt/lists/* && apt-get autoremove -y && apt-get clean
RUN pip uninstall pipenv poetry -y

FROM scratch AS runtime-image

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    LANG="C.UTF-8" \
    PYTHON_VERSION=3.10.8 \
    PYTHONUNBUFFERED=1 \
    WORKDIR=/srv/www/
WORKDIR $WORKDIR

COPY --from=compile-image / /

COPY . .

RUN chown python:python /srv -R
EXPOSE 8000
USER python:python
CMD ["/srv/www/bin/app/${EXECUTABLE}"]