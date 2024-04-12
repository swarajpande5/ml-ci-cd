FROM python:3.11

ENV GIT_SSL_NO_VERIFY=true

WORKDIR /app

COPY . .

RUN pip install --upgrade pip \
    && pip install -r --no-cache-dir requirements.txt

RUN python3 code/get_data.py

CMD ["dvc", "repro", "--force"]