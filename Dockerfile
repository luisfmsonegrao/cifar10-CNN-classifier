FROM python:3.12-slim

WORKDIR /app

ENV HOST_NAME = "REMOTE CONTAINER"

RUN pip install pipenv

COPY ["Pipfile","Pipfile.lock","./"]

RUN pipenv install --system --deploy

COPY ["src/predict.py","./"]
COPY ["models/cifar_model_v1.bin","./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve","--listen=0.0.0.0:9696","predict:app"]