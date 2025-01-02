FROM python:3.12-slim

WORKDIR /app

ENV HOST_NAME="REMOTE CONTAINER"

RUN pip install pipenv

COPY ["Pipfile","Pipfile.lock","./"]

RUN pipenv install --system --deploy --categories flask-app

COPY ["app/cifar10_app.py","./"]
COPY ["trained_models/cifar_model_cpu_v1.bin","./"]
COPY ["models/SmallCNN.py","models/__init__.py","./models/"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve","--listen=0.0.0.0:9696","cifar10_app:app"]