FROM python:3.12-slim

WORKDIR /app

RUN pip install pipenv

COPY ["Pipfile","Pipfile.lock","./"]

RUN pipenv install --system --deploy --categories gateway-app

COPY ["app/gateway_app.py","./"]

EXPOSE 80

ENTRYPOINT ["waitress-serve","--listen=0.0.0.0:80","gateway_app:app"]