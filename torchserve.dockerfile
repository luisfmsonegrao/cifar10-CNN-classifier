FROM pytorch/torchserve:latest
RUN pip install pipenv
COPY ["Pipfile","Pipfile.lock","./"]
RUN pipenv install --system --deploy --categories torchserve-app
COPY model-store/cifar10-classifier-v2.mar ./model-store
CMD ["torchserve", "--start", "--disable-token-auth", "--ncs", "--model-store ./model-store", "--models cifar10-classifier=cifar10-classifier-v2.mar"]