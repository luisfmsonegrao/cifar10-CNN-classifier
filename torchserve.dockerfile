FROM pytorch/torchserve:latest
RUN pip install torchserve torch-model-archiver torch-workflow-archiver
COPY model-store/cifar10-classifier.mar ./model-store
CMD ["torchserve", "--start", "--disable-token-auth", "--ncs", "--model-store ./model-store", "--models cifar10-classifier=cifar10-classifier.mar"]