# Dockerfile to build custom container for training
# use PyTorch DLVM as base image
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest

WORKDIR /root

# installs transformers, datasets and google-cloud-storage sdk
RUN pip install google-cloud-storage nltk tqdm sacrebleu docopt sentencepiece cloudml-hypertune

# Copies the trainer code to the docker image.
COPY ./run.sh ./run.sh
COPY ./run.py ./run.py
COPY ./vocab.py ./vocab.py
COPY ./download_data.py ./download_data.py
COPY ./src_40k.vocab ./src_40k.vocab
COPY ./src_40k.model ./src_40k.model
COPY ./vocab_40k.json ./vocab_40k.json
COPY ./trainer/__init__.py ./trainer/__init__.py
COPY ./trainer/model_embeddings.py ./trainer/model_embeddings.py
COPY ./trainer/utils.py ./trainer/utils.py
COPY ./trainer/nmt_model.py ./trainer/nmt_model.py


# Set up the entry point to invoke the trainer.
ENTRYPOINT ["bash", "run.sh"]