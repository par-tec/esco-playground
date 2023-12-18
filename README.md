# ESCO Playground

This repository contains the code for the ESCO Playground.
The jupyter notebook should work without the ESCO dataset,
since an excerpt of the dataset is already included in `esco.json.gz`.

To regenerate the NER model, you need the ESCO dataset in turtle format.

:warning: before using this repository, you need to:

1. download the ESCO 1.1.1 database in text/turtle format  `ESCO dataset - v1.1.1 - classification -  - ttl.zip` from the [ESCO portal](https://ec.europa.eu/esco/portal) and unzip the `.ttl` file under the`vocabularies` folder.

1. execute the sparql server that will be used to serve the ESCO dataset,
   and wait for the server to spin up and load the ~700MB dataset.
   :warning: It will take a couple of minutes, so you need to wait for the server to be ready.

   ```bash
   docker-compose up -d virtuoso
   ```

1. run the tests

   ```bash
   tox -e py3
   ```

1. run the API

   ```bash
   connexion run api/openapi.yaml &
   xdg-open http://localhost:5000/esco/v0.0.1/ui/
   ```

## Regenerate the model

To regenerate the model, you need to setup the ESCO dataset as explained above
and then run the following commands:

```bash
rm ./generated/output/ -fr
mkdir -p generated/output
pip install -r requirements-dev.txt
python model/model.py
python -m spacy package ./generated/en_core_web_trf_esco_ner ./generated/output --build wheel
(
   cd huggingface-hub push generated/output/en_core_web_trf_esco_ner*/dist/;
   python -m spacy en_core_web_trf_esco_ner*.whl
)
```

## Contributing

Please, see [CONTRIBUTING.md](CONTRIBUTING.md) for more details on:

- using [pre-commit](CONTRIBUTING.md#pre-commit);
- following the git flow and making good [pull requests](CONTRIBUTING.md#making-a-pr).

## Using this repository

You can create new projects starting from this repository,
so you can use a consistent CI and checks for different projects.

Besides all the explanations in the [CONTRIBUTING.md](CONTRIBUTING.md) file, you can use the docker-compose file
(e.g. if you prefer to use docker instead of installing the tools locally)

```bash
docker-compose run pre-commit
```


## Using on GCP

If you need a GPU server, you can

1. create a new GPU machine using the pre-built `debian-11-py310` image.
   The command is roughly the following

   ```bash
   gcloud compute instances create instance-2 \
      --machine-type=n1-standard-4 \
      --create-disk=auto-delete=yes,boot=yes,device-name=instance-1,image=projects/ml-images/global/images/c0-deeplearning-common-gpu-v20231209-debian-11-py310,mode=rw,size=80,type=projects/${PROJECT}/zones/europe-west1-b/diskTypes/pd-standard \
      --no-restart-on-failure \
      --maintenance-policy=TERMINATE \
      --provisioning-model=STANDARD \
      --accelerator=count=1,type=nvidia-tesla-t4 \
      --no-shielded-secure-boot \
      --shielded-vtpm \
      --shielded-integrity-monitoring \
      --labels=goog-ec-src=vm_add-gcloud \
      --reservation-affinity=any \
      --zone=europe-west1-b \
      ...

   ```

2. access the machine and finalize the CUDA installation. Rember to enable port-forwarding for the jupyter notebook

   ```bash
   gcloud compute ssh --zone "europe-west1-b" "deleteme-gpu-1" --project "esco-test" -- -NL 8081:localhost:8081

   ```

3. checkout the project and install the requirements

   ```bash
   git clone https://github.com/par-tec/esco-playground.git
   cd esco-playground
   pip install -r requirements-dev.txt -r requirements.txt
   ```
