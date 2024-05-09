# ESCO Playground

The ESCO Playground is a repository to play with the ESCO dataset,
and to test different approaches to extract skills from text.

:warning: This is a work in progress, and it is not ready for production.

## Installation

To install the package, you can use pip:

```bash
pip install git+https://github.com/par-tec/esco-playground
```

Optional dependencies are:

```bash
pip install esco[langchain]
pip install esco[dev]
```

## Usage

The simplest way to use this module is via the `LocalDB` class,
that wraps the ESCO dataset embedded in the package via the [json files](esco/esco.json.gz):

```python
from esco import LocalDB

esco_data = LocalDB()

# Get a skill by its cURIe.
skill = esco_data.get("esco:b0096dc5-2e2d-4bc1-8172-05bf486c3968")

# Search a list of skill using labels.
skills = esco_data.search_products({"python", "java"})

# Further queries can be done using the embedded dataframe.
esco_data.skills.__class__ == pandas.core.frame.DataFrame

esco_data.skills[esco_data.skills.label == "SQL Server"]
```

To use extra features such as text to skill extraction
you need to install the optional dependencies
(which are really slow if you don't have a GPU).

```bash
pip install esco[langchain]
```

Use the `EscoCV` and the `Ner` classes to extract skills from text:

```python
from esco.cv import EscoCV
from esco import LocalDB
from esco.ner import Ner

# Initialize the vector index (slow) on disk.
# This can be reused later.
datadir = Path("/tmp/esco-tmpdir")
datadir.mkdir(exist_ok=True)
cfg = {
      "path": datadir / "esco-skills",
      "collection_name": "esco-skills",
   }
db = LocalDB()
db.create_vector_idx(cfg)
db.close()

# Now you can create a new db that loads the vector index.
db = LocalDB(vector_idx_config=cfg)

# and a recognizer class that used both the ESCO dataset and the vector index.
cv_recognizer = Ner(db=db, tokenizer=nltk.sent_tokenize)

# Now you can use the recognizer to extract skills from text.
cv_text = """I am a software developer with 5 years of experience in Python and Java."""
cv = cv_recognizer(text)

# This will take some time.
cv_skills = cv.skills()
```

If you have a sparql server with the ESCO dataset, you can use the `SparqlClient`:

```python
from esco.sparql import SparqlClient

client = SparqlClient("http://localhost:8890/sparql")

skills_df = client.load_skills()

occupations_df = client.load_occupations()

# You can even use custom queries returning a CSV.
query = """SELECT ?skill ?label
WHERE {
    ?skill a esco:Skill .
    ?skill skos:prefLabel ?label .
    FILTER (lang(?label) = 'en')
}"""
skills = client.query(query)
```

## Development

The jupyter notebook should work without the ESCO dataset,
since an excerpt of the dataset is already included in `esco.json.gz`.

To regenerate the NER model, you need the ESCO dataset in turtle format.

:warning: before using this repository, you need to:

1. download the ESCO 1.1.1 database in text/turtle format
`ESCO dataset - v1.1.1 - classification -  - ttl.zip`
from the [ESCO portal](https://ec.europa.eu/esco/portal)
and unzip the `.ttl` file under the [`vocabularies`](vocabularies/) folder.

1. execute the sparql server that will be used to serve the ESCO dataset,
   and wait for the server to spin up and load the ~700MB dataset.
   :warning: It will take a couple of minutes,
   so you need to wait for the server to be ready.

   ```bash
   docker-compose up -d virtuoso
   ```

1. run the tests

   ```bash
   tox -e py3
   ```

## Regenerate the model

To regenerate the model, you need to setup the ESCO dataset as explained above
and then run the following command:

```bash
tox -e model   
```

To build and upload the model, provided you did `huggingface-cli login`:

```bash
tox -e build-model -- upload
```

```bash

## Contributing

Please, see [CONTRIBUTING.md](CONTRIBUTING.md) for more details on:

- using [pre-commit](CONTRIBUTING.md#pre-commit);
- following the git flow and making good [pull requests](CONTRIBUTING.md#making-a-pr).

## Using this repository

You can create new projects starting from this repository,
so you can use a consistent CI and checks for different projects.

Besides all the explanations in the [CONTRIBUTING.md](CONTRIBUTING.md) file,
you can use the docker-compose file
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
