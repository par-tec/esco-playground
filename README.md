# ESCO Playground

This repository contains the code for the ESCO Playground.
The jupyter notebook should work without the ESCO dataset,
since an excerpt of the dataset is already included in `esco.json.gz`.

To regenerate the NER model, you need the ESCO dataset in turtle format.

:warning: before using this repository, you need to:

1. download the ESCO 1.1.1 database in text/turtle format  `ESCO dataset - v1.1.1 - classification -  - ttl.zip` from the [ESCO portal](https://ec.europa.eu/esco/portal) and unzip the `.ttl` file under the`vocabularies` folder.

2. execute the sparql server that will be used to serve the ESCO dataset,
   and wait for the server to spin up and load the ~700MB dataset.
   This can be done with the following command:

   ```bash
   docker-compose up -d virtuoso
   ```

3. run the tests

   ```bash
   tox -e py3
   ```

4. run the API

   ```bash
   connexion run api/openapi.yaml &
   xdg-open http://localhost:5000/esco/v0.0.1/ui/
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
