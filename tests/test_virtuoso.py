import pytest
from rdflib import Graph, URIRef
from rdflib.plugins.stores import sparqlstore

query_endpoint = "http://virtuoso:8890/sparql"
update_endpoint = "http://virtuoso:8890/sparql-auth"


@pytest.mark.skip(reason="For development only")
def test_add_rdf():
    store = sparqlstore.SPARQLUpdateStore(
        query_endpoint,
        update_endpoint,
        method="POST",
        autocommit=False,
        auth=("dba", "dba"),
    )

    g = Graph(store, identifier=URIRef("http://example.org/"))
    g.add(
        (
            URIRef("http://example.org/subject"),
            URIRef("http://example.org/predicate"),
            URIRef("http://example.org/object"),
        )
    )
    store.commit()

    raise NotImplementedError
