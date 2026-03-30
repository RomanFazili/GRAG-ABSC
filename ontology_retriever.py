from data_set_ontology import DataSetOntology
from dotenv import load_dotenv
import os
from rdflib import Graph


class OntologyRetriever:

    def __init__(self, data_set_ontology: DataSetOntology):
        self.data_set_ontology: DataSetOntology = data_set_ontology

    def verbalize_type1_sentiments_laptop(self):
        """
        Verbalize the type 1 sentiments of the laptop ontology.
        Returns an owlready2 Ontology object with the verbalized type 1 sentiments.
        Fetches GenericPositiveSentiment / GenericNegativeSentiment / GenericNeutralSentiment,
        all their descendants, and all superclasses of those classes.
        Uses SPARQL 1.1 queries to fetch the data.

        We use CONSTRUCT to fetch the Graph
        This keeps the :lex as well
        """

        g: Graph = self.data_set_ontology.get_rdflib_graph()
        sparql_query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl:  <http://www.w3.org/2002/07/owl#>

        CONSTRUCT {
          # Build a graph with triples about direct children only.
          ?cls ?p ?o .
        }
        WHERE {
          {
            SELECT DISTINCT ?cls WHERE {
              VALUES ?targetLocalName {
                "GenericPositiveSentiment"
                "GenericNegativeSentiment"
                "GenericNeutralSentiment"
              }

              # Locate exactly those root classes by local name (fragment or last path segment).
              ?root a owl:Class .
              BIND(REPLACE(STR(?root), "^.*[#/]", "") AS ?rootLocalName)
              FILTER(?rootLocalName = ?targetLocalName)

              # Include roots and all descendants.
              {
                ?cls rdfs:subClassOf* ?root .
              }
              UNION
              # Include all superclasses of roots/descendants to preserve hierarchy.
              {
                ?member rdfs:subClassOf* ?root .
                ?member rdfs:subClassOf* ?cls .
              }
            }
          }

          # Pull all outgoing triples of selected classes.
          ?cls ?p ?o .
        }
        """
        qres = g.query(sparql_query)
        return qres.graph


if __name__ == "__main__":
    load_dotenv()

    domain = 'restaurant'
    domain = 'laptop'

    if domain == 'laptop':
        file_path = os.getenv("PATH_TO_LAPTOP_ONTOLOGY")
        data_set_ontology = DataSetOntology(file_path)
        ontology_retriever = OntologyRetriever(data_set_ontology)

        reachable_graph: Graph = ontology_retriever.verbalize_type1_sentiments_laptop()

        # Serialize result as proper RDF/XML
        reachable_graph.serialize(destination="reachable_subgraph.owl", format="xml")


    if domain == 'restaurant':
        file_path = os.getenv("PATH_TO_RESTAURANT_ONTOLOGY")
        data_set_ontology = DataSetOntology(file_path)
        ontology_retriever = OntologyRetriever(data_set_ontology)