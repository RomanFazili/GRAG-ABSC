from data_set_ontology import DataSetOntology
from dotenv import load_dotenv
import os
from rdflib import Graph

class OntologyRetriever:

    def __init__(self, data_set_ontology: DataSetOntology):
        self.data_set_ontology: DataSetOntology = data_set_ontology

        self._saved_graphs: dict[str, Graph] = {}

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

    def verbalize_aspect_category_sentiments_restaurant_type_1(self):
        """
        Verbalize the type 1 sentiments of the restaurant ontology.
        Fetches GenericNegativeAction / GenericNegativePropertyMention / GenericPositiveAction / GenericPositivePropertyMention,
        all their descendants, and all superclasses of those classes.
        Uses SPARQL 1.1 queries to fetch the data.

        We use CONSTRUCT to fetch the Graph
        This keeps the :lex as well
        """

        base_ns = "http://www.kimschouten.com/sentiment/restaurant#"
        generic_negative_action_uri = f"{base_ns}GenericNegativeAction"
        generic_negative_property_mention_uri = f"{base_ns}GenericNegativePropertyMention"
        generic_positive_action_uri = f"{base_ns}GenericPositiveAction"
        generic_positive_property_mention_uri = f"{base_ns}GenericPositivePropertyMention"

        g: Graph = self.data_set_ontology.get_rdflib_graph()
        sparql_query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        CONSTRUCT {{
          ?cls ?p ?o .
        }}
        WHERE {{
          VALUES ?seed {{
            <{generic_negative_action_uri}>
            <{generic_negative_property_mention_uri}>
            <{generic_positive_action_uri}>
            <{generic_positive_property_mention_uri}>
          }}
          # Keep classes that are either superclasses or descendants of a seed class.
          {{
            ?seed rdfs:subClassOf* ?cls .
          }}
          UNION
          {{
            ?cls rdfs:subClassOf+ ?seed .
          }}
          ?cls ?p ?o .
        }}
        """
        qres = g.query(sparql_query)
        return qres.graph

    def verbalize_aspect_category_sentiments_restaurant_type_2(self, aspect_category: str) -> Graph:
        """
        Given an aspect category (e.g. AMBIENCE#GENERAL), find the owl:Class with that aspect
        annotation, walk up to the pivot directly under EntityMention, then select classes
        that are strict descendants of that pivot and of Positive/Negative/Neutral.
        """
        g: Graph = self.data_set_ontology.get_rdflib_graph()
        base_ns = "http://www.kimschouten.com/sentiment/restaurant#"
        aspect_uri = f"{base_ns}aspect"
        entity_mention_uri = f"{base_ns}EntityMention"
        positive_uri = f"{base_ns}Positive"
        negative_uri = f"{base_ns}Negative"
        neutral_uri = f"{base_ns}Neutral"

        sparql_query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>

        CONSTRUCT {{
          ?cls ?p ?o .
        }}
        WHERE {{
          {{
            # Find the annotated class.
            ?mention a owl:Class ;
                     <{aspect_uri}> "{aspect_category}" .

            # Ascend from the annotated class to the direct subclass of EntityMention.
            ?mention rdfs:subClassOf* ?pivot .
            ?pivot rdfs:subClassOf <{entity_mention_uri}> .

            # Descendants of that pivot under a polarity root.
            ?cls rdfs:subClassOf+ ?pivot .
            ?cls rdfs:subClassOf+ ?polarityRoot .
            VALUES ?polarityRoot {{
              <{positive_uri}>
              <{negative_uri}>
              <{neutral_uri}>
            }}
          }}
          UNION
          {{
            ?cls a owl:Class ;
                 <{aspect_uri}> "{aspect_category}" .
          }}
          ?cls ?p ?o .
        }}
        """

        qres = g.query(sparql_query)
        return qres.graph

    def verbalize_aspect_category_sentiments_restaurant_type_3(self, aspect_category: str) -> Graph:
        """
        Anonymous intersections under Positive/Negative whose member list contains a class
        annotated with this aspect (or any of its descendants), or a strict superclass of
        an annotated class that remains a strict subclass of EntityMention. Emits the full
        intersection list and triples for every member class.
        """

        base_ns = "http://www.kimschouten.com/sentiment/restaurant#"
        aspect_uri = f"{base_ns}aspect"
        entity_mention_uri = f"{base_ns}EntityMention"
        positive_uri = f"{base_ns}Positive"
        negative_uri = f"{base_ns}Negative"

        g: Graph = self.data_set_ontology.get_rdflib_graph()
        sparql_query = f"""
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl:  <http://www.w3.org/2002/07/owl#>

        CONSTRUCT {{
          # Matched class data
          ?cls ?p ?o .

          # Keep the intersection class statement
          ?anon owl:intersectionOf ?list .
          ?anon rdfs:subClassOf ?polarity .

          # Keep the full RDF list encoding the intersection members
          ?listNode rdf:first ?listItem .
          ?listNode rdf:rest ?next .
        }}
        WHERE {{
          ?anon owl:intersectionOf ?list ;
                rdfs:subClassOf ?polarity .
          VALUES ?polarity {{
            <{positive_uri}>
            <{negative_uri}>
          }}

          # Some list member is in the focus set: an annotated class or any descendant
          # thereof (under EntityMention), or a strict superclass of an annotated class
          # that is still under EntityMention.
          ?list rdf:rest*/rdf:first ?fSel .
          {{
            ?base a owl:Class ;
                  <{aspect_uri}> "{aspect_category}" .
            ?fSel rdfs:subClassOf* ?base .
            ?fSel rdfs:subClassOf+ <{entity_mention_uri}> .
          }}
          UNION
          {{
            ?seed a owl:Class ;
                  <{aspect_uri}> "{aspect_category}" .
            ?seed rdfs:subClassOf+ ?fSel .
            ?fSel rdfs:subClassOf+ <{entity_mention_uri}> .
          }}

          # Whole intersection: every list member and its triples.
          ?list rdf:rest*/rdf:first ?cls .

          ?list rdf:rest* ?listNode .
          ?listNode rdf:first ?listItem ;
                    rdf:rest ?next .

          ?cls ?p ?o .
        }}
        """
        qres = g.query(sparql_query)
        return qres.graph

    def verbalize(self, aspect_category: str) -> Graph:
        """
        Verbalize the sentiments of the aspect category.
        """

        if aspect_category in self._saved_graphs:
            graph: Graph = self._saved_graphs[aspect_category]
            return graph

        type_1_graph: Graph = self.verbalize_aspect_category_sentiments_restaurant_type_1()
        type_2_graph: Graph = self.verbalize_aspect_category_sentiments_restaurant_type_2(aspect_category)
        type_3_graph: Graph = self.verbalize_aspect_category_sentiments_restaurant_type_3(aspect_category)

        graph: Graph = type_1_graph + type_2_graph + type_3_graph

        self._saved_graphs[aspect_category] = graph

        return graph

    def relative_verbalized_graph_size(self, aspect_category: str) -> float:
        """
        Ratio of triple counts: len(verbalize(aspect)) / len(full ontology graph).

        This is the same as the percentage of the full ontology graph that is verbalized.
        """

        verbalized: Graph = self.verbalize(aspect_category)
        full: Graph = self.data_set_ontology.get_rdflib_graph()

        return len(verbalized) / len(full)

if __name__ == "__main__":
    load_dotenv()

    domain = 'laptop'
    domain = 'restaurant'

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

        reachable_graph: Graph = ontology_retriever.verbalize('FOOD#QUALITY')
        reachable_graph: Graph = ontology_retriever.verbalize_aspect_category_sentiments_restaurant_type_3('FOOD#QUALITY')
        reachable_graph.serialize(destination="reachable_subgraph.owl", format="xml")

        print(ontology_retriever.relative_verbalized_graph_size("FOOD#QUALITY"))