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

    @staticmethod
    def _aspect_to_mention_local_name(aspect_category: str) -> str:
        """
        Convert aspect category to mention class local name.
        Example: FOOD#STYLE_OPTIONS -> FoodMention
        """

        first_part = aspect_category.split("#", 1)[0].strip().capitalize()
        return f'{first_part}Mention'

    def verbalize_aspect_category_sentiments_restaurant_type_2_old(self, aspect_category: str) -> Graph:
        """
        Given an aspect category (e.g. FOOD#STYLE_OPTIONS), find the corresponding
        mention class (FoodMention), climb to the ancestor whose direct parent is
        EntityMention, then select classes that are descendants of that ancestor
        and also descendants of Positive/Negative/Neutral.
        """

        raise NotImplementedError("This method is deprecated, use verbalize_aspect_category_sentiments_restaurant_type_2 instead.")

        mention_local_name = self._aspect_to_mention_local_name(aspect_category)
        g: Graph = self.data_set_ontology.get_rdflib_graph()
        base_ns = "http://www.kimschouten.com/sentiment/restaurant#"
        mention_uri = f"{base_ns}{mention_local_name}"
        entity_mention_uri = f"{base_ns}EntityMention"
        positive_uri = f"{base_ns}Positive"
        negative_uri = f"{base_ns}Negative"
        neutral_uri = f"{base_ns}Neutral"

        sparql_query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        CONSTRUCT {{
          ?cls ?p ?o .
        }}
        WHERE {{
          # Find the ancestor on the mention path whose direct parent is EntityMention.
          <{mention_uri}> rdfs:subClassOf* ?pivot .
          ?pivot rdfs:subClassOf <{entity_mention_uri}> .

          # Keep classes under that pivot...
          ?cls rdfs:subClassOf+ ?pivot .
          # ...and also descendants of Positive OR Negative OR Neutral.
          ?cls rdfs:subClassOf+ ?polarityRoot .
          VALUES ?polarityRoot {{
            <{positive_uri}>
            <{negative_uri}>
            <{neutral_uri}>
          }}
          ?cls ?p ?o .
        }}
        """

        qres = g.query(sparql_query)
        return qres.graph

    def verbalize_aspect_category_sentiments_restaurant_type_3(self, aspect_category: str) -> Graph:
        """
        Verbalize the type 3 sentiments of the restaurant ontology.
        Fetches GenericNegativeAction / GenericNegativePropertyMention / GenericPositiveAction / GenericPositivePropertyMention,
        all their descendants, and all superclasses of those classes.
        Uses SPARQL 1.1 queries to fetch the data.

        We use CONSTRUCT to fetch the Graph
        This keeps the :lex as well
        """

        mention_local_name = self._aspect_to_mention_local_name(aspect_category)
        base_ns = "http://www.kimschouten.com/sentiment/restaurant#"
        mention_uri = f"{base_ns}{mention_local_name}"
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
          # Find anonymous intersection classes that are subclasses of Positive or Negative.
          ?anon owl:intersectionOf ?list ;
                rdfs:subClassOf ?polarity .
          VALUES ?polarity {{
            <{positive_uri}>
            <{negative_uri}>
          }}

          # Intersection contains mention class.
          ?list rdf:rest*/rdf:first <{mention_uri}> .

          # Intersection also contains candidate class ?cls.
          ?list rdf:rest*/rdf:first ?cls .
          FILTER(?cls != <{mention_uri}>)

          # Materialize complete intersection list in output.
          ?list rdf:rest* ?listNode .
          ?listNode rdf:first ?listItem ;
                    rdf:rest ?next .

          # Keep outgoing triples of matching classes.
          ?cls ?p ?o .
        }}
        """
        qres = g.query(sparql_query)
        return qres.graph

    def verbalize_aspect_category_sentiments_restaurant_type_2(self, aspect_category: str) -> Graph:
        """
        Given an aspect category (e.g. AMBIENCE#GENERAL), find the mention class that has
        the exact aspect annotation, then select classes that are descendants of that mention
        and also descendants of Positive/Negative/Neutral.
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
            ?mention a owl:Class ;
                     <{aspect_uri}> "{aspect_category}" .
            # Keep classes that are descendants of the mention...
            ?cls rdfs:subClassOf+ ?mention .
            # ...and also descendants of Positive OR Negative OR Neutral.
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

    def verbalize(self, aspect_category: str) -> Graph:
        """
        Verbalize the sentiments of the aspect category.
        """

        type_1_graph: Graph = self.verbalize_aspect_category_sentiments_restaurant_type_1()
        type_2_graph: Graph = self.verbalize_aspect_category_sentiments_restaurant_type_2(aspect_category)
        type_3_graph: Graph = self.verbalize_aspect_category_sentiments_restaurant_type_3(aspect_category)

        return type_1_graph + type_2_graph + type_3_graph

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

        reachable_graph: Graph = ontology_retriever.verbalize_aspect_category_sentiments_restaurant_type_2('FOOD#QUALITY')
        reachable_graph: Graph = ontology_retriever.verbalize('FOOD#QUALITY')
        reachable_graph.serialize(destination="reachable_subgraph.owl", format="xml")

        print(ontology_retriever.relative_verbalized_graph_size("FOOD#QUALITY"))