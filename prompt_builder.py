import os
from dotenv import load_dotenv
from data_set import DataSet, Polarity
from data_set_ontology import DataSetOntology
from ontology_retriever import OntologyRetriever
from sentence_retriever import SentenceRetriever
from enum import Enum
from rdflib import Graph


class OntologySelectionMethod(Enum):
    Nothing = "Nothing"
    Partial = "Partial"
    Full = "Full"

class PromptFormat(Enum):
    Type1 = "Type1"
    Type2 = "Type2"
    Type3 = "Type3"

class OntologyFormat(Enum):
    XML = "xml"
    N3 = "n3"
    NT = "nt"

class DemonstrationSelectionMethod(Enum):
    BM25 = "BM25"
    SimCSE = "SimCSE"
    Graph = "Graph"


class PromptBuilder:
    """
    Build ABSA prompts with as little setup as possible.
    
    The input variables to the prompt builder are:
    - input_sentence
    - aspect: the aspect of the input sentence
    - demonstration_sentences:
        - demonstration_selection_method
        - top_k
        - train data filepath
    - ontology_injection:
        - ontology_selection_method [nothing, partial (type 1&2&3 combined), full]
        - ontology filepath
    - prompt format
    """

    @staticmethod
    def build_prompt(
        input_sentence: str,
        aspect: str, 
        demonstration_selection_method: DemonstrationSelectionMethod, 
        top_k: int, 
        train_data_filepath: str, 
        ontology_selection_method: OntologySelectionMethod, 
        ontology_filepath: str, 
        ontology_format: OntologyFormat,
        prompt_format: PromptFormat
    ) -> str:

        sentence_retriever = SentenceRetriever(DataSet(train_data_filepath))

        demonstration_sentences: list[tuple[str, list[tuple[str, Polarity]]]] = []

        if demonstration_selection_method == DemonstrationSelectionMethod.BM25:
            demonstration_sentences = sentence_retriever.BM25_demonstration_selection(
                input_sentence, top_k
            )
        elif demonstration_selection_method == DemonstrationSelectionMethod.SimCSE:
            raise NotImplementedError("SimCSE demonstration selection is not implemented yet, still needs to return aspect and polarity")
            demonstration_sentences = sentence_retriever.SimCSE_demonstration_selection(
                input_sentence, top_k
            )
        elif demonstration_selection_method == DemonstrationSelectionMethod.Graph:
            raise NotImplementedError("Graph-based demonstration selection is not implemented yet")
            demonstration_sentences = sentence_retriever.graph_based_demonstration_selection(
                input_sentence, top_k
            )

        formatted_demonstrations = PromptBuilder._format_demonstrations(demonstration_sentences)

        data_set_ontology = DataSetOntology(ontology_filepath)
        ontology_retriever = OntologyRetriever(data_set_ontology)

        selected_ontology: Graph | None = None
        if ontology_selection_method == OntologySelectionMethod.Nothing:
            selected_ontology: Graph = None
        elif ontology_selection_method == OntologySelectionMethod.Partial:
            raise NotImplementedError("Partial ontology selection method is not implemented yet")
            selected_ontology: Graph = ontology_retriever.verbalize_aspect_category_sentiments_restaurant_type_1()
        elif ontology_selection_method == OntologySelectionMethod.Full:
            selected_ontology: Graph = ontology_retriever.data_set_ontology.get_rdflib_graph()
        else:
            raise ValueError(f"Invalid ontology selection method: {ontology_selection_method}")

        
        formatted_ontology: str | None = None
        if selected_ontology:
            if ontology_format == OntologyFormat.XML:
                formatted_ontology = selected_ontology.serialize(format="xml")
            elif ontology_format == OntologyFormat.N3:
                formatted_ontology = selected_ontology.serialize(format="n3")
            elif ontology_format == OntologyFormat.NT:
                formatted_ontology = selected_ontology.serialize(format="nt")
            else:
                raise ValueError(f"Invalid ontology format: {ontology_format}")


        if prompt_format == PromptFormat.Type1:
            prompt = PromptBuilder._build_prompt_type1(
                input_sentence=input_sentence,
                aspect=aspect,
                formatted_demonstrations=formatted_demonstrations,
                formatted_ontology=formatted_ontology,
            )
        else:
            raise NotImplementedError("Prompt format is not implemented yet")

        return prompt

    @staticmethod
    def _format_demonstrations(
        demonstration_sentences: list[tuple[str, list[tuple[str, Polarity]]]]
    ) -> str:
        return "\n".join(
            [
                f"Sentence: {sentence}\nAspects and Polarities: {[(aspect, str(polarity)) for aspect, polarity in aspects_and_polarities]}"
                for sentence, aspects_and_polarities in demonstration_sentences
            ]
        )

    @staticmethod
    def _build_prompt_type1(
        input_sentence: str,
        aspect: str,
        formatted_demonstrations: str,
        formatted_ontology: str | None
    ) -> str:

        prompt = (
            f"You must return the sentiment polarity of the following sentence:\n"
            f"{input_sentence}\n"
            f"With the given aspect of {aspect}\n"
            "\n"
            f"You may use these demonstration sentences with the given aspects and polarities to help you:\n"
            f"{formatted_demonstrations}\n"
        )

        if formatted_ontology:
            prompt += (
                "\n"
                f"You may use this ontology to help you:\n"
                f"{formatted_ontology}\n"
            )

        return prompt

if __name__ == "__main__":
    load_dotenv()

    prompt = PromptBuilder.build_prompt(
        input_sentence="The food was good",
        aspect="food",
        demonstration_selection_method=DemonstrationSelectionMethod.BM25,
        top_k=3,
        train_data_filepath=os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_15_RESTAURANTS_TRAIN_DATA"),
        ontology_selection_method=OntologySelectionMethod.Nothing,
        ontology_filepath=os.getenv("PATH_TO_RESTAURANT_ONTOLOGY"),
        ontology_format=OntologyFormat.XML,
        prompt_format=PromptFormat.Type1,
    )

    print(prompt)
    exit()

    # for domain in ["laptop", "restaurant"]:
    #     for yeah in [2015, 2016]:
    #         for input_sentence, aspect in training_sentences:
    #             for demonstration_selection_method in DemonstrationSelectionMethod:
    #                 for top_k in [0, 3]:
    #                     for ontology_selection_method in OntologySelectionMethod:
    #                         for ontology_format in OntologyFormat:
    #                             # for prompt_format in PromptFormat:
    #                                 for ai_model in [...]