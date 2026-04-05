import os
from dotenv import load_dotenv
from data_set import DataSet
from data_set_ontology import DataSetOntology
from ontology_retriever import OntologyRetriever
from sentence_retriever import SentenceRetriever
from rdflib import Graph

from enums import (
    DemonstrationSelectionMethod,
    OntologyFormat,
    LLMModel,
    OntologySelectionMethod,
    Polarity,
)


class PromptBuilder:
    """
    Build ABSA prompts with as little setup as possible.
    
    The input variables to the prompt builder are:
    - input_sentence
    - aspect: the aspect of the input sentence
    - aspect_category: the category of the aspect
    - demonstration_sentences:
        - demonstration_selection_method
        - top_k
        - sentence_retriever: built from the training ``DataSet`` (create once per run)
    - ontology_injection:
        - ontology_selection_method [nothing, partial (type 1&2&3 combined), full]
        - ontology_retriever: wrap a single loaded ``DataSetOntology`` (create once per run)
    - prompt format
    """

    @staticmethod
    def build_prompt(
        input_sentence: str,
        aspect: str,
        aspect_category: str,
        demonstration_selection_method: DemonstrationSelectionMethod,
        top_k: int,
        sentence_retriever: SentenceRetriever,
        ontology_retriever: OntologyRetriever,
        ontology_selection_method: OntologySelectionMethod,
        ontology_format: OntologyFormat,
    ) -> str:

        assert isinstance(demonstration_selection_method, DemonstrationSelectionMethod)
        assert isinstance(ontology_selection_method, OntologySelectionMethod)
        assert isinstance(ontology_format, OntologyFormat)

        formatted_demonstrations = PromptBuilder._format_demonstrations(
            demonstration_selection_method,
            top_k,
            sentence_retriever,
            input_sentence,
        )

        formatted_ontology = PromptBuilder._format_ontology(
            ontology_retriever,
            ontology_selection_method, 
            ontology_format,
            aspect_category
        )

        prompt = PromptBuilder._build_prompt(
            input_sentence=input_sentence,
            aspect=aspect,
            aspect_category=aspect_category,
            formatted_demonstrations=formatted_demonstrations,
            formatted_ontology=formatted_ontology,
        )

        return prompt

    @staticmethod
    def _format_demonstrations(
        demonstration_selection_method: DemonstrationSelectionMethod,
        top_k: int,
        sentence_retriever: SentenceRetriever,
        input_sentence: str,
    ) -> str | None:

        demonstration_sentences: list[tuple[str, list[tuple[str, Polarity]]]] = []

        if demonstration_selection_method == DemonstrationSelectionMethod.BM25:
            demonstration_sentences = sentence_retriever.BM25_demonstration_selection(
                input_sentence, top_k
            )
        elif demonstration_selection_method == DemonstrationSelectionMethod.SimCSE:
            demonstration_sentences = sentence_retriever.SimCSE_demonstration_selection(
                input_sentence, top_k
            )
        elif demonstration_selection_method == DemonstrationSelectionMethod.Graph:
            raise NotImplementedError("Graph-based demonstration selection is not implemented yet")
            demonstration_sentences = sentence_retriever.graph_based_demonstration_selection(
                input_sentence, top_k
            )

        if not demonstration_sentences:
            return None

        return "\n\n".join(
            [
                f"Sentence: {sentence}\nTarget Aspect: {aspect}\nPolarity: {polarity.value}"
                for sentence, aspects_and_polarities in demonstration_sentences
                for aspect, polarity in aspects_and_polarities
            ]
        )
        # return "\n".join(
        #     [
        #         f"Sentence: {sentence}\nAspects and Polarities: {[(aspect, str(polarity)) for aspect, polarity in aspects_and_polarities]}"
        #         for sentence, aspects_and_polarities in demonstration_sentences
        #     ]
        # )

    @staticmethod
    def _format_ontology(
        ontology_retriever: OntologyRetriever,
        ontology_selection_method: OntologySelectionMethod,
        ontology_format: OntologyFormat,
        aspect_category: str,
    ) -> str | None:

        selected_ontology: Graph | None = None
        if ontology_selection_method == OntologySelectionMethod.Nothing:
            selected_ontology = None
        elif ontology_selection_method == OntologySelectionMethod.Partial:
            selected_ontology = ontology_retriever.verbalize(aspect_category=aspect_category)
        elif ontology_selection_method == OntologySelectionMethod.Full:
            selected_ontology = ontology_retriever.data_set_ontology.get_rdflib_graph()

        if selected_ontology:
            if ontology_format == OntologyFormat.XML:
                return selected_ontology.serialize(format="xml")
            elif ontology_format == OntologyFormat.N3:
                return selected_ontology.serialize(format="n3")
            elif ontology_format == OntologyFormat.NT:
                return selected_ontology.serialize(format="nt")

        return None

    @staticmethod
    def _build_prompt(
        input_sentence: str,
        aspect: str,
        aspect_category: str,
        formatted_demonstrations: str | None,
        formatted_ontology: str | None
    ) -> str:

        prompt = (
            "Your task is to classify the sentiment of a target aspect within a sentence.\n"
            "You must respond with only one of the following words: positive, negative, or neutral.\n"
        )

        if formatted_ontology:
            prompt += (
                "\n"
                f"You may use this ontology to help you:\n"
                f"{formatted_ontology}\n"
            )

        if formatted_demonstrations:
            prompt += (
                "\n"
                f"You may use these demonstration sentences with the given aspects and polarities to help you:\n"
                f"{formatted_demonstrations}\n"
            )

        prompt += (
            "\n"
            f"Sentence: {input_sentence}\n"
            f"Target Aspect: {aspect}\n"
            f"Sentiment:"
        )

        return prompt

    @staticmethod
    def get_median_prompt_length_tokens(test_path: str, ontology_path: str, model: LLMModel) -> float:
        """
        Get the median prompt length in tokens for all combinations of settings.

        Restaurant 2015 test data on Gemma 2b IT tokenizer: 14254 tokens
        """
        import statistics

        from transformers import AutoTokenizer

        assert test_path and ontology_path

        load_dotenv()
        hf_tok = os.getenv("HF_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(model.value, token=hf_tok)

        sentence_retriever = SentenceRetriever(DataSet(test_path))
        ontology_retriever = OntologyRetriever(DataSetOntology(ontology_path))

        token_counts: list[int] = []
        for demonstration_selection_method in DemonstrationSelectionMethod:
            if demonstration_selection_method == DemonstrationSelectionMethod.Graph:
                continue

            for selection_method in OntologySelectionMethod:
                for ontology_format in OntologyFormat:
                    for top_k in [0, 3]:
                        for sentence, aspects_categories_and_polarities in sentence_retriever.data_set.all_sentences_with_aspects_categories_and_polarities:
                            for aspect, aspect_category, _ in aspects_categories_and_polarities:
                    
                                prompt = PromptBuilder.build_prompt(
                                    input_sentence=sentence,
                                    aspect=aspect,
                                    aspect_category=aspect_category,
                                    demonstration_selection_method=demonstration_selection_method,
                                    top_k=top_k,
                                    sentence_retriever=sentence_retriever,
                                    ontology_retriever=ontology_retriever,
                                    ontology_selection_method=selection_method,
                                    ontology_format=ontology_format,
                                )

                                token_counts.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
        return statistics.median(token_counts)

if __name__ == "__main__":
    load_dotenv()
    train_path = os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_15_RESTAURANTS_TEST_DATA")
    ontology_path = os.getenv("PATH_TO_RESTAURANT_ONTOLOGY")

    print(PromptBuilder.build_prompt(
        input_sentence="The food was good",
        aspect="food",
        aspect_category="FOOD#QUALITY",
        demonstration_selection_method=DemonstrationSelectionMethod.BM25,
        top_k=3,
        sentence_retriever=SentenceRetriever(DataSet(train_path)),
        ontology_retriever=OntologyRetriever(DataSetOntology(ontology_path)),
        ontology_selection_method=OntologySelectionMethod.Partial,
        ontology_format=OntologyFormat.XML,
    ))
    exit()

    test_path = os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_15_RESTAURANTS_TEST_DATA")
    ontology_path = os.getenv("PATH_TO_RESTAURANT_ONTOLOGY")
    model = LLMModel.GEMMA_2_2B_IT

    median_tokens = PromptBuilder.get_median_prompt_length_tokens(test_path, ontology_path, model)
    print(
        f"Median prompt length: {median_tokens:.1f} tokens "
    )
