from data_set import DataSet
from sentence_retriever import SentenceRetriever
from dotenv import load_dotenv
import os
from ontology_retriever import OntologyRetriever
from data_set import Polarity
from data_set_ontology import DataSetOntology

top_k = 3

input_sentence = "The food was good"
aspect = "food"

load_dotenv()
file_path = os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_15_RESTAURANTS_TRAIN_DATA")
data_set = DataSet(file_path)
sentence_retriever = SentenceRetriever(data_set)

demonstration_sentences: list[tuple[str, list[tuple[str, Polarity]]]] = sentence_retriever.BM25_demonstration_selection(input_sentence, top_k)
formatted_demonstration_sentences = "\n".join(
    [
        f"Sentence: {sentence}\nAspects and Polarities: {[(aspect, str(polarity)) for aspect, polarity in aspects_and_polarities]}"
        for sentence, aspects_and_polarities in demonstration_sentences
    ]
)
data_set_ontology = DataSetOntology(os.getenv("PATH_TO_RESTAURANT_ONTOLOGY"))
ontology_retriever = OntologyRetriever(data_set_ontology)
ontology_injection = ontology_retriever.verbalize_aspect_category_sentiments_restaurant_type_3(aspect)
formatted_ontology_injection: str = ontology_injection.serialize(format="xml")

example_prompt_format = f"""
You must return the sentiment polarity of the following sentence:
{input_sentence}
With the given aspect of {aspect}

You may use these demonstrations sentences with the given aspects and polarities to help you:
{formatted_demonstration_sentences}

You may use this ontology to help you:
{formatted_ontology_injection}
"""

print(example_prompt_format)