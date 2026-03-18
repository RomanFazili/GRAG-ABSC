from data_set import DataSet
from sentence_retriever import SentenceRetriever
from dotenv import load_dotenv
import os

input_sentence = "The food was good"
aspect = "food"

load_dotenv()
file_path = os.getenv("PATH_TO_SEMEVAL_16_TRAIN_DATA")
data_set = DataSet(file_path)
sentence_retriever = SentenceRetriever(data_set)

demonstration_sentences = sentence_retriever.BM25_demonstration_selection(input_sentence, 5)



demonstration_sentences_with_aspects_and_polarities = None

ontology = None


example_prompt_format = f"""
You must return the sentiment polarity of the following sentence:
{input_sentence}
With the given aspect of {aspect}

You may use these demonstrations sentences with the given aspects and polarities to help you:
{demonstration_sentences_with_aspects_and_polarities}

You may use this ontology to help you:
{ontology}
"""

print(example_prompt_format)