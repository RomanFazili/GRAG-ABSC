from data_set import DataSet, Polarity
from rank_bm25 import BM25Okapi
import numpy as np
from typing import Callable
from dotenv import load_dotenv
import os
import re
from sentence_transformers import SentenceTransformer, util


Tokenizer = Callable[[str], list[str]]


def default_tokenizer(text: str) -> list[str]:
    """
    Better option than splitting on whitespace.
    This will also tokenize contractions and other words with apostrophes.
    """
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())

class SentenceRetriever:

    def __init__(self, data_set: DataSet, tokenizer: Tokenizer | None = None):
        self.data_set = data_set
        self._tokenizer: Tokenizer = tokenizer or default_tokenizer

    def tokenize(self, text: str) -> list[str]:
        return self._tokenizer(text)

    def BM25_demonstration_selection(self, query_sentence: str, top_k: int) -> list[tuple[str, list[tuple[str, Polarity]]]]:
        """
        Use the BM25 algorithm to retrieve the top k most similar sentences 
        to the query sentence.
        Returns a list of tuples containing the sentence, the aspect, and the polarity.
        """

        all_sentences_with_aspects_and_polarities: list[tuple[str, list[tuple[str, Polarity]]]] = self.data_set.all_sentences_with_aspects_and_polarities

        tokenized_train_data: list[list[str]] = [
            self.tokenize(sentence) for sentence, _ in all_sentences_with_aspects_and_polarities
        ]
        tokenized_query: list[str] = self.tokenize(query_sentence)

        bm25 = BM25Okapi(corpus=tokenized_train_data)

        scores: np.ndarray = bm25.get_scores(tokenized_query)

        # Invert and get the k most similar sentences
        top_indices: np.ndarray = scores.argsort()[-top_k:][::-1]

        return [all_sentences_with_aspects_and_polarities[i] for i in top_indices]


    def SimCSE_demonstration_selection(self, query_sentence: str, top_k: int) -> list[tuple[str, list[tuple[str, Polarity]]]]:
        """Use a SimCSE-like SentenceTransformer model to retrieve
        the top k most similar sentences to the query sentence."""
        model: SentenceTransformer = SentenceTransformer('princeton-nlp/unsup-simcse-bert-base-uncased')

        all_sentences_with_aspects_and_polarities: list[tuple[str, list[tuple[str, Polarity]]]] = self.data_set.all_sentences_with_aspects_and_polarities

        sentence_embeddings: np.ndarray = model.encode([sentence for sentence, _ in all_sentences_with_aspects_and_polarities])

        query_embedding: np.ndarray = model.encode([query_sentence])

        scores: np.ndarray = util.pytorch_cos_sim(query_embedding, sentence_embeddings)

        top_indices: np.ndarray = scores[0].argsort(descending=True)[:top_k]

        return [all_sentences_with_aspects_and_polarities[i] for i in top_indices]

    def graph_based_demonstration_selection(self, query_sentence: str, top_k: int, ontology):
        
        raise NotImplementedError("Graph-based demonstration selection is not implemented yet")

if __name__ == "__main__":
    load_dotenv()
    file_path = os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_15_RESTAURANTS_TRAIN_DATA")
    data_set = DataSet(file_path)
    sentence_retriever = SentenceRetriever(data_set)
    print(sentence_retriever.BM25_demonstration_selection("The food was good", 3))
    print(sentence_retriever.SimCSE_demonstration_selection("The food was good", 3))