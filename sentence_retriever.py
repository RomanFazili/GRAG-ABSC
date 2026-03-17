from data_set import DataSet
from rank_bm25 import BM25Okapi
import numpy as np
from typing import Callable
from dotenv import load_dotenv
import os


Tokenizer = Callable[[str], list[str]]


def default_tokenizer(text: str) -> list[str]:
    return text.lower().split()


class SentenceRetriever:

    def __init__(self, data_set: DataSet, tokenizer: Tokenizer | None = None):
        self.data_set = data_set
        self._tokenizer: Tokenizer = tokenizer or default_tokenizer

    def tokenize(self, text: str) -> list[str]:
        return self._tokenizer(text)

    def BM25_demonstration_selection(self, query_sentence: str, top_k: int):

        all_sentences: list[str] = self.data_set.all_sentences_as_text

        tokenized_train_data: list[list[str]] = [
            self.tokenize(sentence) for sentence in all_sentences
        ]
        tokenized_query: list[str] = self.tokenize(query_sentence)

        bm25 = BM25Okapi(corpus=tokenized_train_data)

        scores: np.ndarray = bm25.get_scores(tokenized_query)

        # Invert and get the k most similar sentences
        top_indices: np.ndarray = scores.argsort()[-top_k:][::-1]

        return [all_sentences[i] for i in top_indices]

    def SimCSE_demonstration_selection(self, query_sentence: str, top_k: int):
        raise NotImplementedError("SimCSE demonstration selection is not implemented")



if __name__ == "__main__":
    load_dotenv()
    file_path = os.getenv("PATH_TO_SEMEVAL_16_TRAIN_DATA")
    data_set = DataSet(file_path)
    sentence_retriever = SentenceRetriever(data_set)
    print(sentence_retriever.BM25_demonstration_selection("The food was good", 3))
    # print(sentence_retriever.SimCSE_demonstration_selection("The food was good", 3))