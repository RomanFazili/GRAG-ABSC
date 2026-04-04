import os
import re
from typing import Callable

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

from data_set import DataSet
from enums import Polarity

Tokenizer = Callable[[str], list[str]]

_SIMCSE_MODEL_NAME = "princeton-nlp/unsup-simcse-bert-base-uncased"



class SentenceRetriever:
    """Retrieves demonstration sentences; BM25 / SimCSE indices are built lazily once each."""

    def __init__(self, data_set: DataSet):
        self.data_set = data_set

        self._corpus_rows: list[tuple[str, list[tuple[str, Polarity]]]] | None = None
        self._bm25_tokenized: list[list[str]] | None = None
        self._bm25_index: BM25Okapi | None = None
        self._simcse_model: SentenceTransformer | None = None
        self._simcse_embeddings: np.ndarray | None = None

    def _get_corpus_rows(self) -> list[tuple[str, list[tuple[str, Polarity]]]]:
        if self._corpus_rows is None:
            self._corpus_rows = list(self.data_set.all_sentences_with_aspects_and_polarities)
        return self._corpus_rows


    def _ensure_bm25_index(self) -> None:
        if self._bm25_index is not None:
            return
        corpus = self._get_corpus_rows()
        self._bm25_tokenized = [self.tokenize_bm25(sentence) for sentence, _ in corpus]
        self._bm25_index = BM25Okapi(corpus=self._bm25_tokenized)

    def _ensure_simcse_embeddings(self) -> None:
        if self._simcse_embeddings is not None:
            return
        corpus = self._get_corpus_rows()
        if self._simcse_model is None:
            self._simcse_model = SentenceTransformer(_SIMCSE_MODEL_NAME)
        sentences = [sentence for sentence, _ in corpus]
        self._simcse_embeddings = self._simcse_model.encode(
            sentences,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def tokenize_bm25(self, text: str) -> list[str]:
        """Tokenize for BM25: lowercase alnum tokens including simple apostrophe forms."""
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())

    def BM25_demonstration_selection(
        self, query_sentence: str, top_k: int
    ) -> list[tuple[str, list[tuple[str, Polarity]]]]:
        """BM25 over the training corpus; index built on first call only."""
        self._ensure_bm25_index()
        corpus = self._get_corpus_rows()
        assert self._bm25_index is not None

        tokenized_query = self.tokenize_bm25(query_sentence)
        scores: np.ndarray = self._bm25_index.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]
        return [corpus[int(i)] for i in top_indices]

    def SimCSE_demonstration_selection(
        self, query_sentence: str, top_k: int
    ) -> list[tuple[str, list[tuple[str, Polarity]]]]:
        """SentenceTransformer similarity; model + corpus embeddings built on first call only."""
        self._ensure_simcse_embeddings()
        corpus = self._get_corpus_rows()
        assert self._simcse_model is not None
        assert self._simcse_embeddings is not None

        query_embedding: np.ndarray = self._simcse_model.encode(
            [query_sentence],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = util.pytorch_cos_sim(query_embedding, self._simcse_embeddings)
        top_indices = scores[0].argsort(descending=True)[:top_k].cpu().tolist()
        return [corpus[int(i)] for i in top_indices]

    def graph_based_demonstration_selection(self, query_sentence: str, top_k: int, ontology):
        raise NotImplementedError("Graph-based demonstration selection is not implemented yet")


if __name__ == "__main__":
    load_dotenv()
    file_path = os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_15_RESTAURANTS_TRAIN_DATA")
    assert file_path
    sentence_retriever = SentenceRetriever(DataSet(file_path))
    print(sentence_retriever.BM25_demonstration_selection("The food was good", 3))
    print(sentence_retriever.SimCSE_demonstration_selection("The food was good", 3))
