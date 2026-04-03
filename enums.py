from enum import StrEnum


class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ExperimentDomain(StrEnum):
    RESTAURANT = "restaurant"
    LAPTOP = "laptop"


class OntologySelectionMethod(StrEnum):
    Nothing = "Nothing"
    Partial = "Partial"  # type 1&2&3 combined
    Full = "Full"


class OntologyFormat(StrEnum):
    XML = "xml"
    N3 = "n3"
    NT = "nt"


class DemonstrationSelectionMethod(StrEnum):
    BM25 = "BM25"
    SimCSE = "SimCSE"
    Graph = "Graph"


class LLMModel(StrEnum):
    """Hugging Face repo ids for locally loaded causal LMs."""

    GEMMA_2_2B_IT = "google/gemma-2-2b-it"
