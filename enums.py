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


# Might not be complete:
# class AspectCategory(StrEnum):
#     FOOD_QUALITY = "FOOD#QUALITY"
#     FOOD_PRICES = "FOOD#PRICES"
#     FOOD_STYLE_OPTIONS = "FOOD#STYLE_OPTIONS"
#     FOOD_GENERAL = "FOOD#GENERAL"
#     DRINKS_QUALITY = "DRINKS#QUALITY"
#     DRINKS_PRICES = "DRINKS#PRICES"
#     DRINKS_STYLE_OPTIONS = "DRINKS#STYLE_OPTIONS"
#     SERVICE_GENERAL = "SERVICE#GENERAL"
#     RESTAURANT_GENERAL = "RESTAURANT#GENERAL"
#     RESTAURANT_PRICES = "RESTAURANT#PRICES"
#     RESTAURANT_MISCELLANEOUS = "RESTAURANT#MISCELLANEOUS"
#     LOCATION_GENERAL = "LOCATION#GENERAL"