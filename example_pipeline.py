from openrouter import OpenRouter
import prompt_builder
from data_set import DataSet
from enums import (
    DemonstrationSelectionMethod,
    OntologyFormat,
    OntologySelectionMethod,
    Polarity,
)
from prompt_builder import PromptBuilder
from sentence_retriever import SentenceRetriever
from dotenv import load_dotenv
import os
from ontology_retriever import OntologyRetriever
from rdflib import Graph
from data_set_ontology import DataSetOntology
from openai import OpenAI
from sklearn.metrics import f1_score, classification_report
import json


def client_response(client: OpenAI, model: str, prompt: str) -> Polarity:
    """Returns Polarity from openAI client based on the given prompt"""
    response = client.chat.completions.create(
        model = model,
        messages = [{"role": "user", "content": prompt}]
    )
    response_text = response.choices[0].message.content.lower().strip()

    raise NotImplementedError("Client response is not implemented yet, we don't yet know how to let the model respond")
    
    # Extract polarity from response
    if "positive" in response_text:
        return Polarity.POSITIVE
    elif "negative" in response_text:
        return Polarity.NEGATIVE
    elif "neutral" in response_text:
        return Polarity.NEUTRAL
    else:
        # Default fallback if polarity not clearly identified
        raise ValueError(f"Could not extract polarity from response: {response.choices[0].message.content}")


def store_prediction(sentence_text: str, aspect: str, predicted_polarity: Polarity, true_polarity: Polarity) -> dict:
    """Store a single prediction result"""
    return {
        "sentence_text": sentence_text,
        "aspect": aspect,
        "predicted_polarity": str(predicted_polarity),
        "true_polarity": str(true_polarity),
        "is_correct": true_polarity == predicted_polarity
    }


def calculate_f1_scores(predictions: list[dict]) -> dict:
    """Calculates weighted and macro F1 scores from predictions"""
    if not predictions:
        raise ValueError("No predictions to calculate F1 scores")
    
    y_true = [Polarity(p["true_polarity"]) for p in predictions]
    y_pred = [Polarity(p["predicted_polarity"]) for p in predictions]
    
    # Convert to comparable format for sklearn
    y_true_labels = [str(p) for p in y_true]
    y_pred_labels = [str(p) for p in y_pred]
    
    return {
        "macro_f1": f1_score(y_true_labels, y_pred_labels, average="macro"),
        "weighted_f1": f1_score(y_true_labels, y_pred_labels, average="weighted"),
        "accuracy": sum(p["is_correct"] for p in predictions) / len(predictions),
        "classification_report": classification_report(y_true_labels, y_pred_labels, output_dict=True),
        "total_predictions": len(predictions),
        "correct_predictions": sum(p["is_correct"] for p in predictions)
    }


def save_predictions_to_file(predictions: list[dict], filepath: str):
    """Save predictions to a JSON file for later analysis"""
    with open(filepath, 'w') as f:
        json.dump(predictions, f, indent=2)


# demonstration_sentences: list[tuple[str, list[tuple[str, Polarity]]]] = sentence_retriever.BM25_demonstration_selection(input_sentence, top_k)
# formatted_demonstration_sentences = "\n".join(
#     [
#         f"Sentence: {sentence}\nAspects and Polarities: {[(aspect, str(polarity)) for aspect, polarity in aspects_and_polarities]}"
#         for sentence, aspects_and_polarities in demonstration_sentences
#     ]
# )
# data_set_ontology = DataSetOntology(os.getenv("PATH_TO_RESTAURANT_ONTOLOGY"))
# ontology_retriever = OntologyRetriever(data_set_ontology)
# ontology_injection: Graph = ontology_retriever.verbalize_aspect_category_sentiments_restaurant_type_2(aspect)
#
# # formatted_ontology_injection: str = ontology_injection.serialize(format="xml")
# formatted_ontology_injection: str = ontology_injection.serialize(format="n3")
# # formatted_ontology_injection: str = ontology_injection.serialize(format="nt")
#
# example_prompt_format = f"""
# You must return the sentiment polarity of the following sentence:
# {input_sentence}
# With the given aspect of {aspect}
#
# You may use these demonstrations sentences with the given aspects and polarities to help you:
# {formatted_demonstration_sentences}
#
# You may use this ontology to help you:
# {formatted_ontology_injection}
# """

# print(example_prompt_format)


# Top part of main (now unused) for single predictions, bottom part for predictions over full dataset
if __name__ == "__main__":
    load_dotenv()
    file_path = os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_15_RESTAURANTS_TEST_DATA")
    data_set = DataSet(file_path)
    sentence_retriever = SentenceRetriever(data_set)

    # client and model example, probably move later
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = "gpt-5.4-mini"

    predictions: list[dict] = []

    demonstration_selection_method = DemonstrationSelectionMethod.BM25
    top_k = 3
    train_data_filepath = os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_15_RESTAURANTS_TRAIN_DATA")
    ontology_selection_method = OntologySelectionMethod.Nothing
    ontology_filepath = os.getenv("PATH_TO_RESTAURANT_ONTOLOGY")
    ontology_format = OntologyFormat.XML

    i=0
    for sentence_text, aspects_with_true_polarities in data_set.all_sentences_with_aspects_and_polarities:
        i+=1
        if i>120:
            break
        for aspect, true_polarity in aspects_with_true_polarities:
            prompt = PromptBuilder.build_prompt(
                sentence_text,
                aspect,
                demonstration_selection_method,
                top_k,
                train_data_filepath,
                ontology_selection_method,
                ontology_filepath,
                ontology_format
            )

            predicted_polarity = client_response(client, model, prompt)

            prediction = store_prediction(
                sentence_text=sentence_text,
                aspect=aspect,
                predicted_polarity=predicted_polarity,
                true_polarity=true_polarity
            )
            predictions.append(prediction)

    f1_results = calculate_f1_scores(predictions)
    print(f"\n=== Results ===")
    print(f"Macro F1: {f1_results['macro_f1']:.4f}")
    print(f"Weighted F1: {f1_results['weighted_f1']:.4f}")
    print(f"Accuracy: {f1_results['accuracy']:.4f}")
    print(f"Correct: {f1_results['correct_predictions']} / {f1_results['total_predictions']}")



    # The part art below is only for testing sentences individually

    # test_sentence = 'The space is limited so be prepared to wait up to 45 minutes - 1 hour, but be richly rewarded when you savor the delicious indo-chinese food.'
    # test_aspect = 'space'

    # prompt = PromptBuilder.build_prompt(test_sentence, test_aspect, demonstration_selection_method, top_k, train_data_filepath, ontology_selection_method, ontology_filepath, ontology_format)
    # predicted_polarity = client_response(openai_client, model_gpt54, prompt)
    #
    # print(prompt)
    # print(predicted_polarity)
