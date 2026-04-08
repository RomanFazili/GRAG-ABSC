import asyncio
from dataclasses import dataclass
from openai import AsyncOpenAI, APIConnectionError, BadRequestError
from prompt_builder import PromptBuilder
from enums import DemonstrationSelectionMethod, OntologySelectionMethod, OntologyFormat, Polarity
from sentence_retriever import SentenceRetriever
from data_set import DataSet
from data_set_ontology import DataSetOntology
from ontology_retriever import OntologyRetriever
import os
from dotenv import load_dotenv
import json
from sklearn.metrics import classification_report

load_dotenv()

@dataclass
class Job:
    input_sentence: str
    aspect: str
    aspect_category: str
    true_polarity: Polarity
    demonstration_selection_method: DemonstrationSelectionMethod
    top_k: int
    sentence_retriever: SentenceRetriever
    ontology_retriever: OntologyRetriever
    ontology_selection_method: OntologySelectionMethod
    ontology_format: OntologyFormat
    model: str
    prompt: str | None
    llm_output: str | None

    @property
    def is_correct(self) -> bool:
        return self.true_polarity.value.lower() == self.llm_output.lower()

    def as_dict(self) -> dict:
        return {
            "input_sentence": self.input_sentence,
            "aspect": self.aspect,
            "aspect_category": self.aspect_category,
            "true_polarity": self.true_polarity.value,
            "demonstration_selection_method": self.demonstration_selection_method.value,
            "top_k": self.top_k,
            "sentence_retriever_path": self.sentence_retriever.data_set.file_path,
            "ontology_retriever_path": self.ontology_retriever.data_set_ontology.file_path,
            "ontology_selection_method": self.ontology_selection_method.value,
            "ontology_format": self.ontology_format.value,
            "model": self.model,
            "prompt": self.prompt,
            "llm_output": self.llm_output,
        }

def calculate_evaluation_metrics(finished_jobs: list[Job]) -> dict:
    """Calculates weighted and macro F1 scores from predictions"""
    if not finished_jobs:
        raise ValueError("No finished jobs to calculate evaluation metrics")
    
    y_true_labels: list[str] = [job.true_polarity.value for job in finished_jobs]
    y_pred_labels: list[str] = [job.llm_output for job in finished_jobs]
    
    return {
        "classification_report": classification_report(y_true_labels, y_pred_labels, output_dict=True),
        "total_predictions": len(finished_jobs),
        "correct_predictions": sum(job.is_correct for job in finished_jobs)
    }

client = AsyncOpenAI(
    base_url=os.getenv("OPEN_AI_BASE_URL"),
    api_key=os.getenv("OPEN_AI_API_KEY")
)

# Create all prompts
prompts: list[str] = []
jobs: list[Job] = []

test_path = os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_15_RESTAURANTS_TEST_DATA")
train_path = os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_15_RESTAURANTS_TRAIN_DATA")
ontology_path = os.getenv("PATH_TO_RESTAURANT_ONTOLOGY")

test_data_set = DataSet(test_path)
train_data_set = DataSet(train_path)

sentence_retriever = SentenceRetriever(train_data_set)
ontology_retriever = OntologyRetriever(DataSetOntology(ontology_path))

def full_run(
    demonstration_selection_method: DemonstrationSelectionMethod,
    ontology_selection_method: OntologySelectionMethod,
    ontology_format: OntologyFormat,
    top_k: int,
    model: str,
):
    jobs: list[Job] = []
    for sentence, aspects_categories_and_polarities in test_data_set.all_sentences_with_aspects_categories_and_polarities:
        for aspect, aspect_category, true_polarity in aspects_categories_and_polarities:
            prompt = PromptBuilder.build_prompt(
                input_sentence=sentence,
                aspect=aspect,
                aspect_category=aspect_category,
                demonstration_selection_method=demonstration_selection_method,
                top_k=top_k,
                sentence_retriever=sentence_retriever,
                ontology_retriever=ontology_retriever,
                ontology_selection_method=ontology_selection_method,
                ontology_format=ontology_format,
            )
            jobs.append(Job(
                input_sentence=sentence,
                aspect=aspect,
                aspect_category=aspect_category,
                true_polarity=true_polarity,
                demonstration_selection_method=demonstration_selection_method,
                top_k=top_k,
                sentence_retriever=sentence_retriever,
                ontology_retriever=ontology_retriever,
                ontology_selection_method=ontology_selection_method,
                ontology_format=ontology_format,
                model=model,
                prompt=prompt,
                llm_output=None,
            ))

    jobs.sort(key=lambda x: x.prompt) # make use of kv caching
    print(len(jobs))

    async def run(job: Job):
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=job.model,
                    messages=[{"role":"user","content":job.prompt}],
                    max_tokens=1
                )
                answer = response.choices[0].message.content
                print(answer)
                job.llm_output = answer
                return job
            except APIConnectionError:
                if attempt == 2:
                    raise
            except BadRequestError:
                job.llm_output = "error"
                return job

    async def main():
        tasks = [run(job) for job in jobs]
        return await asyncio.gather(*tasks)

    results = asyncio.run(main())

    evaluation_metrics = calculate_evaluation_metrics(results)

    evaluation_metrics["test_path"] = test_path
    evaluation_metrics["train_path"] = train_path
    evaluation_metrics["ontology_path"] = ontology_path

    evaluation_metrics["model"] = model
    evaluation_metrics["demonstration_selection_method"] = demonstration_selection_method.value
    evaluation_metrics["ontology_selection_method"] = ontology_selection_method.value
    evaluation_metrics["ontology_format"] = ontology_format.value
    evaluation_metrics["top_k"] = top_k

    with open("final_results_roman.jsonl", "a") as f:
        f.write(json.dumps(evaluation_metrics) + "\n")

    print(evaluation_metrics)


model = "meta-llama/Llama-3.2-3B-Instruct"
# 3 * 2 * 3 * 4 = 72 options minus 3 for nothing ontology selection method = 69 options
for demonstration_selection_method in DemonstrationSelectionMethod:
    for top_k in [0, 3]:
        for ontology_selection_method in OntologySelectionMethod:
            if ontology_selection_method == OntologySelectionMethod.Nothing:
                full_run(demonstration_selection_method, ontology_selection_method, OntologyFormat.XML, top_k, model)
                break
            for ontology_format in OntologyFormat:
                full_run(demonstration_selection_method, ontology_selection_method, ontology_format, top_k, model)