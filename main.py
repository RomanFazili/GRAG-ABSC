import asyncio
from dataclasses import dataclass
from openai import AsyncOpenAI
from prompt_builder import PromptBuilder
from enums import DemonstrationSelectionMethod, OntologySelectionMethod, OntologyFormat, Polarity
from sentence_retriever import SentenceRetriever
from data_set import DataSet
from data_set_ontology import DataSetOntology
from ontology_retriever import OntologyRetriever
import os
from dotenv import load_dotenv
import json

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
demonstration_selection_method = DemonstrationSelectionMethod.SimCSE
ontology_selection_method = OntologySelectionMethod.Nothing
ontology_format = OntologyFormat.XML
top_k = 3
model = "meta-llama/Llama-3.2-3B-Instruct"

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

async def run(job: Job):

    response = await client.chat.completions.create(
        model=job.model,
        messages=[{"role":"user","content":job.prompt}],
        max_tokens=1
    )

    answer = response.choices[0].message.content
    print(answer)
    job.llm_output = answer
    return job

async def main():
    tasks = [run(job) for job in jobs]
    return await asyncio.gather(*tasks)

results = asyncio.run(main())

# Save the results to a JSONL file
with open("results.jsonl", "w") as f:
    for job in results:
        f.write(json.dumps(job.as_dict()) + "\n")


print({job.llm_output for job in jobs})
print(sum(1 for job in jobs if job.llm_output.lower() == job.true_polarity.value.lower()))
print(len(jobs))