## Marks
The more information we put in the prompt the less accurate the answer atm
-> better prompt formatting
-> Group prompts by shared prefix/instruction format so the inference server can reuse the prefill KV cache and reduce latency/cost -> Just sort by prompt AND make sure first parts of prompt change the least

We need -IT (instruction) models because we don't do any fine-tuning ourselves

We can use models that are on huggingface:
llama / gemma

Works 100%:
- google/gemma-3-12b-it
- meta-llama/Llama-3.2-3B-Instruct

Doesn't work:
- models above like 20b
- gemma 4 (too new)

We need to make pre-processing more smooth (permanently add the duplication removal)

Does top_k mean the max amount of relevant sentences (even though one sentence can have multiple aspects) or does it mean max amount of sentence/aspect pairings (in this case, we could get 3 same sentences with 3 aspects)
-> needs to be formatted correctly into the prompt as well