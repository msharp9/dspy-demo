# from dspy.datasets import HotPotQA
import random
from typing import Literal

import dspy
from datasets import load_dataset
from dspy.datasets import DataLoader

# Load the Banking77 dataset.
CLASSES = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features["label"].names
kwargs = {"fields": ("text", "label"), "input_keys": ("text",), "split": "train", "trust_remote_code": True}

# Load the first 2000 examples from the dataset, and assign a hint to each *training* example.
trainset = [
    dspy.Example(x, hint=CLASSES[x.label], label=CLASSES[x.label]).with_inputs("text", "hint")
    for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[:2000]
]
random.Random(0).shuffle(trainset)

dspy.configure(lm=dspy.LM("ollama_chat/llama3.2:1b", api_base="http://localhost:11434", api_key=""))
# trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train]


# # This is suppose to run but I get an error
# def search_wikipedia(query: str) -> list[str]:
#     results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
#     return [x['text'] for x in results]

# react = dspy.ReAct("question -> answer", tools=[search_wikipedia])

# tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
# optimized_react = tp.compile(react, trainset=trainset)

# # need to define a Search for this example to work
# class RAG(dspy.Module):
#     def __init__(self, num_docs=5):
#         self.num_docs = num_docs
#         self.respond = dspy.ChainOfThought('context, question -> response')

#     def forward(self, question):
#         context = search(question, k=self.num_docs)   # defined in tutorial linked below
#         return self.respond(context=context, question=question)

# tp = dspy.MIPROv2(metric=dspy.evaluate.SemanticF1(decompositional=True), auto="medium", num_threads=24)
# optimized_rag = tp.compile(RAG(), trainset=trainset, max_bootstrapped_demos=2, max_labeled_demos=2)


# Define the DSPy module for classification. It will use the hint at training time, if available.
signature = dspy.Signature("text -> label").with_updated_fields("label", type_=Literal[tuple(CLASSES)])
classify = dspy.ChainOfThoughtWithHint(signature)

# Optimize via BootstrapFinetune.
optimizer = dspy.BootstrapFinetune(metric=(lambda x, y, trace=None: x.label == y.label), num_threads=24)
optimized = optimizer.compile(classify, trainset=trainset)

optimized(text="What does a pending cash withdrawal mean?")
