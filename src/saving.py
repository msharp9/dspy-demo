import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

dspy.settings.configure(lm=dspy.LM("ollama_chat/llama3.2:1b", api_base="http://localhost:11434", api_key=""))

gsm8k = GSM8K()
gsm8k_trainset = gsm8k.train[:1000]
dspy_program = dspy.ChainOfThought("question -> answer")

optimizer = dspy.BootstrapFewShot(metric=gsm8k_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5)
compiled_dspy_program = optimizer.compile(dspy_program, trainset=gsm8k_trainset)

compiled_dspy_program.save("./dspy_program2/program.json", save_program=False)

# loaded_dspy_program = dspy.ChainOfThought("question -> answer") # Recreate the same program.
# loaded_dspy_program.load("./dspy_program/program.json")

# print(compiled_dspy_program)
# print(loaded_dspy_program)

# # these asserts are failing
# assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
# for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
#     # Loaded demo is a dict, while the original demo is a dspy.Example.
#     assert original_demo.toDict() == loaded_demo
# assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)

compiled_dspy_program.save("./dspy_program3/", save_program=True)

loaded_dspy_program = dspy.load("./dspy_program3/")

assert len(compiled_dspy_program.predict.demos) == len(loaded_dspy_program.predict.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.predict.demos, loaded_dspy_program.predict.demos):
    # Loaded demo is a dict, while the original demo is a dspy.Example.
    assert original_demo.toDict() == loaded_demo.toDict()
assert str(compiled_dspy_program.predict.signature) == str(loaded_dspy_program.predict.signature)

result = loaded_dspy_program(
    question="If it takes 1 hour for 60 people to play an orchestra how long will it take for 600 people to play the same orchestra?"
)
print(result)
