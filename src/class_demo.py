import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

dspy.settings.configure(lm=dspy.LM("ollama_chat/llama3.2:1b", api_base="http://localhost:11434", api_key=""))

gsm8k = GSM8K()
gsm8k_trainset = gsm8k.train[:1000]
dspy_program = dspy.ChainOfThought("question -> answer")

optimizer = dspy.BootstrapFewShot(metric=gsm8k_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=10)
compiled_dspy_program = optimizer.compile(dspy_program, trainset=gsm8k_trainset)

compiled_dspy_program.save("./dspy_demo/", save_program=True)
compiled_dspy_program.save("./dspy_demo/program.json", save_program=False)  # save the program.json for reference

loaded_dspy_program = dspy.load("./dspy_demo/")

result = loaded_dspy_program(
    question="If it takes 1 hour for 60 people to play an orchestra how long will it take for 600 people to play the same orchestra?"
)
print(result)

result = loaded_dspy_program(
    question="Dan has 2 quarters, 3 dimes, 3 nickels, and 4 pennies. Does he have enough money to buy a pen for 75 cents?"
)
print(result)
