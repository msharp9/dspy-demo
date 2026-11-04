from typing import Literal

import dspy


def main():
    lm = dspy.LM("ollama_chat/llama3.2:1b", api_base="http://localhost:11434", api_key="")
    dspy.configure(lm=lm)
    # Math
    math = dspy.ChainOfThought("question -> answer")
    response = math(question="Two dice are tossed. What is the probability that the sum equals two?")
    print(response)

    # Sentiment Analysis
    classify = dspy.Predict(Classify)
    response = classify(sentence="This book was super fun to read, though not the last chapter.")
    print(response)

    # Tools - throws pydantic error
    # react = dspy.ReAct("question -> answer", tools=[evaluate_math, search_wikipedia])
    # pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
    # print(pred)

    # Pipeline
    draft_article = DraftArticle()
    article = draft_article(topic="World Cup 2002")
    print(article)


def evaluate_math(expression: str):
    return dspy.PythonInterpreter({}).execute(expression)


def search_wikipedia(query: str):
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
    return [x["text"] for x in results]


class Classify(dspy.Signature):
    """Classify sentiment of a given sentence."""

    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
    confidence: float = dspy.OutputField()


class Outline(dspy.Signature):
    """Outline a thorough overview of a topic."""

    topic: str = dspy.InputField()
    title: str = dspy.OutputField()
    sections: list[str] = dspy.OutputField()
    section_subheadings: dict[str, list[str]] = dspy.OutputField(desc="mapping from section headings to subheadings")


class DraftSection(dspy.Signature):
    """Draft a top-level section of an article."""

    topic: str = dspy.InputField()
    section_heading: str = dspy.InputField()
    section_subheadings: list[str] = dspy.InputField()
    content: str = dspy.OutputField(desc="markdown-formatted section")


class DraftArticle(dspy.Module):
    def __init__(self):
        self.build_outline = dspy.ChainOfThought(Outline)
        self.draft_section = dspy.ChainOfThought(DraftSection)

    def forward(self, topic):
        outline = self.build_outline(topic=topic)
        sections = []
        for heading, subheadings in outline.section_subheadings.items():
            section, subheadings = f"## {heading}", [f"### {subheading}" for subheading in subheadings]
            section = self.draft_section(topic=outline.title, section_heading=section, section_subheadings=subheadings)
            sections.append(section.content)
        return dspy.Prediction(title=outline.title, sections=sections)


if __name__ == "__main__":
    main()
