from typing import Literal

import dspy
import httpx

from config import local_lm


def main(lm=None):
    dspy.configure(lm=lm or local_lm())
    # Math
    math = dspy.ChainOfThought("question -> answer")
    response = math(question="Two dice are tossed. What is the probability that the sum equals two?")
    print(response)

    # Sentiment Analysis
    classify = dspy.Predict(Classify)
    response = classify(sentence="This book was super fun to read, though not the last chapter.")
    print(response)

    # Tools: ReAct lets the model call Python functions in a think -> act -> observe loop.
    # Expected: born 1625, so ~5761.3. Small models often grab the wrong David Gregory; try a bigger model.
    react = dspy.ReAct("question -> answer: float", tools=[search_wikipedia, divide])
    pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
    print(pred)

    # Pipeline
    draft_article = DraftArticle()
    article = draft_article(topic="World Cup 2002")
    print(article)


def divide(a: float, b: float) -> float:
    """Divide a by b."""
    return a / b


def search_wikipedia(query: str) -> list[str]:
    """Search Wikipedia and return short summaries of the top results."""
    response = httpx.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": 3,
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "exsentences": 3,
        },
        headers={"User-Agent": "dspy-demo/0.1 (https://github.com/msharp9/dspy-demo)"},
        timeout=10,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {}).values()
    return [f"{page['title']}: {page.get('extract', '')}" for page in pages]


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
