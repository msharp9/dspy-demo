import dspy

from main import Classify, DraftArticle, DraftSection, Outline


def test_classify_signature_fields():
    assert set(Classify.input_fields) == {"sentence"}
    assert set(Classify.output_fields) == {"sentiment", "confidence"}


def test_article_signatures():
    assert set(Outline.output_fields) == {"title", "sections", "section_subheadings"}
    assert set(DraftSection.input_fields) == {"topic", "section_heading", "section_subheadings"}


def test_draft_article_is_module():
    article = DraftArticle()
    assert isinstance(article, dspy.Module)
    assert len(article.predictors()) == 2
