from ingestion.cleaner import Cleaner
from langchain_core.documents import Document


def test_cleaner_removes_private_use_area_characters():
    cleaner = Cleaner()

    raw_text = (
        "This document contains o strange  characters "
        "that come from PDF fonts."
    )

    doc = Document(
        page_content=raw_text,
        metadata={"source": "test.pdf", "page": 3},
    )

    cleaned = cleaner.clean([doc])

    cleaned_text = cleaned[0].page_content

    # PUA characters should be removed or replaced
    assert "" not in cleaned_text
    assert "" not in cleaned_text
    assert "" not in cleaned_text

    # Core semantic text should remain
    assert "document contains" in cleaned_text
    assert "PDF fonts" in cleaned_text

    # Metadata must be preserved
    assert cleaned[0].metadata == {"source": "test.pdf", "page": 3}


def test_cleaner_fixes_hyphenation_across_line_breaks():
    cleaner = Cleaner()

    raw_text = (
        "This text talks about li-\n"
        "fe science and bio-\n"
        "technology advances."
    )

    doc = Document(page_content=raw_text, metadata={})

    cleaned = cleaner.clean([doc])
    cleaned_text = cleaned[0].page_content

    assert "life science" in cleaned_text
    assert "biotechnology" in cleaned_text
    assert "-\n" not in cleaned_text


def test_cleaner_normalizes_whitespace_and_newlines():
    cleaner = Cleaner()

    raw_text = "Line one.\n\n\n\nLine   two.\t\tLine three."

    doc = Document(page_content=raw_text, metadata={})

    cleaned = cleaner.clean([doc])
    cleaned_text = cleaned[0].page_content

    assert cleaned_text == "Line one.\n\nLine two. Line three."


def test_cleaner_normalizes_unicode_characters():
    cleaner = Cleaner()

    # Full-width characters + compatibility forms
    raw_text = "ＮＴＴ ＤＡＴＡ uses Unicode."

    doc = Document(page_content=raw_text, metadata={})

    cleaned = cleaner.clean([doc])

    assert cleaned[0].page_content == "NTT DATA uses Unicode."


def test_cleaner_does_not_mutate_input_documents():
    cleaner = Cleaner()

    original = Document(
        page_content="Text with   spaces and  glyphs.",
        metadata={"source": "original.pdf"},
    )

    cleaned = cleaner.clean([original])

    # Original must remain unchanged
    assert original.page_content == "Text with   spaces and  glyphs."

    # Cleaned version should differ
    assert cleaned[0].page_content != original.page_content
    assert cleaned[0].metadata == original.metadata


def test_cleaner_handles_empty_input_gracefully():
    cleaner = Cleaner()

    cleaned = cleaner.clean([])

    assert cleaned == []
