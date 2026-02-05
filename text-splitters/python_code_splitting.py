from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """
from langchain_text_splitters.base import (
    Language,
    TextSplitter,
    Tokenizer,
    TokenTextSplitter,
    split_text_on_tokens,
)
from langchain_text_splitters.character import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_text_splitters.html import (
    ElementType,
    HTMLHeaderTextSplitter,
    HTMLSectionSplitter,
    HTMLSemanticPreservingSplitter,
)
from langchain_text_splitters.json import RecursiveJsonSplitter
from langchain_text_splitters.jsx import JSFrameworkTextSplitter
from langchain_text_splitters.konlpy import KonlpyTextSplitter
from langchain_text_splitters.latex import LatexTextSplitter
from langchain_text_splitters.markdown import (
    ExperimentalMarkdownSyntaxTextSplitter,
    HeaderType,
    LineType,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
)
from langchain_text_splitters.nltk import NLTKTextSplitter
from langchain_text_splitters.python import PythonCodeTextSplitter
from langchain_text_splitters.sentence_transformers import (
    SentenceTransformersTokenTextSplitter,
)
"""
splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0
    )
docs = splitter.split_text(text)
print(docs)
print(len(docs))