import nltk
from nltk.corpus import words


nltk.download("words")
ENGLISH_WORDS: list[str] = words.words()
