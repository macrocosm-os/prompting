from collections import Counter

import nltk
from nltk.corpus import brown


def load_most_common_words(n=25000):
    # Download necessary NLTK data
    nltk.download("brown", quiet=True)

    # Get all words from the Brown corpus
    words = brown.words()

    # Convert to lowercase and count frequencies
    word_freq = Counter(word.lower() for word in words)

    # Get the n most common words
    most_common = word_freq.most_common(n)

    # Extract just the words (without frequencies)
    common_words = [word for word, _ in most_common]

    return common_words


ENGLISH_WORDS = load_most_common_words()
