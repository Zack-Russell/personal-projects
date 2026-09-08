from pathlib import Path
from operator import itemgetter
from textblob import TextBlob 
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# This ties into work in my master's program, though is rudementary compared to what might 
# be introduced in production enviornments

# The prelimary part here is just dedicated to some basic notes about the textblob library


''' ===== textblob technical overview ===== '''

# Textblob is based on NLTK (Natural Language Toolkit), a popular and open source python lib
    # NLTK works like this: 
    # raw text --> tokenization --> tokens --> POS tagging --> tagged tokens --> parsing/chunking --> syntax/parse tree

# It runs on punkt tokenizer, an unsupervised1 rule-acquisition model 
    # This doesn't rely on hardcoded rules only. It scans a corpus to learn abbreviations, 
    # collocations, and sentence-initial words via log-likelihood ratios.
    # But sentence boundaries are calced with standard deterministic regex

# POS tagging is done with brill tagging, creaed by Eric Brill the computational linguist in '95
    # Transformation-based learning
    # Initially, all words are tagged as the most likely POS based on lexical frequency, a baseline
    # Then, a rule is applied: e.g. change text T's tag T1 to T2 if condition C is met
    # The rules are established during training from a "golden corpus"

''' ===== TextBlob basics '''

text = 'Today is a beautiful day. Tomorrow looks like bad weather.'
blob = TextBlob(text)

# The sentences property splits the blob into a list of sentences
print(f"List of Sentences: {blob.sentences}\n")

# Words property splits it into a list of words
print(f"list of words: {blob.words}\n")

# Tags shows us a list of tuples, each containing the word and its part of speech tag
# This is done with a pattern tagger from the pattern library, referenced with TextBlob
print(f"List of tags: {blob.tags}\n")

# We can break it out into tokens using the built-in tokenizer in textblob
# This is relatively simple and clean. 
# GenAI implementation will tokenize in a less intuitive way, depending on the context!
print(f"Tokenized: {blob.tokens}\n")

# Noun phrases - larger groups of words that represent a single semantic idea

print(f"List of only noun phrases: {blob.noun_phrases}")


''' ===== TextBlob sentiment analysis '''

    # subjective rankings of sentences below
# "The show is terrible" - Strongly negative
# "The show is not good" - Weakly negative
# "The show is not bad" - Weakly positive
# "The show is amazing" - Strongly positive

# TextBlob sentiment analysis is a combination of polarity and subjectivity

# Get polarity of blob -- scale of -1 to 1

blob_polarity = blob.sentiment.polarity

print(f"Polarity of blob: {blob_polarity}\n") # Slightly positive polarity, probs bc we have both pos and neg ideas

# Get subjectivity of blob -- scale of 0 to 1

blob_subjectivity = blob.sentiment.subjectivity

print(f"Subjectivity of blob: {blob_subjectivity}\n") # more objective than subjective for this corpus

# Get sentiment of the blob -- combines the above 

sentiment = blob.sentiment

print(f"Sentiment of blob: {sentiment}\n")

# Now we can look over the sen1tences and analyze each individually
print("Sentiment of each sentence in blob:")
for sentence in blob.sentences:
    print(sentence.sentiment)
1

''' ===== TextBlob Frequency Analysis ===== '''

'''
try: 
    shakespeare_blob = TextBlob(Path("romeo_and_juliet.txt").read_text())
except Exception as error:
    print(error)
    raise ValueError

# word_counts is case incensitive, attribute that stores data in a dictionary
juliet = shakespeare_blob.word_counts["juliet"]
print(juliet)

romeo = shakespeare_blob.word_counts["romeo"]
print(romeo)

# These are their speaking parts
# .count is more flexible and allows for some customization, doesn't have to be performed on words
juliet_case = shakespeare_blob.words.count("JULIET", case_sensitive = True)
romeo_case = shakespeare_blob.words.count("ROMEO", case_sensitive = True)
'''



# Stopwords, common words that don't convey meaning
# We filter them out to stop them from overpowering more meaningful words in processing

# nltk.download('stopwords') -- gotta run this once
stop_words = stopwords.words("english")


word_list = [word for word in blob.words if word not in stop_words]
print(word_list)

# unique words contained in the blob. returned as a key value pair.
items = blob.word_counts.items()
# no stop words, pythonic
items = [item for item in items if item[0] not in stop_words]

