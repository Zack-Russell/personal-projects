import matplotlib.pyplot as plt
from pathlib import Path
import textblob
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords # adding this as it's needed for the stopwords portion
from operator import itemgetter # need this for sorting efficiency
import pandas as pd
import sys
import unicodedata # for the punctuation removal problem


# Assignment 3
'''For this assignment, complete the following tasks. 
You should use the code below as a starting point. 
Make changes where you see elipses (`...`), 
and feel free to add any other code as needed. 
It's necessary to add comments as you go too!'''

'''For this problem, we are going to analyze the text of _Frankenstein_ 
by Mary Wollstonecraft Shelley. The text of the book is available in 
the assignment'''

## Part (a)
#Read in the data, and create a `TextBlob` object with the words.

# Defines the directory in which this python file lives so we can target the txt file in the same one
SCRIPT_DIR = Path(__file__).resolve().parent
# Targets the file regardless of what the cwd is
file_path = SCRIPT_DIR / "frankenstein.txt"

# error handling for file ingestion
try:
    blob = TextBlob(file_path.read_text(encoding='utf-8')) # without specifying the encoding, codec fails to decode specific bytes
except FileNotFoundError as e: # file not found explicit exception handling
    print(f"File missing: {e}")
    raise  
except Exception as e: # otherwise raise a value error
    print(e)
    raise(ValueError)

## Part (b)
'''Make a dictionary of the frequency of words in the text. Your 
dictionary should consist of key-value pairs where the keys are 
the words in the text, and values are the number of times that 
word appears.'''

items = blob.word_counts.items() # kv pair

## Part (c)
'''Remove the most commonly used English words from this list 
(the "stop words").'''

# this is a one time download that needs to happen on local machine
nltk.download('stopwords', quiet = True) # if it's already there, this will queitly pass
stops = stopwords.words("english")

items = [item for item in items if item[0] not in stops] # exclusion of stop words

## Part (d)
'''Find the 25 most-said words in _Frankenstein_. Be careful to make 
sure they are actually words, and not punctuation marks or any 
other weird things!'''

# Since this is the standard libarary's up to date unicode 8 punctation list, it ensures that we don't have any punc
# Better approach than just brute forcing a list of what's already in the top group, imo
utf8_punctuation = [
    char for char in (chr(i) for i in range(sys.maxunicode + 1)) # iterate over every character once
    if unicodedata.category(char).startswith('P') # and keep only what is considered "punctuation", including all subgroups
]

sorted_items = sorted(items, key=itemgetter(1), reverse=True) # using the sorted command to sort them by most common
top_items = [item for item in sorted_items if item[0] not in utf8_punctuation] # keep only what's in the punc list
top_items = top_items[:25] # slice to take first 25 of sorted list

## Part (e)
'''Make a bar-chart plot of these 25 most-said words, where each 
word is a different bar and the height of the bar is the number 
of times a particular word is said.'''

'''_Hint_: it might be helpful to make a Pandas `DataFrame` object 
from the word counts, and use a plotting method on the `DataFrame` 
to generate this plot.'''

df = pd.DataFrame(top_items, columns = ["word", "count"]) # define basic 2d, 2 column dataframe 
print(df) # display it since need to answer part d

# time for some graphing
axes = df.plot.bar(x = "word", y = "count", legend = False) # basic dataframe bar chart, no legends
plt.show() # call matplotlib to show what just was generated (necessary in vscode at least)