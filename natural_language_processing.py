import pandas as pd
import numpy as np
text = """"
A Scandal in Bohemia! 01
The Red-headed League,2
A Case, of Identity 33
The Boscombe Valley Mystery4
The Five Orange Pips1
The Man with? Twisted Lip
The Adventure of the Blue Carbuncle
The Adventure of the Speckled Band
The Adventure of the Engineer's Thumb
The Adventure of the Noble Bachelor
The Adventure of the Beryl Coronet
The Adventure of the Copper Beeches"""

text.split("\n")

v_text = text.split("\n")

v = pd.Series(v_text)

v

test_vector = v[1:len(v)]

test_vector

mdf = pd.DataFrame(test_vector, columns=["novels"])

mdf

d_mdf = mdf.copy()

list1 = [1,2,3]

str1 = " ".join(str(i) for i in list1)

d_mdf["novels"].apply(lambda x: " ".join(x.lower() for x in x.split()))

d_mdf = pd.DataFrame(d_mdf, columns=["novels"])

d_mdf = d_mdf["novels"].str.replace("[^\w\s]", "")

d_mdf

d_mdf = d_mdf.str.replace("\d","")

d_mdf = pd.DataFrame(d_mdf, columns=["novels"])

d_mdf

import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords

sw = stopwords.words("english")

sw

f = d_mdf["novels"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

f

type(f)

f = pd.DataFrame(f, columns=["novels"])

type(f)

pd.Series(" ".join(f["novels"]).split()).value_counts()

delete = pd.Series(" ".join(f["novels"]).split()).value_counts()[-3:]

delete

f["novels"].apply(lambda x: " ".join(i for i in x.split() if i not in delete))

# Stemming
nltk.download("punkt")

import textblob

from textblob import TextBlob

TextBlob(f["novels"][1]).words

f["novels"].apply(lambda x: TextBlob(x).words)

from nltk.stem import PorterStemmer

st = PorterStemmer()

f["novels"].apply(lambda x: " ".join([st.stem(i) for i in x.split()]))

#Lemmatization

from textblob import Word

nltk.download("wordnet")
nltk.download('omw-1.4')

f["novels"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

f["novels"][0:5]

d_mdf["novels"][0:5]

## NLP Uygulamaları ##

## N-Gram ##

a = """Bu örneği anlaşılabilmesi için daha uzun bir metin üzerinden göstereceğim. 
N-gram'lar birlikte kullanılan kelimelerin kombinasyonlarını gösterir"""

TextBlob(a).ngrams(1)

# N-Gram is used to display word combinations. It is one of the techniques that works for us in terms of variable engineering.

## Part of Speech Tagging (POS) ##

# This process examines whether the words in the text are nouns, adjectives and adverbs.

nltk.download("averaged_perceptron_tagger")

TextBlob(f["novels"][2]).tags

f["novels"].apply(lambda x: TextBlob(x).tags)

# As seen here, words are classified according to adjective and adverb situations.
# Descriptions were made in terms of grammar.

## Chunking(Shallow Parsing) ##

# We use the Chucking method to show the words such as nouns, adjectives and adverbs that we have removed from the text on a diagram.

pos = f["novels"].apply(lambda x: TextBlob(x).tags)

pos

sentence = """R and Python are useful data science tools for the new or old data scientists who eager to do efficent data science task"""

pos = TextBlob(sentence).tags

pos

reg_exp = "NP: {<DT>?<JJ>*<NN>}"

rp = nltk.RegexpParser(reg_exp)

ends = rp.parse(pos)

print(ends)

ends.draw()

# Chunking method can be used if there is a purpose such as how many verbs, adjectives and nouns are used in a study
# carried out according to its place.

## Named Entity Recognition ##

# Performs identification operations in the given texts. We use this method if we want to know the definitions
# if we know what the words are in a structural sense.

from nltk import word_tokenize, pos_tag, ne_chunk

nltk.download("maxent_ne_chunker")

nltk.download("words")

sentence_two = "Hadley is creative people who work for R Studio AND he attented conferance at Newyork last year"

print(ne_chunk(pos_tag(word_tokenize(sentence_two))))

# Mathematical Operations and Simple Feature Extraction

# Number of Letters/Characters
ff = f.copy()
ff["novels"].str.len()

ff["num_letters"] = ff["novels"].str.len()

ff

# Word Count

a = "scandal in a bohemia"

a.split()

len(a.split())

ff.iloc[0:1,0:1]

ff.apply(lambda x: len(str(x).split(" ")))

ff["num_words"] = ff["novels"].apply(lambda x: len(str(x).split(" ")))

ff

# Capturing and Counting Special Characters

ff["num_of_spec_char"] = ff["novels"].apply(lambda x: len([x for x in x.split() if x.startswith("Adventure")]))

ff

# Capturing and Counting Numbers

mdf["novels"].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

ff["num_of_count"] = mdf["novels"].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

ff

## Text Visualization ##

import pandas as pd

data = pd.read_csv("/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_11/train.tsv", sep="\t")

data.head()

data.info()

# Büyük - Küçük Dönüşümü

data["Phrase"].apply(lambda x: " ".join(x.lower() for x in x.split()))

data["Phrase"] = data["Phrase"].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Noktalama İşaretleri

data["Phrase"].str.replace('[^\w\s]', '')

data["Phrase"] = data["Phrase"].str.replace('[^\w\s]', '')

data["Phrase"].str.replace('\d', '')

data["Phrase"] = data["Phrase"].str.replace('\d', '')

# Stopwords

import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords

sw = stopwords.words('english')

data["Phrase"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

data["Phrase"] = data["Phrase"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

# Seyreklerin Silinmesi

pd.Series(' '.join(data["Phrase"]).split()).value_counts()[-1000:]

delete = pd.Series(' '.join(data["Phrase"]).split()).value_counts()[-1000:]

data["Phrase"] = data["Phrase"].apply(lambda x: " ".join(x for x in x.split() if x not in delete))

data["Phrase"].head(10)

#Lemmatization

from textblob import Word

nltk.download("wordnet")

data["Phrase"] = data["Phrase"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

data["Phrase"].head(10)

# Terim Frekansı

tf1 = (data["Phrase"]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf1.columns = ["words", "tf"]

tf1.head()

tf1.info()

tf1.nunique()

import matplotlib.pyplot as plt

a = tf1[tf1["tf"] > 1000]

a.plot.bar(x = "words", y="tf")

plt.show()


## Word Cloud ##

!pip install wordcloud

import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_11/train.tsv", sep="\t")
data.head()
text = data["Phrase"][0]

wordcloud = WordCloud().generate(text)

plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.show()

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("kelime_bulutu.png");

# Tüm Metin

text = " ".join(i for i in data.Phrase)

text

wordcloud = WordCloud(max_font_size=50, background_color="white").generate(text)
plt.figure(figsize= [10,10])
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

## Şablonlara Göre Word Cloud ##

nlp_mask = np.array(Image.open("/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_11/nlp.png"))

nlp_mask

wc = WordCloud(background_color="white",
                      max_words=1000,
                      mask= nlp_mask,
                      contour_width=3,
                      contour_color="firebrick").generate(text)
wc.generate(text)
wc.to_file("nlp_2.png")
plt.figure(figsize= [10,10])
plt.imshow(wc)
plt.axis("off")
plt.show()

## Sentiment Analizi ve Sınıflandırma Modelleri ##

from textblob import TextBlob
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

import pandas as pd
data = pd.read_csv("/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_11/train.tsv", sep = "\t")

data.head()

