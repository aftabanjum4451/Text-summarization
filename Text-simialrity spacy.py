# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

stopwords = list(STOP_WORDS
nlp = spacy.load('en_core_web_sm')

text = """
There are broadly two types of extractive summarization tasks depending on what the summarization program focuses on. The first is generic summarization, which focuses on obtaining a generic summary or abstract of the collection (whether documents, or sets of images, or videos, news stories etc.). The second is query relevant summarization, sometimes called query-based summarization, which summarizes objects specific to a query. Summarization systems are able to create both query relevant text summaries and generic machine-generated summaries depending on what the user needs.
An example of a summarization problem is document summarization, which attempts to automatically produce an abstract from a given document. Sometimes one might be interested in generating a summary from a single source document, while others can use multiple source documents (for example, a cluster of articles on the same topic). This problem is called multi-document summarization. A related application is summarizing news articles. Imagine a system, which automatically pulls together news articles on a given topic (from the web), and concisely represents the latest news as a summary.
Image collection summarization is another application example of automatic summarization. It consists in selecting a representative set of images from a larger set of images.[3] A summary in this context is useful to show the most representative images of results in an image collection exploration system. Video summarization is a related domain, where the system automatically creates a trailer of a long video. This also has applications in consumer or personal videos, where one might want to skip the boring or repetitive actions. Similarly, in surveillance videos, one would want to extract important and suspicious activity, while ignoring all the boring and redundant frames captured.
"""

doc = nlp(text)#creating token

tokens = [token.text for token in doc]
print(tokens)
# from the token we can see that our tokenz have '\n' so need to remove from the token
punctuation = punctuation + '\n'
print(punctuation)

'''
# word frequency 
Now we will make the word frequency table.
 It will contain the number of occurrences of all the distinct words 
 in the text which are not punctuations or stop words. We will create
 a dictionary named word_frequencies.
'''
word_frequencies = {}
for word in doc:
    if word.text.lower() not in stopwords:
        if word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1
                
print(word_frequencies)
#max frequecy 
max_frequency = max(word_frequencies.values())
print(max_frequency)

'''
We will divide each frequency value in word_frequencies with the max_frequency to normalize the frequencies.
'''

for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word]/max_frequency

print(word_frequencies)


'''Now we will do sentence tokenization. The entire text is divided into sentences.'''

sentence_tokens = [sent for sent in doc.sents]

for item in sentence_tokens:
    print('sentence number...........', item)

'''
Now we will calculate the sentence scores. The sentence score for a particular sentence is the sum of the normalized frequencies of the words in that sentence. All the sentences will be stored with their score in the dictionary sentence_scores.

'''

sentence_scores = {}
for sent in sentence_tokens:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores.keys():
                sentence_scores[sent] = word_frequencies[word.text.lower()]
            else:
                sentence_scores[sent] += word_frequencies[word.text.lower()]
                
print(sentence_scores)

'''
Now we are going to select 30% of the sentences having the largest scores. For this we are going to import nlargest from heapq. 
'''
from heapq import nlargest

'''
We want the length of summary to be 30% of the original length which is 4. Hence the summary will have 4 sentences.
'''

select_length = int(len(sentence_tokens)*0.3)
print(select_length)

'''
nlargest() will return a list with the select_length largest elements i.e. 4 largest elements from sentence_scores. key = sentence_scores.get specifies a function of one argument that is used to extract a comparison key from each element in sentence_scores.

'''
summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)
print(summary)

'''Now we will combine this sentence together and make final string which contains the summary.'''

final_summary = [word.text for word in summary]
summary = ' '.join(final_summary)

print(text)#original text

print(summary)# summarization










