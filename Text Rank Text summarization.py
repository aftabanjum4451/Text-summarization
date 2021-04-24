#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install pytextrank')


# In[10]:


get_ipython().system('python -m spacy download en_core_web_lg ')


# In[1]:


import spacy
nlp = spacy.load("en_core_web_lg")


# In[2]:


get_ipython().system('python -m spacy download en_core_web_md ')


# In[1]:



import spacy
import pytextrank

import spacy
nlp = spacy.load("en_core_web_md")

tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name='textrank', last=True)


# In[9]:


text = '''Ferrari is an Italian luxury sports car manufacturer based in Maranello, Italy.
Founded by Enzo Ferrari in 1939 out of the Alfa Romeo race division as Auto Avio Costruzioni, the company built its first car in 1940. However, the company's inception as an auto manufacturer is usually recognized as 1947, when the first Ferrari-badged car was completed
Enzo Ferrari was not initially interested in the idea of producing road cars when he formed 
Scuderia Ferrari in 1929, with headquarters in Modena. Scuderia Ferrari (pronounced [skudeˈriːa]) literally means
fielded Alfa Romeo racing cars for gentleman drivers,functioning as the racing division of Alfa Romeo. 
In 1933, Alfa Romeo withdrew its in-house racing team and Scuderia Ferrari took over as its works team:[1]
the Scuderia received Alfa's Grand Prix cars of the latest specifications and fielded many famous drivers such as Tazio Nuvolari and Achille Varzi.
In 1938, Alfa Romeo brought its racing operation again in-house, forming Alfa Corse in Milan and hired Enzo Ferrari as manager of the new racing department; therefore the Scuderia Ferrari was disbanded.\
[1]In September 1939, Ferrari left Alfa Romeo under the provision he would not use the Ferrari name in association with races or racing cars for at least four years.[1] 
A few days later he founded Auto Avio Costruzioni, headquartered in the facilities of the old Scuderia Ferrari.[1]'''


# In[ ]:


doc = nlp(text)

# examine the top-ranked phrases in the document
for p in doc._.phrases:
    print('{:.4f} {:5d}  {}'.format(p.rank, p.count, p.text))
    print(p.chunks)


# In[12]:


for sent in doc._.textrank.summary(limit_phrases=15, limit_sentences=3):
    print(sent)


# In[ ]:





# **text** summaruzation by gensim.***summarization** 
# 

# In[14]:



from gensim.summarization import summarize


# In[15]:



print(summarize(text))


# In[16]:


print(summarize(text, split=True))


# In[17]:


print(summarize(text, ratio=0.5))


# In[18]:


print(summarize(text, word_count=100))


# In[19]:


from gensim.summarization import keywords
print(keywords(text))


# In[ ]:




