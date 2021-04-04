#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os
os.chdir("drive/My Drive/NLP")


# In[3]:


get_ipython().system('wget https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv')


# In[4]:


import numpy as np
import pandas as pd
import itertools
import os

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
layers = keras.layers
models = keras.models
print("You have TensorFlow version", tf.__version__)
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


# In[ ]:


data = pd.read_csv("bbc-text.csv")


# In[6]:


data.head()


# In[7]:


data['category'].value_counts()


# In[8]:


train_size = int(len(data) * .8)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(data) - train_size))


# In[ ]:


data['text'].dropna(inplace=True)
data['text'] = [entry.lower() for entry in data['text']]


# In[10]:


import nltk
nltk.download("punkt")
data['text']= [word_tokenize(entry) for entry in data['text']]


# In[11]:


nltk.download("wordnet")
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


# In[ ]:


for index,entry in enumerate(data['text']):
  Final_words = []
  word_Lemmatized = WordNetLemmatizer()
  for word, tag in pos_tag(entry):
    if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
  data.loc[index,'text_final'] = str(Final_words)  


# In[13]:


data.head()


# In[ ]:


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(data['text_final'],data['category'],test_size=0.3)


# In[ ]:


Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


# In[16]:


Train_Y[:5]


# In[ ]:


Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(data['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


# In[18]:


print(Tfidf_vect.vocabulary_)


# In[19]:


SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

predictions_SVM = SVM.predict(Test_X_Tfidf)

print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)


# In[20]:


review = """this was a great fa cup game"""
review_vector = Tfidf_vect.transform([review]) 
print(SVM.predict(review_vector))


# In[23]:


x=Encoder.inverse_transform(SVM.predict(review_vector))
print (x)


# In[ ]:


import pandas as pd
from sklearn import model_selection, naive_bayes, svm
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
def pred():
    new_data = pd.read_csv("C:/Users/Aditya Joshi/Documents/Untitled Folder/bbc-text-final.csv")
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(new_data['text_final'],new_data['category'],test_size=0.3)
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(new_data['text_final'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    review=str(e1.get())
    review_vector = Tfidf_vect.transform([review])
    x = StringVar()
    x=(Encoder.inverse_transform(loaded_model.predict(review_vector)))
    print(x)
    Label(master,text=x[0],font="Lato 12 bold").grid(row=7,column=0,stick=W)
from tkinter import* 
import tkinter 
master=Tk() 
master.geometry("350x150")
master.title('BBC News Text Classification ')
Label(master,text="INPUT NEWS").grid(row=0, column=0) 
#e1=Entry(master)
#e1.grid(row=0,column=1)
text1=Text(master)
text1.place(relx=0.2,rely=0.0099,height=50,width=120)
Button(master,text="predict result",command=pred).grid(row=6,column=1,sticky=W,pady=4)
Button(master,text="exit",command=master.quit).grid(row=6,column=2,sticky=W,pady=4)
mainloop()


# In[ ]:




