#install pyvi, nltk, gensim

import sys
import nltk
from pyvi import ViPosTagger
import re
import os
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import KneserNeyInterpolated
from nltk.lm import MLE
import pickle

def load_data(data):
    train_data=[]   
    for line in data:
        line=line.replace(',','.')
        line=line.split('.')
        for sentence in line:
            sentence=re.sub(r'[():;/%$-@!*&^?><_#+]',' ',sentence)
            sentence=sentence.replace('"',' ')
            sentence=sentence.replace("'"," ")
            sentence,pos_tag=ViPosTagger.postagging(sentence)
            for index,pos in enumerate(pos_tag):
                if pos == 'Np':
                    sentence[index]='Np'
                if pos == 'Nc':
                    sentence[index]='Nc'
                if pos == 'X':
                    sentence[index]='X'
                if pos == 'Ny':
                    sentence[index]='Ny'
                if pos == 'M':
                    sentence[index]='M'
            sentence=[word.lower().rstrip('\n') for word in sentence if word != '']
            if len(sentence) > 1:
                train_data.append(sentence)
    return train_data

corpus_dir = sys.argv[1]
model_name=sys.argv[2]
n=sys.argv[3]


if __name__=="__main__":
    data=open(corpus_dir,'r',encoding='utf-8').readlines()
    train_data=load_data(data)
    train_data,vocab=padded_everygram_pipeline(int(n),train_data)
    model=MLE(int(n))
    model.fit(train_data,vocab)
    pickle.dump(model,open(model_name,'wb'))
