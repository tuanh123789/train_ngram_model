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
        line=line.split('.')
        for sentence in line:
            sentence=re.sub(r'[().,:;/%$-@!*&^?><_#+]',' ',sentence)
            sentence=sentence.replace('"',' ')
            sentence=sentence.replace("'"," ")
            sentence,pos_tag=ViPosTagger.postagging(sentence)
            remove=[index for (index,pos) in enumerate(pos_tag) if pos in ['M','X','Np']]
            sentence=[sentence[index] for index in range(len(sentence)) if index not in remove]
            sentence=[word.lower().rstrip('\n') for word in sentence if word != '']
            if len(sentence) > 1:
                train_data.append(sentence)
    return train_data

corpus_dir = sys.argv[1]
#model_path=sys.argv[2]


if __name__=="__main__":
    data=open(corpus_dir,'r',encoding='utf-8').readlines()
    train_data=load_data(data)
    train_data,vocab=padded_everygram_pipeline(3,train_data)
    model=MLE(3)
    model.fit(train_data,vocab)
    pickle.dump(model,open('model.pickle','wb'))
