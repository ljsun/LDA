# encoding = utf-8

from HLDA import HLDA
from LDA import LDA
from Data_Process import data_process

LDA_texts, HLDA_texts, unique_word = data_process()

lda = LDA(iterations=1000, burnIn=100, sampleLag=10, allseg=LDA_texts, uniqueSeg=unique_word)
lda.gibbs(K=4, alpha=12.5, beta=0.01)

hlda = HLDA(iterations=1000, burnIn=100, sampleLag=10, allseg=HLDA_texts, uniqueSeg=unique_word)
hlda.gibbs(K=4, alpha=12.5, beta=0.01)

# LDA Model
print('LDA Model~~~')
phi = lda.get_phi()
for topic in phi.keys():
    Topic_word = sorted(phi[topic].items(), key=lambda abs: abs[1], reverse=True)[0:5]
    print('topic'+str(topic))
    for topic_word in Topic_word:
        print(topic_word[0] + '：'+str(topic_word[1]))

# HLDA Model
print('HLDA Model~~~')
phi = hlda.get_phi()
for topic in phi.keys():
    Topic_word = sorted(phi[topic].items(), key=lambda abs: abs[1], reverse=True)[0:5]
    print('topic'+str(topic))
    for topic_word in Topic_word:
        print(topic_word[0] + '：'+str(topic_word[1]))
