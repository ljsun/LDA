# -*- coding:utf-8 -*-

import random

from Data_Process import data_process


class LDA:

    def __init__(self, iterations, burnIn, sampleLag, allseg, uniqueSeg):
        """
        :param iterations:
        :param burnIn:
        :param sampleLag:
        :param allseg:
        {doc: [word1, word2, word3...]}
        :param uniqueSeg:
        uniqueSeg = set(word1, word2, word3...)
        """
        self.numstats = 0
        self.ITERATIONS = iterations
        self.BURN_IN = burnIn
        # 采样间隔
        self.SAMPLE_LAG = sampleLag
        # 语料库
        self.allseg = allseg
        # 词袋库
        self.uniqueSeg = uniqueSeg

        self.V = len(self.uniqueSeg)

        self.thetasum = {}
        #
        self.phisum = {}
        # 统计次数
        self.numstats = 0
        # 每一个word被赋予每一个topic的次数
        self.nw = {}
        # 每一个doc中被赋予每一个topic的次数
        self.nd = {}
        # 被赋予每一个topic的word的个数
        self.nwsum = {}
        # 每一个doc的word的数目
        self.ndsum = {}
        # 每一个doc中的每一个word所对应的topic
        self.z = {}

    def initial_state(self, K):

        print(u'初始化gibbs采样中所需要的计数值')
        # 先把所有对应的值初始化为0
        # 初始化nw
        for word in self.uniqueSeg:
            self.nw[word] = {}
            for k in range(K):
                self.nw[word][k] = 0
        # 初始化nd
        for key in self.allseg.keys():
            self.nd[key] = {}
            for k in range(K):
                self.nd[key][k] = 0
        # 初始化nwsum
        for k in range(K):
            self.nwsum[k] = 0
        for key in self.allseg.keys():
            # 统计ndsum
            self.ndsum[key] = len(self.allseg[key])
            for word in self.allseg[key]:
                # 为每一个word先随机生成一个topic
                topic = random.randint(0, K-1)
                if key in self.z:
                    self.z[key].append(topic)
                else:
                    self.z[key] = [topic]
                # 统计nw
                self.nw[word][topic] += 1
                # 统计nd
                self.nd[key][topic] += 1
                # 统计nwsum
                self.nwsum[topic] += 1

    def sample_full_conditional(self, key, n):
        """
        sample from p(z_i|z_-i, w), i is (key, n)
        :param key: the num of doc
        :param n: the num in doc
        :return: sample_topic
        """
        topic = self.z[key][n]
        # remove all initialize value corresponding with the word
        self.nw[self.allseg[key][n]][topic] -= 1
        self.nd[key][topic] -= 1
        self.nwsum[topic] -= 1
        self.ndsum[key] -= 1
        # calculate the probability of every word
        p = []
        for k in range(self.K):
            p.append((self.nw[self.allseg[key][n]][k] + self.beta) / (self.nwsum[k] + self.V * self.beta)
                     * (self.nd[key][k] + self.alpha) / (self.ndsum[key] + self.K * self.alpha))
        # random choose depend on probability
        for k in range(1, len(p)):
            p[k] += p[k-1]
        u = random.uniform(0, 1) * p[k]
        for topic in range(len(p)):
            if u < p[topic]:
                break
        # add all initialize value corresponding with the word
        self.nw[self.allseg[key][n]][topic] += 1
        self.nd[key][topic] += 1
        self.nwsum[topic] += 1
        self.ndsum[key] += 1
        return topic

    def update_params(self):
        for key in self.allseg.keys():
            if key not in self.thetasum:
                self.thetasum[key] = {}
            for k in range(self.K):
                if k in self.thetasum[key]:
                    self.thetasum[key][k] += (self.nd[key][k] + self.alpha) / (self.ndsum[key] + self.K * self.alpha)
                else:
                    self.thetasum[key][k] = (self.nd[key][k] + self.alpha) / (self.ndsum[key] + self.K * self.alpha)

        for k in range(self.K):
            if k not in self.phisum:
                self.phisum[k] = {}
            for w in self.uniqueSeg:
                if w in self.phisum[k]:
                    self.phisum[k][w] += (self.nw[w][k] + self.beta) / (self.nwsum[k] + self.V * self.beta)
                else:
                    self.phisum[k][w] = (self.nw[w][k] + self.beta) / (self.nwsum[k] + self.V * self.beta)
        self.numstats += 1

    def gibbs(self, K, alpha, beta):
        """
        :param K: num of topic
        :param alpha: initial parameter of Dirichlet
        :param beta: initial parameter of Dirichlet
        :return:
        """
        self.K = K
        self.alpha = alpha
        self.beta = beta
        # Initialize
        self.initial_state(K)
        print('Sampling ' + str(self.ITERATIONS)+ " iterations with burn-in of " + str(self.BURN_IN))
        for i in range(self.ITERATIONS):
            # for all z_i
            print('第'+str(i+1)+'次迭代采样')
            for key in self.z.keys():
                for n in range(len(self.z[key])):
                    # sample from p(z_i|z_-i, w)
                    topic = self.sample_full_conditional(key, n)
                    self.z[key][n] = topic
            # SAMPLE
            if i>self.BURN_IN and self.SAMPLE_LAG > 0 and i % self.SAMPLE_LAG == 0:
                # print '第'+str(self.numstats+1)+'次参数更新'
                self.update_params()

    def get_theta(self):
        theta = {}
        if self.numstats > 0:
            for key in self.allseg.keys():
                theta[key] = {}
                for k in range(self.K):
                    theta[key][k] = self.thetasum[key][k] / self.numstats
        else:
            for key in self.allseg.keys():
                theta[key] = {}
                for k in range(self.K):
                    theta[key][k] = (self.nd[key][k] + self.alpha) / (self.ndsum[key] + self.K * self.alpha)
        return theta

    def get_phi(self):
        phi = {}
        if self.numstats > 0:
            for k in range(self.K):
                phi[k] = {}
                for w in self.uniqueSeg:
                    phi[k][w] = self.phisum[k][w] / self.numstats
        else:
            for k in range(self.K):
                phi[k] = {}
                for w in self.uniqueSeg:
                    phi[k][w] = (self.nw[w][k] + self.beta) / (self.nwsum[k] + self.V * self.beta)
        return phi


if __name__ == '__main__':
    texts, unique_word = data_process(path='./input/HillaryEmails.csv')
    lda = LDA(iterations=1000, burnIn=100, sampleLag=10, allseg=texts, uniqueSeg=unique_word)
    """
    alpha 是 选择为 50/ k, 其中k是你选择的topic数，beta一般选为0.01吧，，这都是经验值，貌似效果比较好，收敛比较快一点
    """
    lda.gibbs(K=4, alpha=12.5, beta=0.01)
    phi = lda.get_phi()
    for topic in phi.keys():
        Topic_word = sorted(phi[topic].items(), key=lambda abs: abs[1], reverse=True)[0:5]
        print('topic'+str(topic))
        for topic_word in Topic_word:
            print(topic_word[0] + '：'+str(topic_word[1]))




