#!/usr/bin/env python3
# coding: utf-8
# File: detect.py.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-11-17


import os
from urllib import request
from lxml import etree
from urllib import parse
import jieba.posseg as pseg
import jieba.analyse as anse
import numpy as np

class MultiSenDetect(object):
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.embedding_size = 300
        self.embedding_path = os.path.join(cur, 'word_vec_300.bin')
        self.embdding_dict = self.load_embedding(self.embedding_path)
        # 将实体相似度设置为sim_limit,将大于这个数值的两个实体认为是同一个实体
        self.sim_limit = 0.8

    '''请求主页面'''
    def get_html(self, url):
        return request.urlopen(url).read().decode('utf-8').replace('&nbsp;', '')

    '''收集词的多个义项'''
    def collect_mutilsens(self, word):
        url = "http://baike.baidu.com/item/%s?force=1" % parse.quote(word)
        print(url)
        html = self.get_html(url)
        selector = etree.HTML(html)
        sens = [''.join(i.split('：')[1:]) for i in selector.xpath('//li[@class="list-dot list-dot-paddingleft"]/div/a/text()')]
        sens_link = ['http://baike.baidu.com' + i for i in selector.xpath('//li[@class="list-dot list-dot-paddingleft"]/div/a/@href')]
        sens_dict = {sens[i]:sens_link[i] for i in range(len(sens))}
        return sens_dict

    '''概念抽取'''
    def extract_concept(self, desc):
        desc_seg = [[i.word, i.flag] for i in pseg.cut(desc)]
        concepts_candi = [i[0] for i in desc_seg if i[1][0] in ['n','b','v','d']]
        return concepts_candi[-1]

    '''多义词主函数'''
    def collect_concepts(self, wd):
        sens_dict = self.collect_mutilsens(wd)
        if not sens_dict:
            return {}
        concept_dict = {}
        concepts_dict = {}
        for sen, link in sens_dict.items():
            concept = self.extract_concept(sen)
            if concept not in concept_dict:
                concept_dict[concept] = [link]
            else:
                concept_dict[concept].append(link)
        cluster_concept_dict = self.concept_cluster(concept_dict)

        for concept, links in cluster_concept_dict.items():
            link = links[0]
            desc, keywords = self.extract_desc(link)
            context = ''.join(desc + [' '] + keywords)
            concepts_dict[concept] = context
        return concepts_dict

    '''词义项的聚类'''
    def concept_cluster(self, sens_dict):
        sens_list = []
        cluster_sens_dict = {}
        for sen1 in sens_dict:
            sen1_list = [sen1]
            for sen2 in sens_dict:
                if sen1 == sen2:
                    continue
                sim_score = self.similarity_cosine(self.get_wordvector(sen1), self.get_wordvector(sen2))
                if sim_score >= self.sim_limit:
                    sen1_list.append(sen2)
            sens_list.append(sen1_list)
        sens_clusters = self.entity_clusters(sens_list)
        for sens in sens_clusters:
            symbol_sen = list(sens)[0]
            cluster_sens_dict[symbol_sen] = sens_dict[symbol_sen]

        return cluster_sens_dict

    '''对具有联通边的实体进行聚类'''
    def entity_clusters(self, s):
        clusters = []
        for i in range(len(s)):
            cluster = s[i]
            for j in range(len(s)):
                if set(s[i]).intersection(set(s[j])) and set(s[i]).intersection(set(cluster)) and set(
                        s[j]).intersection(set(cluster)):
                    cluster += s[i]
                    cluster += s[j]
            if set(cluster) not in clusters:
                clusters.append(set(cluster))

        return clusters

    '''获取概念描述信息,作为该个义项的意义描述'''
    def extract_desc(self, link):
        html = self.get_html(link)
        selector = etree.HTML(html)
        keywords = selector.xpath('//meta[@name="keywords"]/@content')
        desc = selector.xpath('//meta[@name="description"]/@content')
        return desc, keywords

    '''对概念的描述信息进行关键词提取,作为整个概念的一个结构化表示'''
    def extract_keywords(self, sent):
        # keywords = [i for i in anse.extract_tags(sent, topK=20, withWeight=False, allowPOS=('n', 'v', 'ns', 'nh', 'nr', 'm', 'q', 'b', 'i', 'j')) if i !=wd]
        keywords = [i for i in anse.extract_tags(sent, topK=20, withWeight=False)]
        return keywords

    '''加载词向量'''
    def load_embedding(self, embedding_path):
        embedding_dict = {}
        count = 0
        for line in open(embedding_path):
            line = line.strip().split(' ')
            if len(line) < 300:
                continue
            wd = line[0]
            vector = np.array([float(i) for i in line[1:]])
            embedding_dict[wd] = vector
            count += 1
            if count%10000 == 0:
                print(count, 'loaded')
        print('loaded %s word embedding, finished'%count)
        return embedding_dict

    '''基于wordvector，通过lookup table的方式找到句子的wordvector的表示'''
    def rep_sentencevector(self, sentence):
        word_list = self.extract_keywords(sentence)
        embedding = np.zeros(self.embedding_size)
        sent_len = 0
        for index, wd in enumerate(word_list):
            if wd in self.embdding_dict:
                embedding += self.embdding_dict.get(wd)
                sent_len += 1
            else:
                continue
        return embedding/sent_len

    '''获取单个词的词向量'''
    def get_wordvector(self, word):
        return np.array(self.embdding_dict.get(word, [0]*self.embedding_size))

    '''计算问句与库中问句的相似度,对候选结果加以二次筛选'''
    def similarity_cosine(self, vector1, vector2):
        cos1 = np.sum(vector1*vector2)
        cos21 = np.sqrt(sum(vector1**2))
        cos22 = np.sqrt(sum(vector2**2))
        similarity = cos1/float(cos21*cos22)
        if str(similarity) == 'nan':
            return 0.0
        else:
            return similarity

    '''基于词语相似度计算句子相似度'''
    def distance_words(self, sent1, sent2):
        wds1 = self.extract_keywords(sent1)
        wds2 = self.extract_keywords(sent2)
        score_wds1 = []
        score_wds2 = []
        for word1 in wds1:
            score = max([self.similarity_cosine(self.get_wordvector(word1), self.get_wordvector(word2)) for word2 in wds2])
            score_wds1.append(score)
        for word2 in wds2:
            score = max([self.similarity_cosine(self.get_wordvector(word2), self.get_wordvector(word1)) for word1 in wds1])
            score_wds2.append(score)
        sim_score = max(sum(score_wds1)/len(wds1), sum(score_wds2)/len(wds2))
        return sim_score

    '根据用户输入的句子,进行概念上的一种对齐'
    def detect_main(self, sent, word):
        sent = sent.replace(word, '')
        concept_dict = self.collect_concepts(word)
        sent_vector = self.rep_sentencevector(sent)
        concept_scores_sent = {}
        concept_scores_wds = {}
        for concept, desc in concept_dict.items():
            concept_vector = self.rep_sentencevector(desc)
            similarity_sent = self.similarity_cosine(sent_vector, concept_vector)
            concept_scores_sent[concept] = similarity_sent
            similarity_wds = self.distance_words(desc, sent)
            concept_scores_wds[concept] = similarity_wds
        concept_scores_sent = sorted(concept_scores_sent.items(), key=lambda asd:asd[1],reverse=True)
        concept_scores_wds = sorted(concept_scores_wds.items(), key=lambda asd:asd[1],reverse=True)
        return concept_scores_sent[:3], concept_scores_wds[:3]

def test():
    handler = MultiSenDetect()
    while(1):
        sent = input('enter an sent to search:').strip()
        wd = input('enter an word to identify:').strip()
        sent_embedding_res, wds_embedding_res = handler.detect_main(sent, wd)
        print(sent_embedding_res)
        print(wds_embedding_res)



if __name__ == '__main__':
    test()
