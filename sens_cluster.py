#!/usr/bin/env python3
# coding: utf-8
# File: sens_cluster.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-11-17


def clusters(s):
    clusters = []
    for i in range(len(s)):
        cluster = s[i]
        for j in range(len(s)):
            if set(s[i]).intersection(set(s[j])) and set(s[i]).intersection(set(cluster)) and set(s[j]).intersection(set(cluster)):
                cluster += s[i]
                cluster += s[j]
        if set(cluster) not in clusters:
            clusters.append(set(cluster))

    return clusters



if __name__ == '__main__':
    a = [1,2,3,4,5,6]
    b = [6,8]
    c = [9,10]
    d = [8,11]
    e = [10,12]
    f = [234]
    g = [456]
    s = [a,b,c,d,e,f,g]
    res = clusters(s)
    print(res)