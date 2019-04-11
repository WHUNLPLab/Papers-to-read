# N-ary Relation Extraction using Graph State LSTM

>> Linfeng Song, Yue Zhang, Zhiguo Wang, 2018, EMNLP

[Source code](https://github.com/freesunshine0316/nary-grn) is available.

## Motivation

Most existing work extracts relations within a sentence while ignoring relations among several entity mentions ($n$-ary relation).

Peng et al. (2017) proposed a graph-structured LSTM for $n$-ary relation extraction. They first split the input graph into two directed acyclic graphs (DAGs) by separating left-to-right edges from right-to-left edges. Then, two separate gated recurrent neural networks were adopted for each single-directional DAG, respectively.

But the bidirectional DAG LSTM model suffers from several limitations:

1. First, important information can be lost when converting a graph into two separate DAGs.

2. Second, using LSTMs on both DAGs, information of only ancestors and descendants can be incorporated for each word. Sibling information, which may also be important, is not included.

## Overview

First, it keeps the original graph structure, and therefore no information is lost. Second, sibling information can be easily incorporated by passing information up and then down from a parent. Third, information exchange allows more parallelization, and thus can be very efficient in computation.

## Task Definition

Formally, the input for cross-sentence $n$-ary relation extraction can be represented as a pair $(\mathcal{E}, \mathcal{T})$, where $\mathcal{E} = (\epsilon_1,\dots, \epsilon_N)$ is the set of entity mentions, and $\mathcal{T} = [S_1,\dots,S_M]$ is a text consisting of multiple sentences. Each entity mention $\epsilon_i$ belongs to one sentence in $\mathcal{T}$ . There is a predefined relation set $R = (r_1,\dots,r_L, None)$, where $None$ represents that no relation holds for the entities.

This task can be formulated as a binary classification problem or a multi-class classification problem of detecting which relation holds for the entity mentions.

## Mehodology

Please read the [paper](https://arxiv.org/abs/1808.09101) for details.

## Dataset

A biomedical-domain dataset focusing on drug-gene-mutation ternary relation, which is extracted from PubMed.