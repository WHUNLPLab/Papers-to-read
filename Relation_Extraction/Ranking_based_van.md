# Ranking-Based Automatic Seed Selection and Noise Reduction for Weakly Supervised Relation Extraction

>> Van-Thuy Phi, Joan Santoso, Masashi Shimbo, 2018, ACL

## Introduction

* Bootstrapping for RE:
  
  Initialized by a small set of example instances called seed, to represent a particular semantic relation, the bootstrapping system operates iteratively to aquire new instances of a target relation. Selecting "good" seeds is one of the most important steps to reduce semantic drift, which is a typical phenomenon of the bootstrapping process.

We propose methods that can be applied for both automatic seed selection and noise reduction by formulating these tasks as ranking problems according to different ranking criteria.

## Problem Formulation

For each target relation $r \in R^{*}$, we assume there is a set $D_r$ of triples. The triples in $D_r$ have the form $(e_1, p, e_2)$, where $(e_1, e_2)$ is called an instance, $p$ denotes the pattern.

### Seed Selection for Bootstrapping RE

Given $R^{*}$ and sets of instance-patther triples $D_r=\{(e_1,p,e_2)\}$, the task is to choose good seeds from the instance appearing in $D_r$ for each $r \in R^{*}$.

### Noise Reduction for Distantly Supervised RE

The goal of noise reduction is to filter out these noisy triples, so that they do not deteriorate the quality of the triple classifier trained subsequently.

>> To be precise, in each triple $(e_1,s,e_2)$ generated by DS, $s$ is not a pattern but a sentence that contains entities $e_1$ and $e_2$. However, we can easily convert each instance-sentence triple $(e_1, s, e_2)$ to an instance-pattern triple $(e_1, p, e_2)$ by looking for a pattern $p$ that connects two entities in sentence.

### Formulaton as Ranking Tasks:

In the seed selection task, we use the $k$ highest ranked instances as the seeds for bootstrapping RE. Likewise, in noise reduction for DS, we only use the $k$ highest ranked triples from the DS-generated data to train a classifier. Note that the value of $k$ in noise reduction may be much larger than in seed selection.

## Approaches to Automatic Seed Selection and Noise Reduction

### K-means-based Approach

1. Determine the number $k$ of instances/triples that should be selected
2. Run the K-means algorithm to partition all instances in the input triples into $k$ clusters.
3. The instance closest to the centroid is selected in each cluster.

### HITS-based Approach

1. Determine the number $k$ of instances/triples that should be selected
2. Build the bipartite graph of instances and patterns based on the instance-pattern co-occurrence matrix $A$
3. Retain the top-$k$ instances with the highest hubness scores as the outputs

### HITS-and K-means-based Approach

We first rank the instances and patterns based on their bipartite graph and then run K-means to cluster instances in our annotated dataset. However, instead of choosing the instance nearest to the centroid we retain the one that has the highest HITS hubness score in each cluster.

### LSA-based Approach

$$
\mathbf{A}\approx \mathbf{ISP}^T
$$

where $A$ is the instance-pattern co-occurrence matrix, $I,S,P$ are SVD matrix.

1. Specify the desired number $k$ of triples
2. Use the LSA algorithm to decompose the instance-pattern co-occurrence matrix $A$ into three matrices $I$, $S$, and $P$
3. We can consider LSA as a form of soft clustering, with each column of the SVD instance matrix $I$ corresponding to a cluster. Then, we select the $k$ instances that have the highest absolute values from each column of $I$

### NMF-based Approach

$$
\mathbf{A} \approx \mathbf{WH}
$$

where $A \in \mathbb{R}^{M\times N}$ is non-negative data matrix, $W \in \mathbb{R}^{M \times K}$ and  $H \in \mathbb{R}^{K \times N}$ are non-negative factors.

The non-negativity constraint is the main difference between NMF and LSA. We then select the $k$ instances that have the highest values from each column of $W$.