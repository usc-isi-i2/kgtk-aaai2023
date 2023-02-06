# Welcome

This page contains the notebooks corresponding to the tutorial [KGTK: User-friendly Toolkit for Manipulation of Large Knowledge Graphs](https://usc-isi-i2.github.io/kgtk-tutorial-aaai-2023) given at AAAI'23.

## Installation

```
conda create -n kgtk23 python=3.9
conda activate kgtk23

conda install -c conda-forge jupyterlab

git clone https://github.com/usc-isi-i2/kgtk.git
pip install -e .
```

## Use case notebooks

1. [Internet Memes](https://github.com/usc-isi-i2/kgtk-aaai2023/blob/main/04-InternetMemes.ipynb) - we show how KGTK can help connect the dots between internet meme sources and external knowledge graphs, like Wikidata. We use KGTK to perform scalable analytics of the resulting graph and execute novel entity-centric and hybrid queries.

2. [Financial transactions](https://github.com/usc-isi-i2/kgtk-aaai2023/blob/main/02-FinancialTransactions.ipynb) - we describe how KGTK can be used analyze financial transaction data. We illustrate how to construct KGTK pipelines with graph transformations, analytics and visualization steps for the financial sector. The KGTK pipelines enable us to highlight trading behaviors, to find potential colluders, and to find inconsistencies through differentiating knowledge graph structures.

3. [Publication graphs (PubGraphs)](https://github.com/usc-isi-i2/kgtk-aaai2023/blob/main/01-PubGraph.ipynb) - The recent advent of public large-scale research publications metadata repositories such as OpenAlex (Priem, Piwowar, and Orr 2022) enables us to study innovation at scales that have not been possible before. However, dealing with these large-scale repositories is extremely difficult and requires special toolkits. In this notebook, we describe how KGTK can be used for data filtering, data transformation, knowledge graph extraction, and knowledge graph embedding training of knowledge graphs with scientific publications.

4. Morality in events - we will demonstrate how our knowledge graph tools are applied to make sense of complex events. Focused on a specific domain (or location) we track the changes in moral foundations (Johnson and Goldwasser 2018) and emotions to understand public perception of these events. The use of KGTK in this project makes it easy to scale up, to generalize to other domains and locations, and to browse and visualize the data.
