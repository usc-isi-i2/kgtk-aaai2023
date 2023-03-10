# Welcome

This page contains the notebooks corresponding to the tutorial [KGTK: User-friendly Toolkit for Manipulation of Large Knowledge Graphs](https://usc-isi-i2.github.io/kgtk-tutorial-aaai-2023) given at AAAI'23.

## Installation

To install KGTK, run `pip install kgtk`. **Note: there are known issues with Python >3.9, so we suggest using a virtual (conda) environment with Python v3.9**.

To run the notebooks locally, you can use Jupyter Lab, which is installed with `conda install -c conda-forge jupyterlab`.

If you run into problems, please visit [the KGTK GitHub page](https://github.com/usc-isi-i2/kgtk) for other installation possibilities. If the problems persist, please open an issue on the KGTK GitHub page and we will take a look.

## Use case notebooks

1. [Internet Memes](https://github.com/usc-isi-i2/kgtk-aaai2023/blob/main/01-InternetMemes.ipynb) - we show how KGTK can help connect the dots between internet meme sources and external knowledge graphs, like Wikidata. We use KGTK to perform scalable analytics of the resulting graph and execute novel entity-centric and hybrid queries.

2. [Financial transactions](https://github.com/usc-isi-i2/kgtk-aaai2023/blob/main/03-FinancialTransactions.ipynb) - we describe how KGTK can be used analyze financial transaction data. We illustrate how to construct KGTK pipelines with graph transformations, analytics and visualization steps for the financial sector. The KGTK pipelines enable us to highlight trading behaviors, to find potential colluders, and to find inconsistencies through differentiating knowledge graph structures.

3. [Publication graphs (PubGraphs)](https://github.com/usc-isi-i2/kgtk-aaai2023/blob/main/02-PubGraph.ipynb) - The recent advent of public large-scale research publications metadata repositories such as OpenAlex (Priem, Piwowar, and Orr 2022) enables us to study innovation at scales that have not been possible before. However, dealing with these large-scale repositories is extremely difficult and requires special toolkits. In this notebook, we describe how KGTK can be used for data filtering, data transformation, knowledge graph extraction, and knowledge graph embedding training of knowledge graphs with scientific publications.

4. [Morality in events](https://github.com/usc-isi-i2/kgtk-aaai2023/blob/main/04-MoralityInEvents.ipynb) - we will demonstrate how our knowledge graph tools are applied to make sense of complex events. Focused on a specific domain (or location) we track the changes in moral foundations (Johnson and Goldwasser 2018) and emotions to understand public perception of these events. The use of KGTK in this project makes it easy to scale up, to generalize to other domains and locations, and to browse and visualize the data. This notebook can be run in [Google Colab](https://bit.ly/3XfNnkR)
