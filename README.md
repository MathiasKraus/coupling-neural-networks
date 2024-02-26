# coupling-neural-networks

This is the code corresponding to the paper "Coupling Neural Networks Between Clusters for Better Personalized Care" by Kraus, Mathias and Hambauer, Nico and Müller, Kristina and Kröckel, Pavlina and Ulapane, Nalika and De Caigny, Arno and De Bock, Koen and Coussement, Kristof. 

## Abstract

Personalized healthcare powered by machine learning (ML) is at the forefront of modern medicine, promising to optimize treatment outcomes, reduce adverse effects, and improve patient satisfaction. However, simple ML models generally lack the complexity to accurately model individual characteristics, while powerful ML models require large amounts of data, which are often unavailable in the healthcare domain. We address this problem with cluster-level personalization. In this method, similar patients are grouped into clusters and a local ML model is trained for each cluster. Since the amount of patient data to train ML models naturally decreases for each cluster, we introduce a novel objective function called "coupling" that allows information to be shared between clusters, so that smaller clusters can also benefit from information from larger clusters, thereby improving patient outcome prediction. Our method provides a compromise between a single global model for all patients and completely independent local cluster models. We show that coupling leads to statistically significant improvements on a simulated and a real-world dataset in the context of diabetes.

## Link to paper
Details about the method can be found in the paper https://scholarspace.manoa.hawaii.edu/items/aa6333c4-0ee3-47ca-b072-8392aeeeb68f.

## Citations

```latex
@article{kraus2024coupling,
  title={Coupling Neural Networks Between Clusters for Better Personalized Care},
  author={Kraus, Mathias and Hambauer, Nico and M{\"u}ller, Kristina and Kr{\"o}ckel, Pavlina and Ulapane, Nalika and De Caigny, Arno and De Bock, Koen and Coussement, Kristof},
  booktitle={2024 57th Hawaii international conference on system sciences (HICSS)},
  pages={3627--3636},
  year={2024}
}
```
