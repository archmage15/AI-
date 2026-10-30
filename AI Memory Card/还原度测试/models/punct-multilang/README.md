---
language:
- en
- de
- fr
- it
- nl
- multilingual
tags:
- punctuation prediction
- punctuation
datasets: wmt/europarl
license: mit
widget:
- text: "Ondanks dat het nu bijna voorjaar is hebben we nog steds best koude dagen"
  example_title: "Dutch"
- text: "Ho sentito che ti sei laureata il che mi fa molto piacere"
  example_title: "Italian"
- text: "Tous les matins vers quatre heures mon père ouvrait la porte de ma chambre"
  example_title: "French"
- text: "Ist das eine Frage Frau Müller"
  example_title: "German"
- text: "My name is Clara and I live in Berkeley California"
  example_title: "English"  
metrics:
- f1
---

# Work in progress

## Classification report over all languages 
```
             precision    recall  f1-score   support

           0       0.99      0.99      0.99  47903344
           .       0.94      0.95      0.95   2798780
           ,       0.85      0.84      0.85   3451618
           ?       0.88      0.85      0.87     88876
           -       0.61      0.32      0.42    157863
           :       0.72      0.52      0.60    103789

    accuracy                           0.98  54504270
   macro avg       0.83      0.75      0.78  54504270
weighted avg       0.98      0.98      0.98  54504270
```




## How to cite us

```
@article{guhr-EtAl:2021:fullstop,
  title={FullStop: Multilingual Deep Models for Punctuation Prediction},
  author    = {Guhr, Oliver  and  Schumann, Anne-Kathrin  and  Bahrmann, Frank  and  Böhme, Hans Joachim},
  booktitle      = {Proceedings of the Swiss Text Analytics Conference 2021},
  month          = {June},
  year           = {2021},
  address        = {Winterthur, Switzerland},
  publisher      = {CEUR Workshop Proceedings},  
  url       = {http://ceur-ws.org/Vol-2957/sepp_paper4.pdf}
}

```

```
@misc{https://doi.org/10.48550/arxiv.2301.03319,
  doi = {10.48550/ARXIV.2301.03319},
  url = {https://arxiv.org/abs/2301.03319},
  author = {Vandeghinste, Vincent and Guhr, Oliver},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences, I.2.7},
  title = {FullStop:Punctuation and Segmentation Prediction for Dutch with Transformers},
  publisher = {arXiv},
  year = {2023},  
  copyright = {Creative Commons Attribution Share Alike 4.0 International}
}

```
