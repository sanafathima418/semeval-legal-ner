# Legal Named Entities Extraction (L-NER)
India is a highly populated country with the problem of a high number of pending legal cases. Some of these problems can be tackled by automating processes in the pipeline. In this paper, we will be targeting entity recognition for English legal documents to identify entities in the judgment and preamble. 14 Entities like petitioner, respondent, court, statute, provision, precedents, etc are identified which then be used for retrieval-based tasks. Existing NER models cannot be employed to extract these entities as legal texts are different from commonly occurring texts used to train NLP models which is why a custom NER model is necessary to perform NER on Legal Documents using the LegalEval dataset.

The baseline model makes use of spacy-transformers to evaluate legal documents which makes use of compute-intensive pre-trained models like BERT, XLNet, and GPT-2. We have tried to use simpler models to achieve comparable accuracies. The models used take around 15 minutes to train with the usage of 1 GPU core and achieve an overall F1 score of 92.82 and trained using the spacy config system. The model employs sentence-based embeddings using a token-to-vector model followed by a NER parser to identify legal entities in the judgment and preamble. 

We have built the model to take in the data of the LegaEval dataset and process the input into vectors using the tok2vec subnetwork for mapping tokens into vector representations followed by identifying the entities using a transition-based parser model consisting of 2 subnetworks. 

## System Design

<img width="700" alt="Model Architecture" src="https://github.com/maazshaik/semeval-legal-ner/blob/main/images/sysdesign.png">

The model consists of 2 segments. The first segment is to embed tokens into context-independent word vector representations which are performed using the Tok2Vec.v2 architecture with MultiHashEmbed being used. This model passes data through a feed-forward subnetwork to build a mixed representation of the word. A MaxoutWindowEncoder is also used to encode context using convolutions with maxout activation, layer normalization, and residual connections.

The next segment of the model consists of the actual ner which takes into vector representations of a word generated in the previous step and makes use of the transition-based parser model to identify the custom entities. Transition-based parsing is a structured prediction technique where a series of state transitions are used to the structure of a sentence. This neural network model consists of 3 subnetworks, the first for mapping each token into a vector representation which is executed once for every batch. The second layer constructs a feature vector for every token, and feature pair. The final layer is a feed-forward network that predicts scores from state representation.

## Experimental Results

<img width="700" alt="Exp Results" src="https://github.com/maazshaik/semeval-legal-ner/blob/main/images/metrics.png">

## References

1. Kalamkar, P., Agarwal, A., Tiwari, A., Gupta, S., Karn, S., Raghavan, V.: Named
entity recognition in indian court judgments. arXiv preprint arXiv:2211.03442
(2022)
2. Spacy 3.4: Tok2vec architecture
3. Spacy 3.4: Transition-based parser
4. Ushio, A., Camacho-Collados, J.: T-ner: An all-round python library for
transformer-based named entity recognition. arXiv preprint arXiv:2209.12616
(2022)
5. Dozier, C., Kondadadi, R., Light, M., Vachher, A., Veeramachaneni, S., Wudali, R.:
Named entity recognition and resolution in legal text. In: Semantic Processing of
Legal Texts. Springer (2010) 27–43
6. Spacy 3.4: Spacy training pipelines and models
7. Kaggle: Ner using spacy
8. Spacy 3.4: Docbin file


