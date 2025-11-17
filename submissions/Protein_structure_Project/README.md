# Predicting Protein Secondary Structure: From Biological Signals to Advanced Learning Approaches

## Project Summary

This project investigates **Protein Secondary Structure Prediction (PSSP)** from the primary amino-acid sequence, mapping the linear sequence of residues (primary structure) to a one-dimensional annotation of local folding states (secondary structure). We evaluate the impact of different biological descriptors and compare three core modeling strategies: **Random Forest**, **1D Convolutional Neural Networks (CNN)**, and **specialized Protein Language Models (BERT)**.


## The Project Team

| Name | Role | Affiliation | GitHub | LinkedIn |
| :--- | :--- | :--- | :--- | :--- |
| **BOULET Faustine** | Agronomic Engineering Student | Institut Agro Rennes | [GitHub](https://github.com/faustineboulet) | [LinkedIn](https://www.linkedin.com/in/faustine-boulet-066451246/) |
| **BEAUFILS Constance** | Agronomic Engineering Student | Institut Agro Rennes | [GitHub](https://github.com/constancebfls) | [LinkedIn](https://www.linkedin.com/in/constance-beaufils-940672259/) |
| **PLACIER Moïse** | Agronomic Engineering Student | Institut Agro Rennes & ENSAT | [GitHub](https://github.com/MoisePlacier) | [LinkedIn](www.linkedin.com/in/moïse-placier-9639bb24b) | 


## Introduction

Proteins play important roles for living organisms due to their diverse functions, for example, acting as a catalyst in cell metabolism, playing an essential role in DNA replication, forming cell structures, forming living tissues, and constructing antibodies for the immune system [1].
Protein has four different levels of structure: primary structure, secondary structure, tertiary structure, and quaternary structure.

### Definitions

The protein's **primary structure** refers to the linear sequence of amino acids (residues) in a protein. 

The protein's **secondary structure** describes the local organization of the protein backbone. Most residues adopt one of two regular conformations: **alpha-helices** and **beta-strands**.

For modeling, we simplify the standard eight secondary structure categories into three classes:
**H** for helix (including 3-10 helix "G" and $\pi$-helix "I" mapped to 'H').
**E** for $\beta$-strand (including $\beta$-bridge "B")
**C** for coil (turns "T", bends "S", loops "L").

The **tertiary structure** of protein refers to the overall three-dimensional conformation of a polypeptide.

The **quaternary structure** of protein is made up of multiple polypeptide chains that come together.

### Why focus on sequence-based prediction?

Proteins function differently from one another due to variations in their structures, mainly due to the folds that make up varying tertiary structures [2,3]. Since the function of a particular protein is influenced by its tertiary structure, understanding the protein’s tertiary structure is necessary to reveal its functionality. 

A central principle of molecular biophysics is that the amino-acid sequence largely determines the final three-dimensional fold, a concept supported both experimentally and theoretically, known as Anfinsen's dogma [4]. The dogma asserts that the three-dimensional structure of a native protein is determined by its primary amino-acid sequence. This makes predicting structure from sequence an attractive computational goal, especially because sequencing a protein is vastly simpler, faster, and cheaper than experimentally determining its structure.

However, it turns out that predicting protein structure from sequence is not so simple. This is because of a second concept called Levinthal’s paradox [5]. The classical Levinthal paradox illustrates well the combinatorial explosion of possible conformations: if each residue can adopt approximately three stable states, a protein of 101 amino acids would have ∼3^100 potential conformations. Therefore, finding the native folded state of a protein by a random search among all possible configurations can take an enormously long time.

### Why predict secondary structure?

Protein secondary structure prediction (PSSP) is one of the subsidiary tasks of protein structure prediction and is regarded as an intermediary step in predicting protein tertiary structure. If protein secondary structure can be determined precisely, it helps to predict various structural properties useful for tertiary structure prediction.

Therefore, we focus on secondary-structure prediction because it simplifies both the modeling and the evaluation, while the resulting one-dimensional output remains highly informative for understanding folding and guiding tertiary-structure prediction.


## Dataset: ProteinNet

This project uses **ProteinNet**, a curated dataset designed to standardize machine-learning benchmarks for protein structure prediction, mirroring the evaluation protocol of the CASP challenge [6]. It provides protein sequences, structures (secondary and tertiary), multiple sequence alignments (MSAs), position-specific scoring matrices (PSSMs), and standardized training / validation / test splits. ProteinNet builds on the biennial CASP assessments, which carry out blind predictions of recently solved but publicly unavailable protein structures, to provide test sets that push the frontiers of computational methodology.

It is organized as a series of data sets, spanning CASP 7 through 12 (covering a ten-year period), to provide a range of data set sizes that enable assessment of new methods in relatively data poor and data rich regimes.
We chose to focus our model development and evaluation specifically on the historical CASP 8 assessment data, which is included within the comprehensive ProteinNet dataset.

The full CASP8 ProteinNet dataset is not included in the repository because its size exceeds GitHub’s storage limits (approximately 5 GB). To allow full reproducibility, the dataset can be accessed and downloaded from the following source : [DATASET](https://github.com/aqlaboratory/proteinnet)

### Characteristics of Protein Datasets

Protein datasets differ fundamentally from standard machine-learning datasets, and these differences deeply impact training, evaluation, and generalization.

**Non-I.I.D. Nature and Evolutionary Coupling**

Proteins are not independent samples. They arise from evolutionary processes and thus share phylogenetic relationships. Many proteins within a dataset have detectable homology, violating the i.i.d. assumption that underpins most ML models. Two sequences may share a common ancestor even if they differ in function, which makes naïve dataset splitting misleading.

**High Structural and Sequential Redundancy**

Homologous proteins can exceed 90% sequence identity, meaning they are nearly identical at the residue level. This level of redundancy has no equivalent in common ML domains such as vision, where similar classes remain pixel-distinct. Without proper control of redundancy, models trivially memorize homologous examples rather than learning the underlying biophysics of folding.

### Impact on Machine-Learning Training

These dataset characteristics create several pitfalls that must be explicitly mitigated.

**Severe Risk of Overfitting**

If training, validation, and test sets share proteins above 30–40% sequence identity, a model can achieve high accuracy simply by memorizing close homologs. This yields deceptively strong performance, especially on short-range structure, without genuine understanding of folding constraints. When homologous leakage occurs, a model does not generalize the folding process and will fail to predict structures for truly novel proteins, such as proteins from under-sampled species or newly sequenced metagenomic datasets. This undermines the biological utility of the predictor.

### Implemented Solutions

ProteinNet employs several mechanisms to avoid these pitfalls.

**Homology-Aware Splitting Through Sequence Clustering for train and validation split**

Dataset splits are not random but based on sequence-identity clustering. Tools such as jackHMMER [7] group proteins by similarity, ensuring that clusters assigned to training do not overlap with those assigned to validation. This preserves evolutionary independence across folds and yields more realistic performance estimates.

**Blind Evaluation via CASP Protocol**

ProteinNet uses the CASP (Critical Assessment of Structure Prediction) [8] evaluation framework, where models are tested on protein structures that were not publicly available at training time. This ensures transparent, unbiased assessment of generalization. CASP remains the gold standard benchmark for structure prediction.

## Biological Inputs and Modelling Strategy

### 1. Local Model: Physico-Chemical Features + Random Forest (Baseline)

Our initial modeling strategy leverages the intrinsic physico-chemical properties of individual amino acids and their immediate sequence context to predict secondary structure. By employing a sliding window approach, we capture the local interactions that drive helix and coil formation. This framework provides a straightforward yet biologically informed baseline

#### Data Preprocessing for the Random Forest

To prepare the input data for our Random Forest model, each protein sequence is first segmented into overlapping windows of size 11, such that for a sequence of length $L$, $L$ windows are generated, each centered on a single residue. Within each window, individual amino acids are represented as vectors of physico-chemical descriptors derived from the [AAindex1 database](https://www.genome.jp/aaindex/). Residues that correspond to padding ('-') at sequence termini are assigned zero-filled vectors to maintain consistent dimensionality. The descriptors from all residues in a window are then concatenated to form a single flattened feature vector, which preserves local contextual information around each residue while producing a tabular format compatible with classical machine learning models. 

We selected descriptors that influence secondary structure formation:

1) **Hydrophobicity / Hydrophilicity**

Determines helix folding and internal beta-strand stability.

- ARGP820101 Hydrophobicity index [9] 

2) **Side-chain Volume / Size**

Steric effects can favor loops versus compact helices.

- BIGC670101 Residue volume [10] 
- FAUJ880106 Max width of side chain [11]

3) **Polarity**

Influences hydrogen bonding and solvent exposure.

- CHAM820101 Polarizability [12]
- GRAR740102 Polarity [13]
- RADA880108 Mean polarity [14]

4) **Charge**

Local electrostatics affect helix and beta-strand formation.

- FAUJ880111 Positive charge [11]
- FAUJ880112 Negative charge [11]

5) **Flexibility / Rigidity**

Glycine is flexible, proline rigid; this affects loop/corner formation.

- BHAR880101 Flexibility index [15]

6) **Hydrogen-bond potential**

Ability to donate/accept H-bonds stabilizes helices and strands.

- CHAM830107 Charge transfer parameter [16]
- FAUJ880109 Number of hydrogen-bond donors [11]

Following this preprocessing pipeline, we obtain a comprehensive tabular dataset suitable for Random Forest training. For a sliding window of size 11, each residue is represented by 11 × 11 = **121 features**, reflecting the concatenation of 11 physico-chemical descriptors across the local sequence context. The resulting training set comprises **2,270,581 windows**, with secondary structure labels distributed as **1,011,153 coils (C)**, **472,863 beta strands (E)**, and **786,565 helices (H)**.

#### Model Training

To train our Random Forest classifier efficiently, we first extracted a representative subset of **200,000 windows** from the training set, stratified by secondary structure labels to preserve class distributions (C: 1,011,153; E: 472,863; H: 786,565). This subset was used for rapid **hyperparameter tuning**, allowing systematic exploration of the number of estimators, tree depth, minimum samples per leaf, and the number of features considered at each split. Each combination of hyperparameters was evaluated on a held-out validation set using multiple metrics, including Q3 accuracy (the fraction of residues correctly classified into the three secondary structure classes H, E, and C), balanced Q3 accuracy and macro F1 score.

After exhaustive grid search, the optimal parameters were identified as **200 trees**, a **maximum depth of 20**, **minimum samples per leaf of 1**, and sqrt features considered at each split. With thoses hyperparameters determined, the final Random Forest was trained on the full training set and persisted to disk. 

#### Results and discussion 

**Overall Metrics**

| Metric             | Value  |
|-------------------|--------|
| Q3 Accuracy       | 0.665  |
| Balanced Accuracy | 0.626  |
| Macro F1          | 0.633  |

**Class-wise Performance**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| H     | 0.63      | 0.68   | 0.65     | 3788    |
| E     | 0.65      | 0.43   | 0.52     | 2551    |
| C     | 0.69      | 0.77   | 0.73     | 5317    |
| **Macro Avg** | 0.66 | 0.63 | 0.63 | 11656 |
| **Weighted Avg** | 0.66 | 0.66 | 0.66 | 11656 |

Although Random Forests trained on **local physico-chemical** features provide a biologically interpretable baseline, they are inherently limited, which directly accounts for the observed performance. These models are fundamentally local, relying on sliding windows that capture only short-range interactions between residues. 

Beta strands are fundamentally **non-local** structures, as their formation often involves interactions between residues that are distant in the primary sequence, unlike alpha helices, which are largely local and can be stabilized by as few as five consecutive residues through hydrogen bonding. Moreover, a given amino acid sequence can adopt a stable alpha helix in isolation but form a beta strand in the context of long-range interactions with other distant residues. This distinction explains why the model predicts helices and coils relatively well but struggles with beta strands. As Baldwin and Rose note, “the accuracy of secondary structure predictions is only 65–70%. This fact is usually interpreted to imply that the remaining variance of 30–35% is caused by non-local interactions” [17], which is consistent with the ~66% Q3 accuracy achieved by our Random Forest.

Another limitation stems from the residue-level granularity of predictions, which can fragment continuous secondary structure elements. Helices and beta strands span multiple residues, but predicting each residue independently often produces biologically inconsistent patterns.
For instance, a target sequence :
['C',**'H','H','H','H','H','H'**,'C','C','C','C','E','E','E','E'] may be predicted as : ['C',**'H','H','H'**,'C','C','E','C','C','E','E','E','E','E','E']
resulting in short, fragmented structures that do not reflect realistic motifs. This is particularly problematic for alpha helices, which require a minimum of five consecutive residues to form a stable helix. If only three consecutive residues are predicted as helix, it is unclear how to interpret them biologically: should they be converted to coil, extended to form a full helix, or treated as part of a beta strand? Such inconsistencies highlight the limitations of purely residue-level, local prediction approaches.

Finally, the Random Forest model itself has limitations: it treats features as independent and is **invariant to permutations**, so it cannot exploit sequential correlations or detect motifs across neighboring residues within the window as a CNN or transformer-based model could. Additionally, the input features—physico-chemical descriptors—represent only local properties and is a **redundant information**. Taken together, these factors naturally limit the model’s performance, making a plateau around 65–66% accuracy expected for purely local, feature-based methods.

### 2. Evolutionary information and convolutional models (PSSM + 1D CNN) 

This subsection introduces evolutionary descriptors, especially Position-Specific Scoring Matrices (PSSMs) derived from multiple sequence alignments, widely used in secondary-structure predictors.

#### The biological input : PSSM 

A Position-Specific Scoring Matrix (PSSM) provides an evolutionary profile for each residue position within a protein sequence. For each position $i$ and amino acid $a$, the PSSM encodes a substitution probability or score $P(a \mid i)$ derived from a Multiple Sequence Alignment (MSA). An MSA is a matrix-like arrangement of homologous sequences where each row represents a sequence and columns align residues considered evolutionarily equivalent.

Suppose we have the following MSA for a protein segment of length 5:

| Sequence | 1 | 2 | 3 | 4 | 5 |
| -------- | - | - | - | - | - |
| S1       | A | L | K | A | V |
| S2       | A | L | R | A | V |
| S3       | A | I | K | A | I |
| S4       | A | L | K | A | V |

First, the multiple sequence alignment is scanned column by column to count how many times each amino acid occurs at position $i$.

| Position | A | L | I | K | R | V |
| -------- | - | - | - | - | - | - |
| **1**    | 4 | 0 | 0 | 0 | 0 | 0 |
| **2**    | 0 | 3 | 1 | 0 | 0 | 0 |
| **3**    | 0 | 0 | 0 | 3 | 1 | 0 |
| **4**    | 4 | 0 | 0 | 0 | 0 | 0 |
| **5**    | 0 | 0 | 1 | 0 | 0 | 3 |

Next, the counts are converted into positional probabilities $P(a \mid i)$ by dividing by the number of sequences in the MSA. For example, at position 2, $P(L \mid 2)=3/4$ and $P(I \mid 2)=1/4$. These probabilities indicate how frequently each amino acid occurs at a given evolutionary position.

Finally, the positional probabilities are compared to the background frequency $P(a)$ of each amino acid in a large non-redundant protein database using the log-odds formula: $$PSSM(a, i) = \log \frac{P(a \mid i)}{P(a)}$$

| AA ↓ / Pos → | 1                                  | 2                        | 3        | 4        | 5        |
| ------------ | ---------------------------------- | ------------------------ | -------- | -------- | -------- |
| A            | **log(4/4 / 0.05)= log(20)= 3.00** | 0                        | 0        | **3.00** | 0        |
| L            | 0                                  | **log(3/4 / 0.05)=2.30** | 0        | 0        | 0        |
| I            | 0                                  | **log(1/4 / 0.05)=1.61** | 0        | 0        | **1.61** |
| K            | 0                                  | 0                        | **2.71** | 0        | 0        |
| R            | 0                                  | 0                        | **1.61** | 0        | 0        |
| V            | 0                                  | 0                        | 0        | 0        | **2.30** |

If $P(a\mid i)$ is higher than the background, the score is positive, indicating evolutionary enrichment of that amino acid at that position. If it is lower, the score is negative. In the example, position 1 is strongly enriched for A because $P(A\mid1)=1.0$ is much larger than the background probability $P(A)=0.05$, giving a high log-odds value of about 3.00. In other words, a high positive PSSM score at position $i$ for amino acid $a$ indicates that $a$ occurs more often than expected by chance at that position among homologous sequences.

The PSSMs used in this project are derived from the ProteinNet dataset, where MSAs were generated using JackHMMER and weighted with Henikoff position-based weights [18] to reduce the influence of closely related sequences.
 
For secondary structure prediction, PSSMs introduce a strong biological signal that is not present in raw amino acid sequences or physico-chemical descriptors. Evolutionarily conserved positions tend to correspond to structurally or functionally important residues, while positions tolerant to substitution often lie in loops or solvent-exposed regions. 

#### Data Preprocessing for the 1D CNN Model

To leverage the sequential nature of proteins, the 1D convolutional neural network (CNN) operates on full-length sequences rather than fixed local windows. Unlike the Random Forest, which relies on sliding windows to capture short-range context, the CNN requires a consistent tensor shape across all proteins while preserving the residue order.

Each amino acid is represented by a 41-dimensional feature vector. This vector is constructed by concatenating the 21-dimensional One-Hot Encoding (OHE) (20 standard residues plus a dedicated padding token) and the 20-dimensional PSSM profile [19]. Protein sequences vary in length, so both inputs and labels are padded to the maximum sequence length in the dataset. As a result, each protein is converted into an input tensor of shape $(L_{\text{max}}, 41)$ and a label tensor of shape $(L_{\text{max}})$, with padded positions assigned a special index to be ignored during loss computation.

During training, input tensors are transposed to $(\text{Batch}, 41, L_{\text{max}})$, the format expected by PyTorch’s Conv1d layers.

#### 1D CNN Architecture

The network consists of three consecutive 1D convolutional layers with multi-scale kernel sizes (3, 7, and 11) [19] corresponding to different lengths of local structural motifs, progressively increasing the number of filters from 128 to 256 and then 512. Dropout layers (rate 0.5) follow the first two convolutions to mitigate overfitting. The final classification layer is a position-wise 1D convolution with kernel size one, producing logits over the three secondary structure classes (H, E, C) for each residue. Padding is applied to maintain the original sequence length throughout all convolutional layers.

This architecture was chosen to balance biological interpretability and computational efficiency: the use of multiple kernel sizes (3, 7, 11) allows the model to simultaneously reflect the scale of short-range interactions and analyze longer contextual segments (up to 11 residues), crucial for defining β-strands. The progressive increase in filter number allows hierarchical feature extraction, and the absence of pooling preserves positional information critical for residue-level classification. By combining sequence order and evolutionary profiles through PSSMs, the network can leverage both local residue context and evolutionary conservation to improve secondary structure prediction accuracy.

#### Results and discussion 

**Overall Metrics**

| Metric             | Value  |
|-------------------|--------|
| Q3 Accuracy       | **82.56%** |
| Balanced Accuracy | **83%**  |
| Macro F1          | **83%%**  |

**Class-wise Performance**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| H     | 0.85       | 0.86   | 0.85    | 3788    |
| E     | 0.83      | 0.77   | 0.80     | 2551    |
| C     | 0.81      | 0.83   | 0.82     | 5317    |
| **Macro Avg** | 0.83 | 0.82 | 0.82 | 11656 |
| **Weighted Avg** |  0.83  |  0.83  |  0.83  | 11656 |

CNNs apply learnable filters that slide along the sequence, sharing weights across positions and exploiting translational invariance (or equivariance). This yields a stronger ability to learn local sequence motifs and conserved patterns than Random Forests. However, CNNs remain inherently local. Although the multi-scale kernels (3, 7, 11) widen the receptive fields significantly, PSSMs themselves carry position-specific but non-contextual information, so true long-range effects (interactions between residues far apart in the sequence) are still not represented explicitly.

Additionally, PSSMs derived from multiple sequence alignments (MSAs) represent aggregated evolutionary information at each position, but they do not implicitly encode co-evolution between residues. That is, correlations between mutations of different residues—which often reflect spatial contacts in the 3D structure—are lost in the positional averaging of PSSMs.

### 3. Protein Secondary Structure Prediction with ProtBERT

This project leverages **ProtBERT**, a deep language model for protein sequences, to predict residue-level secondary structure (H/E/C). ProtBERT is inspired by BERT from natural language processing and is pretrained on over 100 million non-redundant protein sequences from UniRef90 using a dual-task self-supervised approach:

- **Masked Language Modeling (MLM):** random amino acids are masked and reconstructed.  
- **Gene Ontology (GO) annotation prediction:** the model predicts functional annotations from partial sequences.  

ProtBERT embeddings capture both **local sequence patterns** and **global functional context**, providing 1024-dimensional vectors per residue. These embeddings encode physicochemical, evolutionary, and functional information, offering a rich starting point for downstream tasks such as secondary structure prediction.

#### Embedding Generation Pipeline

1. Each protein sequence is tokenized at the residue level and passed through ProtBERT.  
2. Each residue is mapped to a 1024-dimensional embedding.  
3. Embeddings are saved as `.npy` files per protein, and corresponding secondary structure labels are stored as integer vectors.  
4. During training, sequences are padded to form batches, and padding positions are masked during loss computation.

#### Transformer-based Classifier

To adapt ProtBERT embeddings for secondary structure prediction, we use a **lightweight Transformer architecture**. The embeddings are first projected to a lower-dimensional space, then processed through the Transformer to capture relationships between residues, and finally passed through a linear classifier to produce per-residue predictions for H/E/C classes. This approach allows the model to combine ProtBERT’s rich contextual information with sequence-level dependencies.

#### Results and Discussion

The ProtBERT-based Transformer produces the following overall performance:

| Metric            | Value  |
| ----------------- | ------ |
| Accuracy          | 0.8223 |
| Balanced Accuracy | 0.8149 |
| Macro F1          | 0.8156 |

**Class-wise performance:**

| Class            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| H                | 0.84      | 0.84   | 0.84     | 3788    |
| E                | 0.78      | 0.77   | 0.78     | 2551    |
| C                | 0.83      | 0.83   | 0.83     | 5317    |
| **Macro Avg**    | 0.82      | 0.81   | 0.82     | 11656   |
| **Weighted Avg** | 0.82      | 0.82   | 0.82     | 11656   |

Classical PSSM-based predictors report similar metrics (Q3 accuracy ~ 82.5%, balanced accuracy ~ 83%, macro F1 ~83%).
ProtBERT embeddings encode **residue-level and long-range sequence information**, including patterns learned across millions of proteins, which capture implicit evolutionary and structural signals.

## References

1. Ismi, D. P., Pulungan, R., & Afiahayati. Deep learning for protein secondary structure prediction: Pre and post-AlphaFold. Computational and Structural Biotechnology Journal 20, 6271–6286 (2022).
2. Breda, A. et al. Protein structure, modelling and applications. Bioinformatics in tropical disease research: a practical and case-study approach. (eds. Gruber, A. et al.) 137–170 (National Center for Biotechnology Information (US), 2008).
3. Branden, C. I. & Tooze, J. Introduction to Protein Structure. (Garland Science, 2012).
4. Anfinsen, C. B. Principles that Govern the Folding of Protein Chains. Science 181, 223–230 (1973).
5. Zwanzig, R., Szabo, A. & Bagchi, B. Levinthal’s paradox. Proc. Natl. Acad. Sci. U.S.A. 89, 20–22 (1992).
6. AlQuraishi, M. ProteinNet: a standardized data set for machine learning of protein structure. BMC Bioinformatics 20, 311 (2019).
7. Finn, R. D. et al. HMMER web server: 2015 update. Nucleic Acids Res 43, W30–W38 (2015).
8. Kryshtafovych, A., Schwede, T., Topf, M., Fidelis, K. & Moult, J. Critical assessment of methods of protein structure prediction (CASP)—Round XIII. Proteins 87, 1011–1020 (2019).
9. Argos, P., Rao, J. K. M. & Hargrave, P. A. Structural Prediction of Membrane‐Bound Proteins. European Journal of Biochemistry 128, 565–575 (1982).
10. Bigelow, C. C. On the average hydrophobicity of proteins and the relation between it and protein structure. Journal of Theoretical Biology 16, 187–211 (1967).
11. Fauchère, J., Charton, M., Kier, L. B., Verloop, A. & Pliska, V. Amino acid side chain parameters for correlation studies in biology and pharmacology. International Journal of Peptide and Protein Research 32, 269–278 (1988).
12. Charton, M. & Charton, B. I. The structural dependence of amino acid hydrophobicity parameters. Journal of Theoretical Biology 99, 629–644 (1982).
13. Grantham, R. Amino Acid Difference Formula to Help Explain Protein Evolution. Science 185, 862–864 (1974).
14. Radzicka, A., Pedersen, L. & Wolfenden, R. Influences of solvent water on protein folding: free energies of solvation of cis and trans peptides are nearly identical. Biochemistry 27, 4538–4541 (1988).
15. Bhaskaran, R. & Ponnuswamy, P. K. Positional flexibilities of amino acid residues in globular proteins. International Journal of Peptide and Protein Research 32, 241–255 (1988).
16. Charton, M. & Charton, B. I. The dependence of the Chou-Fasman parameters on amino acid side chain structure. Journal of Theoretical Biology 102, 121–134 (1983).
17. Baldwin, R. L. & Rose, G. D. Is protein folding hierarchic? I. Local structure and peptide folding. Trends in Biochemical Sciences 24, 26–33 (1999).
18. Henikoff, S. & Henikoff, J. G. Position-based sequence weights. Journal of Molecular Biology 243, 574–578 (1994).
19. Lu, Y. Protein Secondary Structure Prediction Using Convolutional Bidirectional GRU. JMR 16, 11 (2024).



