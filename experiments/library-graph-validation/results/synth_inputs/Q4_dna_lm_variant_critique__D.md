You are answering a research question using ONLY the provided context from a
researcher's personal paper library. Write a focused synthesis that:
- directly answers the question with cross-paper structure (consensus, disagreement,
  lineage/subsumption, or a taxonomy as the question demands);
- cites specific papers by title for every substantive claim;
- is explicit about genuine disagreements or contested points between papers;
- does NOT pad, does NOT use outside knowledge beyond the context, and flags if the
  context is insufficient to answer part of the question.
Keep it tight: ~400-600 words.

Do not mention which retrieval method produced the context. Just answer.


=== QUESTION ===
Do my DNA language model papers agree they predict variant effects well -- and how does the benchmarking paper's conclusion compare to the method papers' own claims?

=== CONTEXT ===
=== Benchmarking DNA Sequence Models for Causal Regulatory Variant Prediction in Human Genetics (Benegas, Gonzalo; Eraslan, Gökcen; Song, Yun S.) ===
Benchmarking DNA Sequence Models for Causal Regulatory
Variant Prediction in Human Genetics
Gonzalo Benegas1, G ̈okcen Eraslan2,∗, Yun S. Song1,3,4,∗
1Computer Science Division, University of California, Berkeley 2Biology Research | AI Development, gRED Computational Sciences, Genentech 3Department of Statistics, University of California, Berkeley 4Center for Computational Biology, University of California, Berkeley
March 4, 2025
Abstract
Machine learning holds immense promise in biology, particularly for the challenging task of identifying causal variants for Mendelian and complex traits. Two primary approaches have emerged for this task: supervised sequence-to-function models trained on functional genomics experimental data and self-supervised DNA language models that learn evolutionary constraints on sequences. However, the field currently lacks consistently curated datasets with accurate labels, especially for non-coding variants, that are necessary to comprehensively benchmark these models and advance the field. In this work, we present TraitGym, a curated dataset of regulatory genetic variants that are either known to be causal or are strong candidates across 113 Mendelian and 83 complex traits, along with carefully constructed control variants. We frame the causal variant prediction task as a binary classification problem and benchmark various models, including functional-genomics-supervised models, self-supervised models, models that combine machine learning predictions with curated annotation features, and ensembles of these. Our results provide insights into the capabilities and limitations of different approaches for predicting the functional consequences of non-coding genetic variants. We find that alignmentbased models CADD and GPN-MSA compare favorably for Mendelian traits and complex disease traits, while functional-genomics-supervised models Enformer and Borzoi perform better for complex non-disease traits. Evo2 shows substantial performance gains with scale, but still lags somewhat behind alignment-based models, struggling particularly with enhancer variants. The benchmark, including a Google Colab notebook to evaluate a model in a few minutes, is available at https://huggingface.co/datasets/songlab/TraitGym.
1 Introduction
Machine learning is increasingly transforming the fields of genomics, human genetics, and healthcare by offering new avenues for predicting the impact of genetic variants on phenotypes and by potentially improving the accuracy of trait or disease risk predictions from individual human genomes. A major challenge in these domains is determining which among millions of intercorrelated genetic variants are causal for Mendelian and complex traits, including diseases. Tackling this challenge, which has profound implications for human health, requires robust and scalable methods that can
∗To whom correspondence should be addressed: eraslan.gokcen@gene.com, yss@berkeley.edu
1
available under aCC-BY-NC-ND 4.0 International license.
(which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made
bioRxiv preprint doi: https://doi.org/10.1101/2025.02.11.637758; this version posted March 4, 2025. The copyright holder for this preprint


decode the biological syntax of the human genome and how it drives molecular functions across different cells and tissues. Three major classes of approaches have been developed to model DNA sequences and predict the effects of genetic variants. The first approach utilizes supervised machine learning models, commonly referred to as sequence-to-function models, which are trained to predict genome-wide functional genomics experimental data from DNA sequences (Eraslan et al., 2019); we refer to these models as functional-genomics-supervised. These models predict the functional effects of specific variants by assessing how changes in the DNA sequence influence experimental outcomes. The second approach involves self-supervised genomic language models (gLMs), such as masked or autoregressive language models, which are trained only on DNA sequences from one or multiple species without relying on experimental data (Benegas et al., 2025b). Models that utilize sequences from multiple species take advantage of evolutionary conservation to gain functional insights. Variant effects in such models are assessed by comparing the log-likelihood between the alternative and reference alleles of the variant, as well as by quantifying changes in the latent representations. Another class of methods includes integrative approaches, which combine machine learning predictions with curated annotation features to improve the accuracy of variant effect prediction (Schubach et al., 2024). Additionally, traditional conservation scores phastCons (Siepel et al., 2005) and phyloP (Pollard et al., 2010) have been strong predictors of trait-associated variants (Sullivan et al., 2023). Despite its importance, the field currently lacks consistently processed and comprehensively curated datasets of putative causal regulatory genetic variants with reliable labels. Furthermore, there is a pressing need for establishing a common ground for systematically benchmarking state-ofthe-art models based on functional-genomics-supervised, self-supervised and integrative approaches, in order to help advance the field. In this article, we present TraitGym, a curation of two non-coding variant benchmark datasets from human genetics: one comprising causal variants for 113 Mendelian traits, and another consisting of strong causal variant candidates across 83 complex traits, along with carefully constructed control sets matching relevant summary statistics (such as minor allele frequencies, variant types, distances from transcription start sites, and linkage disequilibrium scores) of putative causal variants. We frame the task as binary classification between putatively causal and noncausal variants, allowing to evaluate several state-of-the-art functional-genomics-supervised and self-supervised models, alongside integrative methods and their ensembles. We find that alignmentbased integrative and self-supervised models compare favorably for Mendelian traits and complex disease traits, while functional-genomics-supervised models do better on complex non-disease traits. The classification of variants is substantially harder for complex traits, but consistent improvement is observed by ensembling input and predicted features from different models. Additionally, we introduce a new gLM trained specifically on regulatory regions and demonstrate that it compares favorably with other alignment-free self-supervised language models.
2 Background
One of the essential quests in biology is to understand the genotype-to-phenotype relationship (Figure 1). The genotype is the genetic makeup of an organism, i.e., the set of DNA sequences composing each genome. The phenotype is the collection of observable traits of an individual, such as height or cholesterol levels. Phenotypic variance can be decomposed into components attributed to genetic and environmental factors. The influence of non-coding genetic variants on phenotype is mediated via the expression of genes in different tissues and cell types. Functional-genomicssupervised models attempt to learn the relationship between DNA sequence and gene expression,
2
available under aCC-BY-NC-ND 4.0 International license.
(which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made
bioRxiv preprint doi: https://doi.org/10.1101/2025.02.11.637758; this version posted March 4, 2025. The copyright holder for this preprint


High
Low
Natural selection
Underlying biology
Functional-genomics-supervised models
Self-supervised models
Genotype Gene expression Phenotype
High
Low
Genotype Gene expression
Genotype
Figure 1: Genotype-to-phenotype relationship and general ML approaches for prediction.
Mendelian trait Complex trait
Genes
Figure 2: Mendelian vs. complex traits. A single gene typically controls a Mendelian trait, whereas a complex trait is influenced by multiple mutations across several genes, each contributing a small individual effect.
leveraging genome-wide experimental data (Eraslan et al., 2019). Natural selection closes the loop by impacting which genotypes are favored over time, based on the fitness of the phenotype on a given environment. Therefore, the space of observed DNA sequences contains rich information about the underlying biology; this is precisely the signal leveraged by self-supervised DNA language models (Benegas et al., 2025b). The are two classes of phenotypic traits: Mendelian and complex (Figure 2). Mendelian traits, such as hemophilia, can be strongly affected by a single mutation in a single gene. On the other hand, complex traits, such as the risk to develop Alzheimer’s disease, are affected by several mutations in multiple genes, each typically with a small individual effect. The fact that variants affecting Mendelian traits have larger phenotypic effect sizes than variants affecting complex traits makes the former relatively easier to predict, as they tend to have larger effects on gene expression (the signal picked up by functional-genomics-supervised models) and tend to be subject to stronger purifying selection (the signal picked up by self-supervised models).
3 Related work
Kathail et al. (2024) provide a comprehensive overview of the landscape of non-coding variant effect prediction in human genetics. GeneticsGym (Finucane et al., 2024) evaluates the prediction of causal variants for human complex traits, but limited to protein-coding variants. Dey et al. (2020)
3
available under aCC-BY-NC-ND 4.0 International license.
(which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made
bioRxiv preprint doi: https://doi.org/10.1101/2025.02.11.637758; this version posted March 4, 2025. The copyright holder for this preprint


evaluate the prediction of non-coding causal variants for human complex traits, but limited to a previous generation of functional-genomics-supervised models. A recent work (Fabiha et al., 2024) also evaluates the prediction of causal variants for complex traits, but does not cover self-supervised models nor Mendelian traits. Benegas et al. (2025a) evaluate the prediction of non-coding causal variants for human Mendelian traits, but with a much larger, non-subsampled negative set of 2.6 million variants, which makes it less practical to evaluate some of the latest, computationally expensive models. Tang et al. (2024) and Patel et al. (2024) benchmark the ability of functional-genomics-supervised and self-supervised models to predict non-coding variant effects on gene expression and chromatin accessibility, but they cover neither Mendelian nor complex traits. BEND (Marin et al., 2024) and GV-Rep (Li et al., 2024) evaluate self-supervised models for the prediction of diseaseassociated variants from ClinVar (Landrum et al., 2020). While not documented, it is likely that these variants mostly cover Mendelian rather than complex diseases. Furthermore, expert-reviewed pathogenic variants in ClinVar are highly skewed towards coding and splice region variants, containing only a single promoter variant and no intergenic variants (Supplementary Table S7). Neither of these benchmarks establishes adequate baselines for this task. BEND includes a single earlygeneration functional-genomics-supervised model (Zhou & Troyanskaya, 2015), but does not include any conservation-based model, which are usually strong for this task (Benegas et al., 2025a). GVRep does not include any baseline. Thus, TraitGym is the only benchmark of causal non-coding variant prediction for both Mendelian and complex human traits. Furthermore, it is the only available framework to evaluate both the latest functional-genomics-supervised and self-supervised models, as well as strong non-neural baselines.
4 Benchmark datasets
TraitGym consists of two curated datasets of non-coding genetic variants affecting Mendelian and complex traits (Table 1). We focus on non-coding variants since understanding their impact is a particularly important use case for DNA sequence models, compared to coding variants which are more commonly interpreted using protein sequence models. Further, we focus on single-nucleotide variants, the most common form of genetic variation, which is still challenging to interpret. Our data curation process is outlined in Figure 3 and additional details are provided in Appendix A.
Mendelian traits. Curated causal non-coding variants for 113 Mendelian diseases were collected from Online Mendelian Inheritance in Man, OMIM (Smedley et al., 2016). For additional stringency, we filtered out a small percentage of variants with minor allele frequency (MAF) greater than 0.1% in the Genome Aggregation Database, gnomAD (Chen et al., 2024). We used gnomAD common variants (MAF > 5%) as controls.
Complex traits. Putative causal and control non-coding variants for 83 complex traits were
Table 1: Number of variants and traits in TraitGym.
Dataset Putatively causal variants Total variants Traits
Mendelian traits 338 3,380 113 Complex traits 1,140 11,400 83
4
available under aCC-BY-NC-ND 4.0 International license.
(which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made
bioRxiv preprint doi: https://doi.org/10.1101/2025.02.11.637758; this version posted March 4, 2025. The copyright holder for this preprint


OMIM pathogenic
Causal variants Control variants
Mendelian traits
Complex traits UKBB PIP > 0.9 UKBB PIP < 0.01
gnomAD common
Consequence
 TSS distance
Consequence
 TSS distance
 MAF
 LD Score
Matching
Figure 3: Matching putatively causal and control variants. Nine matched control variants are used for each putatively causal variant, within each chromosome. See the text for the details.
Figure 4: Distribution of consequence classes of putative causal non-coding variants.
obtained by processing statistical fine-mapping results (Kanai et al., 2021) from association studies in the UK BioBank data (Bycroft et al., 2018). Specifically, we used variants with posterior inclusion probability (PIP) in the credible set greater than 0.9 in any trait as positives and variants with PIP < 0.01 in all traits as controls. We additionally filtered the positive set to genome-wide significant variants (p < 5 × 10−8).
Variant type (or consequence) annotation. We annotated the consequence (e.g., intergenic, intronic, 5′ UTR, 3′ UTR, etc.) of each variant using Ensembl (McLaren et al., 2016), and refined this annotation by overlapping with candidate cis-regulatory elements from ENCODE (Epstein et al., 2020). Distal non-exonic variants (potential enhancers) comprise a small proportion (10%) in the Mendelian traits dataset but the vast majority (76%) in the complex traits dataset (Figure 4).
Matching positives and negatives. For each putative causal non-coding variant, we sampled 9 non-coding variants from the control set, matching chromosome, consequence, and distance to transcription start site (TSS). For complex traits, we additionally matched MAF and linkage disequilibrium (LD) score (Bulik-Sullivan et al., 2015) in the UK BioBank. We sampled only 9 controls per positive variant in order to be able to evaluate even the most computationally demanding models. However, we also provide a larger version of the dataset with millions of negative controls per positive variant, for which we evaluate a subset of the models. This expanded version of the dataset for Mendelian traits does not require any subsampling of negatives, but for complex traits we do subsample to match the MAF distribution (Finucane et al., 2024), while still keeping millions of variants.
Task definition. The task is to classify whether a variant is putatively causal for any trait or not. The input data consist of the reference and alternate allele together with the DNA sequence context. As evaluation metric, we calculate the area under the precision recall curve (AUPRC) for each chromosome (for a model trained on the remaining chromosomes), and then compute a weighted average across chromosomes based on sample size, together with a standard error estimated via bootstrapping (described in Appendix B.4). The baseline AUPRC is 0.1, which is the proportion of positives.
5
available under aCC-BY-NC-ND 4.0 International license.
(which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made
bioRxiv preprint doi: https://doi.org/10.1101/2025.02.11.637758; this version posted March 4, 2025. The copyright holder for this preprint


Table 2: Benchmarked models. Evo2 was trained with 1 M context size but utilizes a shorter context for variant effect prediction.
Model Dependencies Params Context size
Extracted features
Source
Functional genomics
Alignment Population data
Functional-genomics-supervised models
Enformer Yes No No 246M 196K 5,138 Avsec et al. (2021) Sei Yes No No 890M 4K 41 Chen et al. (2022) Borzoi Yes No No 186M 524K 7,617 Linder et al. (2025)
Self-supervised models
GPN-MSA No Yes No 86M 128 770 Benegas et al. (2025a) NT No No No 2.5B 6K 2,562 Dalla-Torre et al. (2024) HyenaDNA No No No 14M 160K 258 Nguyen et al. (2023) Caduceus No No No 8M 131K 514 Schiff et al. (2024) SpeciesLM No No No 97M 2K 770 Tomaz da Silva et al. (2024) AIDO.DNA No No No 7B 4K 4,354 Ellington et al. (2024) Evo2 No No No 40B 8,192 8,194 Brixi et al. (2025) GPN-Promoter No No No 152M 512 1,026 This work
Integrative models
CADD Yes Yes Yes N/A N/A 114 Schubach et al. (2024)
Conservation scores
phastCons No Yes No N/A N/A N/A Siepel et al. (2005) phyloP No Yes No N/A N/A N/A Poll

=== A DNA language model based on multispecies alignment predicts the effects of genome-wide variants (Benegas, Gonzalo; Song, Yun S.; Albors, Carlos; Aw, Alan J.; Ye, Chengzhong) ===
Nature Biotechnology
nature biotechnology
Brief Communication https://doi.org/10.1038/s41587-024-02511-w
A DNA language model based on multispecies alignment predicts the effects of genome-wide variants
Gonzalo Benegas1,2, Carlos Albors2,5, Alan J. Aw3,5, Chengzhong Ye3,5 & Yun S. Song 2,3,4
Protein language models have demonstrated remarkable performance in predicting the effects of missense variants but DNA language models have not yet shown a competitive edge for complex genomes such as that of humans. This limitation is particularly evident when dealing with the vast complexity of noncoding regions that comprise approximately 98% of the human genome. To tackle this challenge, we introduce GPN-MSA (genomic pretrained network with multiple-sequence alignment), a framework that leverages whole-genome alignments across multiple species while taking only a few hours to train. Across several benchmarks on clinical databases (ClinVar, COSMIC and OMIM), experimental functional assays (deep mutational scanning and DepMap) and population genomic data (gnomAD), our model for the human genome achieves outstanding performance on deleteriousness prediction for both coding and noncoding variants. We provide precomputed scores for all ~9 billion possible single-nucleotide variants in the human genome. We anticipate that our advances in genome-wide variant effect prediction will enable more accurate rare disease diagnosis and improve rare variant burden testing.
With the rising trend of whole-genome sequencing, there is a pressing need to understand the effects of genome-wide variants, which would lay the foundation for precision medicine1. In particular, predicting variant deleteriousness is key to rare disease diagnosis2 and rare variant burden tests3. Indeed, a recent review highlighted the analysis of functional rare variants as the biggest contribution of human genetics to drug discovery4. Language models are gaining traction as predictors of deleteriousness, with their ability to learn from massive sequence databases and score variants in an unsupervised manner. Given the success of accurately scoring missense variants with protein language models5–7, it is natural to consider scoring genome-wide variants with DNA language models. For this task, we recently developed the genomic pretrained network (GPN), a model based on a convolutional neural network
trained on unaligned genomes, and showed that it achieves excellent variant effect prediction (VEP) results in the compact genome of Arabidopsis thaliana8. However, the human genome, which harbors a similar number of genes but interspersed over nearly 23-fold larger regions and containing many more repetitive elements, most of which may not be functional, is substantially harder to model. In fact, previous attempts at unsupervised VEP with human DNA language models (for example, Nucleotide Transformer9) showed inferior performance compared to simpler conservation scores. Increasing the scale of the model, data and computer improves performance but it can still be poor, even for a model trained for 28 days using 128 top-of-the-line graphics processing units (GPUs)9. To address the above challenge, we here introduce GPN-MSA (GPN with multiple-sequence alignment), a DNA language model that
Received: 31 October 2023
Accepted: 20 November 2024
Published online: xx xx xxxx
Check for updates
1Graduate Group in Computational Biology, University of California, Berkeley, CA, US. 2Department of Electrical Engineering and Computer Sciences, University of California, Berkeley, CA, US. 3Department of Statistics, University of California, Berkeley, CA, US. 4Center for Computational Biology, University of California, Berkeley, CA, US. 5These authors contributed equally: Carlos Albors, Alan J. Aw, Chengzhong Ye. e-mail: yss@berkeley.edu


Nature Biotechnology
Brief Communication https://doi.org/10.1038/s41587-024-02511-w
massive reduction in computational footprint will enable the efficient exploration of new ideas to train improved DNA language models for genome-wide VEP. GPN-MSA was trained on a whole-genome MSA of 100 vertebrate species (Supplementary Fig. 1), after processing (Fig. 1a) and filtering (Fig. 1b). It is an extension of GPN8 to learn nucleotide probability distributions conditioned not only on surrounding sequence contexts but also on aligned sequences from related species that provide important information about evolutionary constraints and adaptation (Fig. 1c and Methods). It draws inspiration from the MSA Transformer19, a protein language model trained on MSAs of diverse protein families; it was originally designed for structure prediction but was later shown to achieve excellent missense VEP performance5. In addition to the fact that our model operates on whole-genome DNA alignments, which comprise small, fragmented synteny blocks with highly variable levels of conservation and, hence, are considerably more complex than protein alignments, there are essential differences in the architecture and training procedure of GPN-MSA from the MSA Transformer (Methods). By using the MSA as auxiliary information, GPN-MSA can accurately predict nucleotides from their context, especially in functional regions (Supplementary Table 1). At sites where the reference allele differs from the inferred ancestral allele20, GPN-MSA usually favors the ancestral allele (Supplementary Table 2). However, predicting the human reference is just a pretext task. What we really care about is the likelihood assigned to human genetic variants that have not been seen during training. Conservation statistics computed on an MSA column, from simple frequencies to more complex phylogeny-aware P values14, are intuitive and powerful measures of deleteriousness. GPN-MSA is designed to process conservation information across multiple MSA columns, as has been exploited by earlier models such as phastCons21 based on a hidden Markov model. To illustrate GPN-MSA’s power beyond single-column statistics and its ability to leverage genomic context, we note that, even at perfectly conserved positions, GPN-MSA assigns more deleterious scores to loss-of-function (for example, stop gain or loss and splice donor or acceptor variant) and missense variants compared to synonymous variants (Fig. 2a). Furthermore, variants with extreme GPN-MSA log-likelihood ratios tend to have lower minor allele frequencies (MAFs) than variants with extreme log-likelihood ratios based on MSA column frequencies, suggesting that GPN-MSA is a better estimator of deleteriousness (Fig. 2b). We demonstrate the capability of GPN-MSA to improve the unsupervised deleteriousness prediction on several human variant datasets (Methods). We emphasize that only the reference genome is used to train GPN-MSA and that no human variant dataset is used in training. Nevertheless, GPN-MSA can still capture several functional attributes of variants, such as epigenetic marks and the impact of natural selection (Supplementary Fig. 2). For evaluation, we first consider the classification of ClinVar22 pathogenic versus common missense variants in gnomAD23. GPN-MSA substantially outperforms other human DNA language models such as Nucleotide Transformer9, with the largest number of parameters (2.5 billion), as well as HyenaDNA12, with the largest context size of 1 Mb (Fig. 2c and Extended Data Fig. 1a). We also find that GPN-MSA achieves improved performance compared to genome-wide predictors CADD24 and phyloP14,15, as well as the missense-specific ESM-1b6,16. These results are based on using common variants as controls instead of ClinVar benign-labeled variants, as recommended by the developers of CADD to reduce ascertainment bias13. When using benign-labeled variants in ClinVar as controls, the area under the receiving operating characteristic curve (AUROC) for every method is reduced and GPN-MSA performs marginally behind CADD and ESM-1b (Fig. 2d); regardless of the control set, the three methods perform very similarly on ClinVar missense variants. Next, we consider the classification of somatic missense variants frequently observed across cancer tumors (COSMIC25) versus
is designed for genome-wide VEP and is based on the biologically motivated integration of MSA across diverse species using the flexible Transformer architecture10. We apply this modeling framework to humans using an MSA of diverse vertebrate genomes11 and show that it outperforms not only recent DNA language models such as Nucleotide Transformer9 and HyenaDNA12 but also current widely used models such as CADD13, phyloP14,15, ESM-1b6,16), Enformer17 and SpliceAI18. Our model took only 3.5 h to train on four NVIDIA A100 GPUs, which is a considerable reduction in the required computing resources compared to the aforementioned Nucleotide Transformer9. We anticipate that this
Masking
Training: mask 15% of positions
Variant effect prediction: mask variant position
Training:
loss = –w × log (REF)
(A)
(C)
(G)
(T)
Variant effect prediction:
Output: masked nucleotide probability
Transformer neural network
? TC? T T A
A - AGTGA
C - ATTC 
Model architecture
With probability q: replace REF with random nucleotide
c
Input: MSA window
Stitch alignment blocks
Remove gaps in human
Compute conservation: phastCons 75th percentile
Pick 5% most conserved + 0.1% at random
Training window selection
MSA processing
Extract 128-bp windows, overlapping 64 bp
Exclude 10 closest primates
a
b
score = log (ALT)
(REF)
Fig. 1 | Overview of GPN-MSA. a, MSA processing. Starting with a multiplealignment format file, alignment blocks are stitched together following the order in the human reference. Columns with gaps in the human reference are discarded, followed by the removal of the ten primate species closest to human (chimp to squirrel monkey). b, Training window selection. For each 128-bp window along the genome, conservation is computed as the 75th percentile of phastCons. The top 5% conserved windows are chosen alongside a random 0.1% from the remaining windows. c, Model architecture. The input is a 128-bp MSA window where certain positions in the human reference are masked and the goal is to predict the nucleotides at the masked positions given the context across both columns (positions) and rows (species) of the MSA. During training, 15% of the positions are masked. During VEP, only the variant position is masked. The sequence of MSA columns is processed through a Transformer neural network, resulting in a high-dimensional contextual embedding of each position. Then, a final layer outputs four nucleotide probabilities at each masked position. The model is trained with a weighted cross-entropy loss, designed to downweight repetitive elements and upweight conserved elements (Methods). As data augmentation in nonconserved regions, before computing the loss, the reference is sometimes replaced by a random nucleotide (Methods). The GPNMSA VEP score is defined as the log-likelihood ratio between the alternate (ALT) and reference (REF) allele. Mouse and fish icons are from Servier (https://smart. servier.com/).


Nature Biotechnology
Brief Communication https://doi.org/10.1038/s41587-024-02511-w
gnomAD common missense variants. Because of the extreme class imbalance in this case, we focus on the precision and recall metrics. GPN-MSA achieves the highest performance, with substantial margins of improvement over other models (Fig. 2e and Extended Data Fig. 1b). We also evaluate deep mutational scanning (DMS) experimental data26 for 31 human proteins (Supplementary Table 3). GPN-MSA and CADD perform comparably on classifying variants labeled according to protein-specific binarization (Methods), with the former slightly outperforming the latter on the area under the precision–recall curve (AUPRC) metric (Supplementary Fig. 3). They both compare favorably with phyloP and phastCons. However, the protein language model ESM-1b achieves the best overall performance on this task; it likely benefits from modeling long-range interactions within each protein and training on diverse proteins across a much larger evolutionary timescale. Another challenge for genome-wide variant effect predictors on this task is that they expect additional context, such as introns,
which is typically lacking in DMS experimental assays. Nevertheless, GPN-MSA performs better than ESM-1b on some proteins (for example, TAR DNA-binding protein (TADBP), for which the GPN-MSA AUROC is 0.83 while the ESM-1b AUROC is 0.75). An intriguing avenue for future research would be to explore the conditions under which one model outperforms another and integrate the strengths of both DNA and protein language models. As another note of caution, none of the models performs exceedingly well relative to ClinVar results. Moving on to regulatory variants, we evaluate the classification of a curated set of variants implicated in Mendelian disorders (OMIM27) versus gnomAD common variants. We again consider precision and recall because of the extreme class imbalance and find that GPN-MSA achieves the best performance overall, as well as in each variant category (Fig. 2f and Extended Data Figs. 1c and 2). Nucleotide Transformer exhibits poor performance compared to other models (Extended Data Fig. 2). For several variant categories, CADD’s precision increases from
ab
f
c
gh
de
Simulated variants at perfectly conserved positions
Stop gained (n = 1.3K) Splice donor region (n = 30) Splice donor (n = 492)
Splice donor 5th base (n = 54)
Splice acceptor (n = 585) Stop lost (n = 41) Missense (n = 26.3K) Synonymous (n = 31)
ClinVar pathogenic versus gnomAD common (missense)
OMIM pathogenic versus gnomAD common (regulatory)
gnomAD rare versus common (genome-wide)
DepMap essential versus not (gene level)
ClinVar pathogenic versus benign (missense)
COSMIC frequent versus gnomAD common (missense)
–12.5
n = 21,300 versus 15,400 GPN-MSA
GPN-MSA
GPN-MSA CADD CADD
CADD
ESM-1b
ESM-1b ESM-1b
phyloP-100v phyloP-100v phyloP-100v
phyloP-241m
phyloP-241m phyloP-241m
phastCons-100v
phastCons-100v phastCons-100v
NT HyenaDNA
GPN-MSA GPN-MSA GPN-MSA
CADD
CADD
CADD
GeneBayes
phyloP-100v
phyloP-100v
phyloP-241m phyloP-241m
phastCons-100v phastCons-100v
phyloP-100v phyloP-241m pLI hs
0.969 0.914
0.914
0.907
0.856
0.804
0.775 0.029
0.089
0.141
0.167
0.214
0.355 0.963 0.944 0.923 0.905 0.874
0.127
0.048
0.038
0.028
0.006 2.3 0.307
0.316
0.319
0.410
0.434
0.436
0.518
9.8
17.6
35.3
103.7
0.597 0.501
n = 21,300 versus 27,000
n = 406 versus 2.6 million n = 252.7 million versus 5.9 million n = 508 versus 2,800
n = 183 versus 15,400
–10.0 –7.5
10–3
10–4
10–5
10–6 10–4 10–2 100
Mean MAF
Score quantile
gnomAD variants (n = 488.7 M)
Model GPN-MSA MSA column frequency
GPN-MSA score
–5.0
0.6 0.8 AUROC
0.05 0.10 AUPRC
50 100 Odds ratio
0.6 0.8 AUROC
0.1 0.2 0.3 AUPRC
0.2 0.4 AUPRC
1.0
Fig. 2 | VEP results. a, Variant-type-specific distribution of GPN-MSA scores at positions in held-out chromosome 22 where the corresponding MSA columns have perfect conservation (that is, no variation) in the 89 nonhuman species seen by the model. b, Mean MAF for different score quantile bins ([0, 10−6), (10−6, 10−5], ..., (10−1, 1]) in the full set of gnomAD biallelic sites. The MSA column frequency score is the log-likelihood ratio based on the empirical column frequencies, with a pseudocount of 1. To break ties, we added a very small random number to each score (the pattern across random seeds was stable). c, Classification of ClinVar pathogenic versus gnomAD common missense variants. Exact sample size, n = 21,273 versus 15,402. NT, Nucleotide Transformer. d, Classification of ClinVar pathogenic versus ClinVar benign missense variants. Exact sample size, n = 21,275 versus 26,993. e, Classification of COSMIC frequent (frequency > 0.1%) versus gnomAD common missense variants. Exact sample size,
n = 183 versus 15,399. f, Classification of OMIM pathogenic versus gnomAD common regulatory variants. We matched OMIM promoter variants with gnomAD upstream-of-gene variants, enhancer with intergenic and ‘all’ with the union of the matches of the specific categories. Exact sample size, n = 406 versus 2,573,918. g, Enrichment of rare (singletons) versus common (MAF > 5%) gnomAD variants in the tail of deleterious scores (the threshold was chosen such that each score made 30 false discoveries). Odds ratios and P values were computed using a one-sided Fisher’s exact test. All shown odds ratios have a P value < 0.05. Exact sample size, n = 252,706,195 versus 5,894,721. h, Classification of DepMap essential versus nonessential genes using VEPs and selection constraint metrics (Methods). A gene is defined to be essential if >1,000 cell lines in DepMap assays depend on it, whereas it is defined to be nonessential if no cell line depends on it. Exact sample size, n = 508 versus 2,815.


Nature Biotechnology
Brief Communication https://doi.org/10.1038/s41587-024-02511-w
near zero as recall increases, which indicates that a substantial fraction of its top discoveries are actually false (Extended Data Fig. 1c). One example of a deleterious variant that was assigned an extreme score by GPN-MSA is rs606231231, lying in the well-known ZRS enhancer that controls the expression of SHH at the long range of 1 Mb (Supplementary Fig. 4). This variant is associated with polydactyly28 and has been experimentally verified to alter gene expression in mouse limb29. Another example is rs1367115848, disrupting hepatocyte nuclear factor 4 binding at the F7 promoter and causing severe factor VII deficiency30 (Supplementary Fig. 5). Following this, we further evaluate the enrichment of rare versus common gnomAD variants in the tail of the distribution of deleteriousness scores. Deleterious variants should be under purifying selection and, hence, their frequencies in populations should tend to be lower. Therefore, if a variant effect predictor is accurate, we expect rare variants to be enriched compared to common variants for extreme de

=== DNA language models are powerful predictors of genome-wide variant effects (Benegas, Gonzalo; Song, Yun S.; Batra, Sanjit Singh) ===
RESEARCH ARTICLE BIOPHYSICS AND COMPUTATIONAL BIOLOGY OPEN ACCESS
DNA language models are powerful predictors of genome-wide
variant effects
Gonzalo Benegasa ID , Sanjit Singh Batrab ID , and Yun S. Songb,c,d,1 ID
Edited by Kathryn Roeder, Carnegie Mellon University, Pittsburgh, PA; received July 3, 2023; accepted September 8, 2023
The expanding catalog of genome-wide association studies (GWAS) provides biological insights across a variety of species, but identifying the causal variants behind these associations remains a significant challenge. Experimental validation is both laborintensive and costly, highlighting the need for accurate, scalable computational methods to predict the effects of genetic variants across the entire genome. Inspired by recent progress in natural language processing, unsupervised pretraining on large protein sequence databases has proven successful in extracting complex information related to proteins. These models showcase their ability to learn variant effects in coding regions using an unsupervised approach. Expanding on this idea, we here introduce the Genomic Pre-trained Network (GPN), a model designed to learn genome-wide variant effects through unsupervised pretraining on genomic DNA sequences. Our model also successfully learns gene structure and DNA motifs without any supervision. To demonstrate its utility, we train GPN on unaligned reference genomes of Arabidopsis thaliana and seven related species within the Brassicales order and evaluate its ability to predict the functional impact of genetic variants in A. thaliana by utilizing allele frequencies from the 1001 Genomes Project and a comprehensive database of GWAS. Notably, GPN outperforms predictors based on popular conservation scores such as phyloP and phastCons. Our predictions for A. thaliana can be visualized as sequence logos in the UCSC Genome Browser (https://genome.ucsc.edu/s/gbenegas/gpn-arabidopsis). We provide code (https://github.com/songlab-cal/gpn) to train GPN for any given species using its DNA sequence alone, enabling unsupervised prediction of variant effects across the entire genome.
machine learning | language models | variant effect prediction | genome-wide association study | Arabidopsis thaliana
The emergence of genome-wide association studies (GWAS) has significantly enhanced our ability to examine the genetic basis of complex traits and diseases in both humans and plants. In humans, GWAS have played a crucial role in identifying genetic variants associated with a range of traits, including schizophrenia and obesity (1). Similarly, in plants, GWAS have shed light on the genetic factors influencing traits such as drought tolerance, disease resistance, and yield (2). A central challenge in GWAS is pinpointing causal variants for a trait, as linkage disequilibrium (LD) can lead to spurious associations (3). This process, known as fine-mapping, serves as a foundation for constructing accurate, portable polygenic risk scores, and understanding the underlying biological mechanisms. Although experimental validation of causal variants is the gold standard, it is not scalable. Instead, a scalable fine-mapping strategy involves utilizing computational variant effect predictors (4), which vary from conservation scores to deep learning models trained on functional genomics data. Accurate variant effect prediction is also vital for diagnosing rare diseases and interpreting rare variants that lie beyond the scope of traditional GWAS (5). Recently, state-of-the-art performance in predicting the effects of missense (coding) variants has been achieved by training unsupervised models on extensive protein sequence databases (6) or their corresponding multiple sequence alignments (7). These large language models can predict missense variant effects in an unsupervised manner, without the need for additional training on labeled data. This progress has been driven by advancements in natural language processing, where significant strides have been made by pretraining language models on vast text corpora. Pretrained models such as BERT can be fine-tuned for downstream tasks such as sentiment analysis (8). More recently, language models like GPT-4 have demonstrated impressive leaps in test performance across various disciplines, from law to computer science (9). A widely used approach to interpreting noncoding variant effects involves training a supervised model to predict functional genomics data—such as chromatin accessibility, transcription factor binding, or gene expression—and then evaluating variants based on
Significance
Genetic variants across the genome contribute to complex human diseases and agricultural traits, but interpreting them can be challenging. We propose a genome-wide variant effect prediction approach based on unsupervised DNA language models, achieving state-of-the-art performance in Arabidopsis thaliana, a model organism for plant biology and a source of insight into human diseases. Our model, trained solely on DNA sequences, can be applied to any species with a reference genome, even in the absence of expensive functional genomics data. As the artificial intelligence field progresses, our approach can incorporate future advancements, offering a powerful and scalable tool to decipher the vast biological sequence diversity observed in nature.
Author affiliations: aGraduate Group in Computational Biology, University of California, Berkeley, CA 94720; bComputer Science Division, University of California, Berkeley, CA 94720; cDepartment of Statistics, University of California, Berkeley, CA 94720; and dCenter for Computational Biology, University of California, Berkeley, CA 94720
Author contributions: G.B., S.S.B., and Y.S.S. designed research; G.B., S.S.B., and Y.S.S. performed research; G.B. contributed new reagents/analytic tools; G.B. analyzed data; Y.S.S. supervised research; and G.B., S.S.B., and Y.S.S. wrote the paper.
The authors declare no competing interest.
This article is a PNAS Direct Submission.
Copyright © 2023 the Author(s). Published by PNAS. This open access article is distributed under Creative Commons Attribution-NonCommercial-NoDerivatives License 4.0 (CC BY-NC-ND).
1To whom correspondence may be addressed. Email: yss@berkeley.edu.
This article contains supporting information online at https://www.pnas.org/lookup/suppl/doi:10.1073/pnas. 2311219120/-/DCSupplemental.
Published October 26, 2023.
PNAS 2023 Vol. 120 No. 44 e2311219120 https://doi.org/10.1073/pnas.2311219120 1 of 9
Downloaded from https://www.pnas.org by COLD SPRING HARBOR LABORATORY on June 27, 2025 from IP address 143.48.49.107.


how they disrupt these predictions. This approach was first introduced by DeepSEA (10), which utilized 919 functional genomics tracks, and has since been refined by Enformer (11) with 6,956 tracks and Sei (12) with 21,907 tracks. However, this approach’s success depends on the availability of high-quality functional genomics data from a diverse array of cell types, which can be prohibitively expensive to generate for most species. Certain models focus on specific classes of noncoding variants. For instance, classifiers trained solely on sequence data can predict the impact of intron variants on splicing patterns (13, 14). To evaluate the effects of regulatory variants, Lee et al. (15) developed a support vector machine that distinguishes putative regulatory sequences from random genomic sequences. More recently, a deep learning model capable of predicting Hi-C signal from sequence data demonstrated its potential to predict the impact of regulatory variants on DNA folding within the nucleus (16). Additionally, a deep learning model (17) was successfully trained to predict DNA methylation levels of CpG sites from sequence data, enabling the prediction of noncoding variant effects on DNA methylation. However, variant type-specific models may not be well suited for detecting trait-associated rare variants, fine-mapping, or calculating polygenic scores, as these tasks are facilitated by the comparison of genome-wide variants all together. For instance, a model that is exclusively designed for either missense or regulatory variants would not be able to prioritize between a de novo missense variant and a de novo promoter variant observed in an individual with a rare disease. An important class of genome-wide scores are conservation scores such as phyloP (18) and phastCons (19), which are computed from genome-wide alignment of multiple species. Since these do not require functional genomics data, they have been widely applied to many systems, including nonmodel organisms (20). In humans, CADD is another important genome-wide variant effect predictor that combines conservation and functional genomics annotations and is trained to distinguish between an inferred set of putative benign and putative pathogenic variants (21, 22). In this paper, we introduce the GPN, a multispecies DNA language model trained using self-supervision. While existing DNA language models (23–29) have not yet demonstrated the ability to make accurate variant effect predictions based on selfsupervision alone, GPN presents a unified approach capable of accurate unsupervised prediction of genome-wide variant effects. We demonstrate its utility by achieving state-of-the-art performance in Arabidopsis thaliana, a model organism for plant biology closely related to many agriculturally important species, as well as a source of insight into human diseases (30). Moreover, GPN outperforms genome-wide conservation scores such as phyloP and phastCons, which rely on whole-genome alignments of 18 closely related species (20). GPN’s internal representation of DNA sequences can distinguish genomic regions like introns, untranslated regions, and coding sequences. Additionally, the confidence of GPN’s predictions can help reveal regulatory grammar, such as transcription factor binding motifs. Our results lay the foundation for developing state-of-the-art genome-wide variant effect predictors for any species using genomic sequence alone, which can be readily integrated into GWAS fine-mapping and polygenic risk scores.
Results
Training a Multispecies DNA Language Model. We used unaligned reference genomes from A. thaliana and seven related species within the Brassicales order to pretrain a language model
Input: DNA
(L=512) ... C T G C G T C T A ...
Training: mask 15% of positions
... C T G C ? T C T A ...
Variant effect prediction: mask variant position
P(A)
P(C)
P(G)
P(T)
Training: Variant effect prediction:
Output: masked nucleotide probabilities
25x
Feed Forward
Dilated convolution
Add & Norm
Add & Norm
Contextual embedding (D=512)
Classification layer
Masked input
Fig. 1. Overview of GPN. The input is a 512-bp DNA sequence where certain positions have been masked, and the goal is to predict the nucleotides at the masked positions. During training, 15% of the positions are masked. During variant effect prediction, only the variant position is masked. The sequence is processed through a convolutional neural network resulting in a high-dimensional contextual embedding of each position. Then, a final layer outputs four nucleotide probabilities at each masked position. The model is trained on the reference sequence with the cross-entropy loss. The GPN variant effect prediction score is defined as the log-likelihood ratio between the alternate and reference allele. L: window length in base pairs. D: embedding dimension. REF: reference allele. ALT: alternate allele.
based on a convolutional neural network (SI Appendix, Table S1). This model was designed to predict masked nucleotides conditioned on their local genomic context (Fig. 1 and Materials and Methods). During the training process, we encountered challenges with repetitive elements, which can be functionally significant but are heavily overrepresented in the genomes (31). We found that reducing the weight of prediction loss for repetitive regions led to lower test perplexity in nonrepetitive regions, which are often of greater interest (SI Appendix, Table S2). Compared to full down-weighting, moderate down-weighting results in a similar improvement in perplexity for nonrepetitive regions without sacrificing genome-wide perplexity as much. Consequently, we focus on this model throughout the remainder of the paper unless otherwise specified.
Unsupervised Clustering of Genomic Regions. To understand how well the model has learned the structure of the genome, we averaged GPN’s contextual embeddings (512 dimensions) of nucleotides over 100 base pair (bp) windows from the reference genome and visualized them using UMAP (32) (Fig. 2A). Notably, GPN, trained without any supervision, has learned to distinguish genomic regions such as intergenic, introns, coding sequences (CDS), untranslated regions (UTR),
2 of 9 https://doi.org/10.1073/pnas.2311219120 pnas.org
Downloaded from https://www.pnas.org by COLD SPRING HARBOR LABORATORY on June 27, 2025 from IP address 143.48.49.107.


A
B
Fig. 2. Unsupervised clustering of genomic windows. (A) UMAP visualization of GPN embeddings averaged over nonoverlapping 100-bp windows along the genome, annotated with gene region. (B) Confusion matrix for classification of gene regions using a logistic regression model trained on averaged embeddings. Each chromosome was predicted from a model trained on the remaining chromosomes.
and noncoding RNA (ncRNA). To quantify GPN’s ability to distinguish genomic regions, we trained a logistic regression classifier using the averaged embeddings as features, achieving the highest accuracy on CDS (96%) and the lowest on ncRNA (51%), the least frequent class. As summarized in Fig. 2B, the highest confusion was observed between intergenic regions and ncRNAs; this may be partly explained by errors in ncRNA annotation, which is especially challenging given their low expression levels and poor conservation (33). This level of classification accuracy cannot be achieved merely through k-mer frequencies (k = 3: 8% to 70%; k = 6: 15% to 67%; see SI Appendix, Fig. S1). We also note that, to some extent, GPN embeddings can distinguish different repeat families (SI Appendix, Fig. S2).
DNA Motifs Revealed by High-Confidence Model Predictions. To further understand GPN, we individually masked each position in the genome and obtained the model output distribution over nucleotides, given its context. To facilitate utilizing these predicted distributions, we created sequence logos that can be visualized in the UCSC Genome Browser (34, 35) (https:// genome.ucsc.edu/s/gbenegas/gpn-arabidopsis), where the height of each letter is proportional to its probability, and the overall height is given by the information content, measured in bits (36) (see Fig. 3A for an example). The model’s prediction confidence correlates with the expected functionality of the sites. For example, exonic positions are predicted with higher confidence than the surrounding introns, except for the canonical splice acceptor and donor dinucleotide motifs. Similarly, within codons, the third nucleotide position (CDS3), which usually does not affect amino acid identity, is generally predicted with lower confidence than the first two positions (CDS1, CDS2). Start and stop codon motifs are also generally well predicted (examples in SI Appendix, Fig. S3). Across a 1-Mb region in the test chromosome (containing 264 genes and 471 transcripts), model perplexities in splice donors (median = 1.02), splice acceptors (median = 1.03), start codons (median = 1.08), CDS2 (median = 2.24), CDS1 (median = 2.44), CDS3 (median = 2.79), and stop codons (median = 2.8) are significantly smaller than those in intergenic and intronic regions (median = 3.24, all MannWhitney P-values < 10−17, SI Appendix, Fig. S4). Perplexity in CDS2 is significantly smaller than that in CDS1, which in turn is significantly smaller than that in CDS3 (all Mann–Whitney P-values < 10−300), consistent with their different expected levels of constraint (18). We hypothesized that scanning promoters for small regions of high-confidence GPN predictions could help identify transcription factor binding sites. To achieve this, we adapted TFMoDISco (37), a tool for de novo identification of transcription factor binding sites using supervised models. This tool clusters high-scoring regions into motifs and compares them to databases of known motifs. Applying the adapted TF-MoDISco to GPN scores in promoter regions, we identified approximately a hundred and sixty motifs (SI Appendix, Fig. S5), with four examples shown in Fig. 3B, the first two having a significant match in PlantTFDB (20) [with q-value < 0.05 in Tomtom (38)]. Some of the identified motifs are well-documented in the literature but do not have a significant match in this database, such as the third motif (39) in Fig. 3B. Some motifs could represent promoter elements not identified previously, like the fourth motif, which is palindromic with symmetrical entropies, suggesting that it could potentially form RNA or DNA alternative secondary structure (40).
Unsupervised Variant Effect Prediction. GPN can be employed to calculate a pathogenicity or functionality score for any singlenucleotide polymorphism (SNP) in the genome using the loglikelihood ratio between the alternate and reference allele (GPN score, Fig. 1). Visually, this involves comparing the heights of the letters in the logo plot (Fig. 3A). In silico mutagenesis. We first computed GPN scores for in silico mutagenesis of SNPs within a 1-Mb region and aggregated the results across variant types (Fig. 4). The ranking of variant types based on the lowest percentile of GPN scores is generally consistent with established notions of deleteriousness (41)*. For example, the four lowest scored variant types are splice donor,
* https://useast.ensembl.org/info/genome/variation/prediction/predicted_data.html.
PNAS 2023 Vol. 120 No. 44 e2311219120 https://doi.org/10.1073/

=== Predicting expression-altering promoter mutations with deep learning (Jaganathan, Kishore; Ersaro, Nicole; Novakovsky, Gherman; Wang, Yuchuan; James, ) ===
Cite as: K. Jaganathan et al., Science 10.1126/science.ads7373 (2025).
RESEARCH ARTICLES
First release: 29 May 2025 science.org (Page numbers not final at time of first release) 1
The precise control of gene expression is broadly important across human health and development, but the mechanisms by which genomic sequence encodes these intricate programs remain incompletely understood. Central to gene regulation is the promoter, the site of transcriptional initiation, which integrates signals across multiple non-coding sequence elements to determine the proper cellular and temporal context for turning genes on and off. Experimental studies of promoters have shown that they can dramatically increase or decrease gene expression (1), suggesting that non-coding variants which fall within the promoters of clinically relevant genes may play key roles in rare genetic disorders and cancer (2, 3). However, clinical interest in promoter variants has been limited, due to the difficulty of distinguishing between functional non-coding variants that affect gene expression and those that are neutral, with only a handful of pathogenic non-coding variants in promoters having been identified to date (4–12). These challenges in finding pathogenic variants that lie outside protein-coding sequence continue to be a
major hindrance to realizing the potential of personalized genome sequencing. Deep learning models, known for their efficacy at recognizing patterns from large volumes of unstructured data, offer a promising path forward for distilling data collected from genome-wide sequencing and functional genomics experiments into models that can accurately predict the clinical impact of human genetic variation (13, 14). Recent examples that have seen adoption in clinical contexts include SpliceAI (15), which has been cited in expert guidelines for splice variant effect prediction in rare genetic disorders and oncology (16), and missense variant prediction based on protein language models and 3D crystal structures (PrimateAI-3D) (17). While deep learning models such as Evo2 (18), DeepSEA (19), Basenji (20), ExPecto (21), Enformer (22), PuffinD (23), ChromBPNet (24), and Borzoi (25) have been developed to infer the regulatory code directly from genomic sequence, predicting the effects of non-coding genetic variants remains an unmet challenge (26, 27).
Predicting expression-altering promoter mutations with
deep learning
Kishore Jaganathan1†, Nicole Ersaro1†, Gherman Novakovsky1†, Yuchuan Wang1, Terena James1, Jeremy Schwartzentruber1, Petko Fiziev1, Irfahan Kassam1, Fan Cao1, Johann Hawe1, Henry Cavanagh1, Ashley Lim1, Grace Png1, Jeremy McRae1, Abhimanyu Banerjee1, Arvind Kumar1, Jacob Ulirsch1, Yan Zhang1, Francois Aguet1, Pierrick Wainschtein1, Laksshman Sundaram1, Adriana Salcedo1, Sofia Kyriazopoulou Panagiotopoulou1‡, Delasa Aghamirzaie1§, Evin Padhi2, Ziming Weng2, Shan Dong3,4, Damian Smedley5, Mark Caulfield5, Anne O’Donnell-Luria6,7,8, Heidi L. Rehm6,7, Stephan J. Sanders3,4,9, Anshul Kundaje10,11, Stephen B. Montgomery2,10,12, Mark T. Ross1, Kyle Kai-How Farh1*
1Illumina Artificial Intelligence Laboratory, Illumina, Inc., San Diego, CA, USA. 2Department of Pathology, Stanford University, Stanford, CA, USA. 3Department of Psychiatry and Behavioral Sciences, UCSF Weill Institute for Neurosciences, University of California San Francisco, San Francisco, CA, USA. 4Institute of Developmental and Regenerative Medicine, Department of Paediatrics, University of Oxford, Oxford, UK. 5William Harvey Research Institute, Queen Mary University of London, London, UK. 6Program in Medical and Population Genetics, Broad Institute of MIT and Harvard, Cambridge, MA, USA. 7Center for Genomic Medicine and Analytic and Translational Genetics Unit, Massachusetts General Hospital, Boston, MA, USA. 8Division of Genetics and Genomics, Boston Children’s Hospital, Harvard Medical School, Boston, MA, USA. 9New York Genome Center, New York, NY, USA. 10Department of Genetics, Stanford University, Stanford, CA, USA. 11Department of Computer Science, Stanford University, Stanford, CA, USA. 12Department of Biomedical Data Science, Stanford University, Stanford, CA, USA.
†These authors contributed equally to this work. ‡Present address: Arsenal Biosciences, Inc., South San Francisco, CA, USA. §Present address: DELFI Diagnostics, Inc., Palo Alto, CA, USA.
*Corresponding author. Email: kfarh@illumina.com
Only a minority of patients with rare genetic diseases are currently diagnosed by exome sequencing, suggesting that additional unrecognized pathogenic variants may reside in non-coding sequence. Here, we describe PromoterAI, a deep neural network that accurately identifies non-coding promoter variants which dysregulate gene expression. We show that promoter variants with predicted expression-altering consequences produce outlier expression at both RNA and protein levels in thousands of individuals, and that these variants experience strong negative selection in human populations. We observe that clinically relevant genes in rare disease patients are enriched for such variants and validate their functional impact through reporter assays. Our estimates suggest that promoter variation accounts for 6% of the genetic burden associated with rare diseases.
Downloaded from https://www.science.org at Cold Spring Harbor Laboratory on August 06, 2025


First release: 29 May 2025 science.org (Page numbers not final at time of first release) 2
Results
PromoterAI predicts the effects of promoter variants on gene expression
We introduce PromoterAI, a convolutional deep neural network model that uses ~20 kb of sequence context around a promoter variant to predict its expression consequence. We first trained the model to predict histone modifications, DNA accessibility, transcription factor (TF) occupancy, and strandspecific CAGE [Cap Analysis of Gene Expression (28, 29)] values around transcription start sites (TSS) at base-pair resolution. We subsequently fine-tuned the model using a curated set of rare non-coding promoter variants associated with unusually high or low gene expression to further improve its accuracy across extensive validation benchmarks, which we publish as a resource (Fig. 1A, fig. S1A, and table S1, A to G). To create this curated list of variants for fine-tuning, we analyzed the Genotype-Tissue Expression v8 (GTEx) cohort, consisting of paired whole genome sequencing (WGS) and RNA-seq data across 49 tissues in 838 individuals (30). Specifically, we cataloged a set of rare (allele frequency < 0.5%) variants in the promoters of genes (TSS +/− 500 bp) with outlier expression across multiple tissues. These multi-tissue outliers were identified via a t-test between expression zscores from all measured tissues in the carriers versus the same tissues in non-carriers. To maximize confidence in the resulting outliers, when generating the expression z-scores, we subtracted out the contribution of other factors that also affect gene expression: principal components from the expression and genotype matrices, cis-effects of common expression quantitative trait loci (eQTLs) on gene expression, and trans-effects of correlated patterns of gene expression (Fig. 1B). To perform the trans-expression correction, we predicted each gene’s expression using the expression of all genes residing on other chromosomes and then subtracted the predicted expression values from the observed values, since these generally reflect global patterns unrelated to the effects of promoter sequence. To assess the gain in specificity after each correction step, we compared the number of observed expression outliers to background expectation, which we estimated by shuffling the cohort with variants being randomly re-assigned between individuals. After initially correcting for genotype and expression principal components, we found 2,540 multi-tissue outliers at a p-value threshold that produced 1,000 outliers in the shuffled controls; this increased to 3,116 after correcting for cis-eQTLs, and finally to 4,030 following the trans-expression correction (Fig. 1B and table S2). Individuals whose observed gene expression deviated from their predicted gene expression were enriched up to 6-fold for carrying a rare promoter variant in that gene for under-expression, and 2.5-fold for overexpression (Fig. 1C). To identify the extent of the region surrounding the TSS where we can robustly identify multi-tissue expression
outliers, we calculated enrichments of variants with significant effects on gene expression (t-test, p < 1e-4) within 100 bp sliding windows at distances up to 5 kb from the TSS (Fig. 1D). This analysis revealed that expression outliers were predominantly located within 500 bp upstream and downstream of the TSS. While these enrichments improved when the analysis was restricted to conserved promoters (fig. S1B), selecting the most frequently used TSS across tissues was critical for maximizing enrichments (fig. S1C and table S3). We benchmarked our approach against other existing methods to quantify aberrant expression, including an unsupervised autoencoder model from OUTRIDER (31), and two summary statistics available from GTEx (32): median z-scores across tissues and median p-values from an allelic imbalance test (ANEVA-DOT) (33). Compared to the other methods, when choosing thresholds that result in the same false discovery rate in shuffled controls, our approach detected greater enrichment of multi-tissue outliers (Fig. 1E). Using allele-specific expression to assess variants with phasing data available, we found that alleles carrying overexpression outliers had significantly higher allelic expression compared to alleles carrying under-expression outliers (p = 7.4e-29; fig. S1D). As a further orthogonal validation, we observed that the proportion of promoter variants with either over- or under-expression effects was highest for positions with high constraint across 470 mammals (phyloP-470 way; Fig. 1F), indicating that both types of outliers disproportionately fall within conserved sequence elements. Next, we used multi-tissue outliers, as well as matched control variants that came from the promoters of the same genes but showed no effects on expression, and trained PromoterAI on the variants from the odd chromosomes (see Methods for details), while variants from the even chromosomes were set aside for testing and evaluating the model. We benchmarked PromoterAI, Evo2, DeepSEA, Basenji, ExPecto, Enformer, PuffinD, ChromBPNet, and Borzoi on three variant classification tasks, namely, under- vs overexpression, under-expression vs control variants that did not affect gene expression (null), and overexpression vs null (Fig. 1G, left). PromoterAI performed the best out of these models across all three metrics by achieving an auROC of 0.89, 0.80, and 0.74, respectively. We also benchmarked the performance of each classifier on multiple massively parallel reporter assay (MPRA) saturation mutagenesis datasets, including the Critical Assessment of Genome Interpretation (CAGI5) dataset, where we focused on promoters of disease-associated genes (3), and another dataset comprising promoters with known eQTLs and hits from genome-wide association studies (GWAS) (34). PromoterAI achieved the best performance (Fig. 1G, right, and fig. S1E) and also demonstrated strong per-gene predictive capability, as evidenced by correlations between model scores and MPRA effect sizes calculated
Downloaded from https://www.science.org at Cold Spring Harbor Laboratory on August 06, 2025


First release: 29 May 2025 science.org (Page numbers not final at time of first release) 3
separately for each gene (fig. S1F).
Under- and overexpression variants primarily act by disrupting motifs
To understand the basis of PromoterAI’s predictions, we systematically performed in-silico mutagenesis in the vicinity of GTEx outliers, using only the variants from the even chromosomes that were set aside for testing and evaluating the model. Consistent with the model learning the underlying biological mechanisms, PromoterAI assigned high scores to variants that disrupted the motifs of TFs with widely known effects on gene expression, with several representative examples shown (Fig. 2A and fig. S2A). Under-expression variants were particularly enriched for disrupting ETS motifs (Fig. 2B), recognized to be involved in a wide range of regulatory processes (35), along with YY1, ATF1, and NRF1 motifs, which are known to have broad regulatory functions at promoters (36–39). In contrast, overexpression outliers were enriched for disrupting E2F motifs (Fig. 2B), which have roles in cell cycle control and tumor suppression (40), alongside other transcriptional families such as NFKB, INSM1, ERR-alpha, and TFAP2 (41–44). We inserted these motifs along the lengths of promoters to assess their effects using PromoterAI and found that the predictions were consistent with the evidence from expression outliers (fig. S2B), while also localizing the optimal region for maximal effect to 100 bp immediately upstream and downstream of the TSS (45). These in-silico experiments reinforce that PromoterAI successfully learns sequence determinants of promoter regulation. We further examined the relative contributions of variants that affect gene expression by strengthening or weakening motifs, using a curated list of motifs (table S4) exclusively associated with activating or repressive Gene Ontology (GO) terms (46). We found that under-expression outliers were enriched for weakening of motifs associated with only activating GO terms (Fig. 2C, left). In contrast, overexpression outliers were enriched for both weakening of motifs associated with only repressive GO terms and, to a lesser extent, strengthening of motifs associated with only activating GO terms (Fig. 2C, right). Similar trends were observed for motifs associated with both activating and repressive GO terms, subcategorized based on their relative proportions, although the effects were less pronounced due to the complexity of these motifs (fig. S2C). Our results show that it is generally easier for newly occurring variants to cause outlier gene expression by disrupting existing regulatory components rather than creating new ones, as well as highlighting the important role of transcriptional repressors in promoter regulation. Next, we investigated the contribution of fine-tuning to PromoterAI’s performance, aiming to discern the specific variant types which saw the largest gains. We first stratified expression outliers by the motifs they overlapped, and
calculated the correlation between the actual and predicted effects of variants before and after fine-tuning. We saw substantial improvement in performance across nearly all motifs (Fig. 2D) as well as across TSS distance (fig. S2D). We also repeated the earlier analysis of inserting motifs along the lengths of promoters, using the models before and after finetuning. We observed that the predicted direction of effect aligned more consistently with the evidence from expression outliers after fine-tuning (fig. S2E), suggesting that fine-tuning may overcome limitations in predicting the directionality of motifs which has been highlighted in recent papers (26, 27). Moreover, we found that the nucleotide positions which experienced the largest differences in PromoterAI scores before and after fine-tuning tended to be highly conserved (Fig. 2E). Given that conservation information was not used during training or fine-tuning, these results provide independent evidence suggesting that fine-tuning with expression outliers enables the model to more effectively learn the underlying causal biological mechanisms. We show representative examples where fine-tuning led to substantial changes in PromoterAI predictions, noting that the model before fine-tuning frequently recognized the underlying motif but predicted the direction of effect incorrectly (Fig. 2F and fig. S2F). In order to investigate PromoterAI’s internal representations of promoter biology, we extracted feature embeddings of all canonical promoters of protein-coding genes, finding that they clustered into three distinct classes (fig. S3A and table S5): a first class of 9,234 genes abundantly expressed across most tissues and enriched for activating histone marks and ubiquitously expressed TFs, a second class of 3,430 genes characterized by bivalent chromatin, and a third class of 6,632 genes with mainly tissue-specific expression and enriched for repressive histone marks (fig. S3, B to F) (47). The three classes also differed in expression variability, with the first class resembling low-variability promoters, while the second and third classes resembled highly variable promoters (48). Observing that the third promoter class bears many enhancer-like characteristics (49), we used PromoterAI to extract feature embeddings from distal enhancers and compared them with those from promoters. Indeed, the feature embeddings of distal enhancers closely resembled those of the third promoter class (fig. S3G). This enhancer-like promoter class was notably depleted for conserved elements, further emphasizing its distinct regulatory properties (fig. S3H).
Promoter variants that impact gene expression are under strong negative selection
To measure the extent of negative selection acting on promoter variants in human populations, we turned to the Genome Aggregation Database v3 (gnomAD) cohort (50, 51), which includes whole genome sequencing data from 71,702 individuals. We used PromoterAI to score each promoter
Downloaded from https://www.science.org at Cold Spring Harbor Laboratory on August 06, 2025


First release: 29 May 2025 science.org (Page numbers not final at time of first release) 4
variant, excluding those with protein-coding or splicing effects, and compared the number of predicted expression-altering variants at common allele frequencies (> 0.1%) to rare sing

