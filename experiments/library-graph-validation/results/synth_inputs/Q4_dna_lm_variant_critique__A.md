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
[A1] A DNA language model based on multispecies alignment predicts the effects of genome-wide variants (Benegas, Gonzalo; Song, Yun S.; Albors, Carlos; Aw, Alan J.; Ye, Chengzhong)
Abstract: Protein language models have demonstrated remarkable performance in predicting the effects of missense variants but DNA language models have not yet shown a competitive edge for complex genomes such as that of humans. This limitation is particularly evident when dealing with the vast complexity of noncoding regions that comprise approximately 98% of the human genome. To tackle this challenge, we introduce GPN-MSA (genomic pretrained network with multiple-sequence alignment), a framework that leverages whole-genome alignments across multiple species while taking only a few hours to train. Across several benchmarks on clinical databases (ClinVar, COSMIC and OMIM), experimental functional assays (deep mutational scanning and DepMap) and population genomic data (gnomAD), our model for the human genome achieves outstanding performance on deleteriousness prediction for both coding and noncoding variants. We provide precomputed scores for all ~9 billion possible single-nucleotide variants in the human genome. We anticipate that our advances in genome-wide variant effect prediction will enable more accurate rare disease diagnosis and improve rare variant burden testing.
Excerpt: Nature Biotechnology
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
massive reduction in computational footprint will enable the efficient exploration of new ideas to train improved DNA language models for genome-wide VEP. GPN-MSA was trained on a whole-genome MSA of 100 vertebrate species (Supplementary Fig. 1), after processing (Fig. 1a) and filtering (Fig. 1b). It is an extension of GPN8 to learn nucleotide probability distributions conditioned not only on surrounding sequence contexts but also on aligned sequences from related species that provide important information about evolutionary constraints and adaptation (Fig. 1c and Methods). It draws inspiration from the MSA Transformer19, a protein language model trained on MSAs of diverse protein families; it was originally designed for structure prediction but was later shown to achieve excellent missense VEP performance5. In addition to the fact that our model operates on whole-genome DNA alignments, which comprise small, fragmented synteny blocks with highly variable levels of conservation and, hence, are considerably more complex than protein alignments, there are essential differences in the architecture and training procedure of GPN-MSA from the MSA Transformer (Methods). By using the MSA as auxiliary information, GPN-MSA can accurately predict nucleotides from their context, especially in functional regions (Supplementary Table 1). At sites where the reference allele differs from the inferred ancestral allele20, GPN-MSA usually favors the ancestral allele (Supplementary Table 2). However, predicting the human reference is just a pretext task. What we really care about is the likelihood assigned to human genetic variants that have not been seen during training. Conservation statistics computed on an MSA column, from simple frequencies to more complex phylogeny-aware P values14, are intuitive and powerful measures of deleteriousness. GPN-MSA is designed to process conservation information across multiple MSA columns, as has been exploited by earlier models such as phastCons21 based on a hidden Markov model. To illustrate GPN-MSA’s power beyond single-column

[A2] DNA language models are powerful predictors of genome-wide variant effects (Benegas, Gonzalo; Song, Yun S.; Batra, Sanjit Singh)
Abstract: The expanding catalog of genome-wide association studies (GWAS) provides biological insights across a variety of species, but identifying the causal variants behind these associations remains a significant challenge. Experimental validation is both labor-intensive and costly, highlighting the need for accurate, scalable computational methods to predict the effects of genetic variants across the entire genome. Inspired by recent progress in natural language processing, unsupervised pretraining on large protein sequence databases has proven successful in extracting complex information related to proteins. These models showcase their ability to learn variant effects in coding regions using an unsupervised approach. Expanding on this idea, we here introduce the Genomic Pre-trained Network (GPN), a model designed to learn genome-wide variant effects through unsupervised pretraining on genomic DNA sequences. Our model also successfully learns gene structure and DNA motifs without any supervision. To demonstrate its utility, we train GPN on unaligned reference genomes of Arabidopsis thaliana and seven related species within the Brassicales order and evaluate its ability to predict the functional impact of genetic variants in A. thaliana by utilizing allele frequencies from the 1001 Genomes Project and a comprehensive database of GWAS. Notably, GPN outperforms predictors based on popular conservation scores such as phyloP and phastCons. Our predictions for A. thaliana can be visualized as sequence logos in the UCSC Genome Browser (https://genome.ucsc.edu/s/gbenegas/gpn-arabidopsis). We provide code (https://github.com/songlab-cal/gpn) to train GPN for any given species using its DNA sequence alone, enabling unsupervised prediction of variant effects across the entire genome.
Excerpt: RESEARCH ARTICLE BIOPHYSICS AND COMPUTATIONAL BIOLOGY OPEN ACCESS
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
This articl

[A3] Benchmarking DNA Sequence Models for Causal Regulatory Variant Prediction in Human Genetics (Benegas, Gonzalo; Eraslan, Gökcen; Song, Yun S.)
Abstract: Machine learning holds immense promise in biology, particularly for the challenging task of identifying causal variants for Mendelian and complex traits. Two primary approaches have emerged for this task: supervised sequence-to-function models trained on functional genomics experimental data and self-supervised DNA language models that learn evolutionary constraints on sequences. However, the field currently lacks consistently curated datasets with accurate labels, especially for non-coding variants, that are necessary to comprehensively benchmark these models and advance the field. In this work, we present TraitGym, a curated dataset of regulatory genetic variants that are either known to be causal or are strong candidates across 113 Mendelian and 83 complex traits, along with carefully constructed control variants. We frame the causal variant prediction task as a binary classification problem and benchmark various models, including functional-genomics-supervised models, self-supervised models, models that combine machine learning predictions with curated annotation features, and ensembles of these. Our results provide insights into the capabilities and limitations of different approaches for predicting the functional consequences of non-coding genetic variants. We find that alignment-based models CADD and GPN-MSA compare favorably for Mendelian traits and complex disease traits, while functional-genomics-supervised models Enformer and Borzoi perform better for complex non-disease traits. Evo2 shows substantial performance gains with scale, but still lags somewhat behind alignment-based models, struggling particularly with enhancer variants. The benchmark, including a Google Colab notebook to evaluate a model in a few minutes, is available at https://huggingface.co/datasets/songlab/TraitGym.
Excerpt: Benchmarking DNA Sequence Models for Causal Regulatory
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


decode the biological syntax of the human genome and how it drives molecular functions across different cells and tissues. Three major classes of approaches have been developed to model DNA sequences and predict the effects of genetic variants. The first approach utilizes supervised machine learning models, commonly referred to as sequence-to-function models, which are trained to predict genome-wide functional genomics experimental data from DNA sequences (Eraslan et al., 2019); we refer to these models as functional-genomics-supervised. These models predict the functional effects of specific variants by assessing how changes in the DNA sequence influence experimental outcomes. The second approach involves self-supervised genomic language models (gLMs), such as masked or autoregressive language models, which are trained only on DNA sequences from one or multiple species without relying on experimental data (Benegas et al., 2025b). Models that utilize sequences from multiple species take advantage of evolutionary conservation to gain functional insights. Variant effects in such models are assessed by comparing the log-likelihood between the alternative and reference alleles of the variant, as well as by quantifying changes in the latent representations. Another class of methods includes integrative approaches, which combine machine learning predictions with curated annotation features to improve the accuracy of variant effect prediction (Schubach et al., 2024). Additionally, traditional conservation scores phastCons (Siepel et al., 2005) and phyloP (Pollard et al., 2010) have been strong predictors of trait-associated variants (Sullivan et al., 2023). Despite its importance, the field currently lacks consistently processed and comprehensively curated datasets of putative causal regulatory genetic variants with reliable labels. Furthermore, there is a pressing need for establishing a common ground for systematically benchmarking state-ofthe-art models based on functional-genomics-supervised, self-supervised and integrative approaches, in order to help advance the field. In this article, we present TraitGym, a curation of two non-coding variant benchmark datasets from human genetics: one comprising causal variants for 113 Mendelian traits, and another consisting of strong causal variant candidates across 83 complex traits, along with carefully constructed control sets matching relevant summary statistics (such as minor allele frequencies, variant types, distances from transcription start sites, and linkage disequilibrium scores) of putative causal variants. We frame the task as binary classification between putatively causal and noncausal variants, all

[A4] Simulating 500 million years of evolution with a language model (Lin, Zeming; Akin, Halil; Rao, Roshan; Verkuil, Robert; Sercu, Tom; Candido, Sal)
Abstract: (none)
Excerpt: Simulating 500 million years of evolution with a language model

Thomas Hayes 1 * Roshan Rao 1 * Halil Akin 1 * Nicholas James Sofroniew 1 * Deniz Oktay 1 * Zeming Lin 1 *

Robert Verkuil 1 * Vincent Quy Tran 2 3 Jonathan Deaton 1 Marius Wiggert 1 Rohil Badkundri 1

Irhum Shafkat 1 Jun Gong 1 Alexander Derry 1 Raul Santiago Molina 1 Neil Thomas 1 Yousuf Khan 4

Chetan Mishra 1 Carolyn Kim 1 Liam J. Bartie 2 Patrick D. Hsu 2 3 Tom Sercu 1 Salvatore Candido 1 Alexander Rives 1 †

Abstract
More than three billion years of evolution have produced an image of biology encoded into the space of natural proteins. Here we show that language models trained on tokens generated by evo-
W lution can act as evolutionary simulators to gen-
erate functional proteins that are far away from known proteins. We present ESM3, a frontier multimodal generative language model that reasons over the sequence, structure, and function
E of proteins. ESM3 can follow complex prompts
combining its modalities and is highly responsive to biological alignment. We have prompted ESM3
I to generate ﬂuorescent proteins with a chain of
thought. Among the generations that we synthesized, we found a bright ﬂuorescent protein at far distance (58% identity) from known ﬂuorescent proteins. Similarly distant natural ﬂuorescent proteins are separated by over ﬁve hundred million years of evolution.
V Introduction
The proteins that exist today have developed into their present forms over the course of billions of years of nat-
E ural evolution, passing through a vast evolutionary sieve. In
parallel experiments conducted over geological time, nature creates random mutations and applies selection, ﬁltering proteins by their myriad sequences, structures, and functions.
R As a result, the patterns in the proteins we observe reﬂect the
action of the deep hidden variables of the biology that have shaped their evolution across time. Gene sequencing surveys
P*Equal contribution 1EvolutionaryScale, PBC 2Arc Insti-

of Earth’s natural diversity are cataloging the sequences (1–3) and structures (4, 5) of proteins, containing billions of sequences and hundreds of millions of structures that illuminate patterns of variation across life. A consensus is building that underlying these sequences is a fundamental language of protein biology that can be understood using large language models (6–10).
A number of language models of protein sequences have now been developed and evaluated (9, 11–14). It has been found that the representations that emerge within language models reﬂect the biological structure and function of proteins (6, 15, 16), and are learned without any supervision on those properties, improving with scale (5, 17, 18). In artiﬁcial intelligence, scaling laws have been found that predict the growth in capabilities with increasing scale, describing a frontier in compute, parameters and data (19–21).
We present ESM3, a frontier multimodal generative model, that reasons over the sequences, structures, and functions of proteins. ESM3 is trained as a generative masked language model over discrete tokens for each modality. Structural reasoning is achieved by encoding three-dimensional atomic structure as discrete tokens rather than with the complex architecture and diffusion in three-dimensional space employed in recent predictive (22) and generative models (14, 23–25) of proteins. All-to-all modeling of discrete tokens is scalable, and allows ESM3 to be prompted with any combination of its modalities, enabling controllable generation of new proteins that respect combinations of prompts.
ESM3 at its largest scale was trained with 1.07×1024 FLOPs on 2.78 billion proteins and 771 billion unique tokens, and has 98 billion parameters. Scaling ESM3 to this 98 billion parameter size results in improvements in the representation of sequence, structure, and function, as well as on generative evaluations. We ﬁnd that ESM3 is highly responsive to prompts, and ﬁnds creative solutions to com-

tute 3University of California, Berkeley 4Work done dur- plex combinations of prompts, including solutions for which

ing internship at EvolutionaryScale, PBC †Correspondence to we can ﬁnd no matching structure in nature. We ﬁnd that

<arives@evolutionaryscale.ai>.

models at all scales can be aligned to better follow prompts.

Preview 2024-06-25. Pending submission to bioRxiv. Copyright Larger models are far more responsive to alignment, and

2024 by the authors.

Simulating 500 million years of evolution with a language model

show greater capability to solve the hardest prompts after To generate from ESM3, tokens are iteratively sampled.

alignment.

Starting from a sequence of all mask tokens, tokens can be

We report the generation of a new green ﬂuorescent protein (GFP) with ESM3. Fluorescent proteins are responsible for the glowing colors of jellyﬁsh and corals (26) and are important tools in modern biotechnology (27). They share an elegant structure: an eleven stranded beta barrel with a helix that threads its center, which scaffolds the formation of a light-emitting chromophore out of the protein’s own atoms. This mechanism is unique in nature—no other protein spontaneously forms a ﬂuorescent chromophore out of its own structure—suggesting that producing ﬂuorescence is hard even for nature.
Our new protein, which we have named esmGFP, has 36% sequence identity to Aequorea victoria GFP, and 58% sequence identity to the most similar known ﬂuorescent protein. Despite GFP’s intense focus as a target for protein
W engineering over several decades, as far as we are aware,
proteins this distant have only been found through the discovery of new GFPs in nature.
Similar amounts of diversiﬁcation among natural GFPs have occurred over predictable timescales. Understood in these
E terms, the generation of a new ﬂuorescent protein at this
distance from existing proteins appears to be equivalent to simulating over 500 million years of evolution.
I ESM3
ESM3 reasons over th

[A5] Predicting expression-altering promoter mutations with deep learning (Jaganathan, Kishore; Ersaro, Nicole; Novakovsky, Gherman; Wang, Yuchuan; James, )
Abstract: Only a minority of patients with rare genetic diseases are currently diagnosed by exome sequencing, suggesting that additional unrecognized pathogenic variants may reside in non-coding sequence. Here, we describe PromoterAI, a deep neural network that accurately identifies non-coding promoter variants which dysregulate gene expression. We show that promoter variants with predicted expression-altering consequences produce outlier expression at both RNA and protein levels in thousands of individuals, and that these variants experience strong negative selection in human populations. We observe that clinically relevant genes in rare disease patients are enriched for such variants and validate their functional impact through reporter assays. Our estimates suggest that promoter variation accounts for 6% of the genetic burden associated with rare diseases.
Excerpt: Cite as: K. Jaganathan et al., Science 10.1126/science.ads7373 (2025).
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
We introduce PromoterAI, a convolutional deep neural network model that uses ~20 kb of sequence context around a promoter variant to predict its expression consequence. We first trained the model to predict histone modifications, DNA accessibility, transcription factor (TF) occupancy, and strandspecific CAGE [Cap Analysis of Gene Expression (28, 29)] values around transcription start sites (TSS) at base-pair resolution. We subsequently 

[A6] Prediction and entropy of printed English (Shannon, C. E.)
Abstract: A new method of estimating the entropy and redundancy of a language is described. This method exploits the knowledge of the language statistics possessed by those who speak the language, and depends on experimental results in prediction of the next letter when the preceding text is known. Results of experiments in prediction are given, and some properties of an ideal predictor are developed.
Excerpt: Prediction and Entropy of Printed English
By C. E. SHANNON
(MaNuscript Received Sept. I5, I950)
A new method of estimating the entropy and redundancy of a language is described. This method exploits the knowledge of the language statistics possessed by those who speak the language, and depends on experimental results in prediction of the next letter when the preceding text is known, Results of experiments in prediction are given, and some properties of an ideal predictor are developed.
1. hTRODUCTIOX
I
N A previous paper' the entropy and redundancy of a language have been defined. The entropy is a statistical parameter which measures, in a certain sense, how much information is produced on the average for each letter of a text in the language. If the language is translated into binary digits (0 or 1) in the most efficient way, the entropy [{ is the average number of binary digits required per letter of the original language. The redundancy, on the other hand, measures the amount of constraint imposed on a text in the language due to its statistical structure, e.g., in English the high frequency of the letter E, the strong tendency of H to follow T or of L' to follow Q. It was estimated that when statistical effects extending over not more than eight letters are considered the entropy is roughly 2.3 bits per letter, the redundancy about 50 per cent. Since then a new method has been found for estimating these quantities, which is more sensitive and takes account of long range statistics, intluences extending over phrases, sentences, etc. This method is based on a study of the predictability of English; how well can the next letter of a text be predicted when the preceding ?{ letters are known. The results of some experiments in prediction will be given, and a theoretical analysis of some of the properties of ideal prediction. By combining the experimental and theoretical results it is possible to estimate upper and lower bounds for the entropy and redundancy. From this analysis it appears that, in ordinary literary English, the long range statistical effects (up to 100 letters) reduce the entropy to something of the order of one bit per letter, with a corresponding redundancy of roughly 75%. The redundancy may be still higher when structure extending over paragraphs, chapters, etc. is included. However, as the lengths involved are increased, the parameters in question become more
Ie. E. Shannon, "A Mathematical Theory of Communication," Bell System Technical
Journal, v. Ti , PI'. 3i9-423, 623-656, July, October, 1948. 50
sed use limited to: Columbia University Libraries. Downloaded on February 03,2025 at 21:39:29 UTC from IEEE Xplore. Re


PRJo:DICTIOl\; AXD EXTROP,," OF PRIl\;TED ENGLISH 51
erratic and uncertain, and they depend more critically on the type of text involved.
2. ENTROPY C.-\.LCUL\TIO~ FROM nm STATISTICS OF EX(;USII
One method of calculating the entropy II is by a series of approximations F«, FI , F~, ... , which successively take more and more of the statistics of the language into account and approach 1I as a limit. F.\· may be called the Y-gram entropy; it measures the amount of information or entropy due to statistics extending over ~Y adjacent letters of text. F:; is given by'
F N = -L:p(b;,j) log2Pb,(j)
- L p(b i , j) log, p(b;, j) + L p(b i) log pCb;)
i,j .
(1)
in which: b , is a block of .Y-1 letters [(Y-1)-gram]
j is an arbitrary letter following bi
p(b;, j) is the probability of the ;V-gram hi, j
hJj) is the conditional probability of letter j after the block bi,
and is given by p(b, , j)/p(b;).
The equation (1) can he interpreted as measuring the average uncertainty (conditional entropy) of the next letter j when the preceding X-1 letters are known. As X is increased, Fx includes longer and longer range statistics and the entropy, H, is given by the limiting value of F.v as ;V -+ or.; :
H = Lim F«,
N-+«J
(2)
The X-gram entropies F.v for small values of N can be calculated from standard tables of letter, digram and trigram Irequencies.' If spaces and punctuation are ignored we have a twenty-six letter alphabet and Fro may be taken (by definition) to be log, 26, or .t/ bits per letter. F t involves letter frequencies and is given by
~6
}.\ = - L p(i) log2 p(i) = 4.14 bits per letter.
i=l
The digram approximation 1"2 gives the result
F2 = - L p(i, j) log, Pi(j)
i.i
- L p(i, j) log2 p(i, j) + L p(i) log, p(i)
i,i i
= 7.70 - 4.14 = .1.56 bits per letter.
2 Fletcher Prall, "Secret and L'rgent," Blue Ribbon Books, 1942.
(3)
(4)
sed use limited to: Columbia University Libraries. Downloaded on February 03,2025 at 21:39:29 UTC from IEEE Xplore. Re


S2 THE BF.LL SYSTEM TECII:\J:CAL JOUR:\AL, JAXUARY 1951
The trigram entropy is given by
F3 = L p(i, i. k) log2 Pij(k)
i,j,k
L p(i, j, k) log- p(i, i. k) + L p(i, j) log, p(i, j) (5)
i.i,k ;,j
- 11.0 - 7.7 = 3.3
In this calculation the trigram table" used did not take into account trigrams bridging two words, such as WOW and O
