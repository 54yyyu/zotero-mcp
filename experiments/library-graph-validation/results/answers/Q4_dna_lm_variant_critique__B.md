## Do the DNA language model papers agree they predict variant effects well, and how does the benchmark compare to the methods' own claims?

**The method papers make confident, largely self-consistent claims of strong performance — but those claims are scoped, and the benchmarking paper qualifies them substantially.**

### What the method papers claim about themselves

The two GPN-lineage method papers report strong results, but each carefully bounds its claim:

- *DNA language models are powerful predictors of genome-wide variant effects* (GPN) claims state-of-the-art VEP performance — but explicitly only **for *Arabidopsis thaliana***, a compact genome, and calls itself a "proof-of-concept model" that is 200x smaller than the largest protein LMs, with "room for improving its training scheme" [B6, B27, B35]. It frames the human genome as "substantially harder" and unsolved [B29].
- *A DNA language model based on multispecies alignment* (GPN-MSA) opens by conceding that "DNA language models have not yet shown a competitive edge for complex genomes such as that of humans," then claims GPN-MSA achieves "outstanding performance" on clinical, functional-assay, and population-genomic benchmarks (ClinVar, COSMIC, OMIM, DMS, DepMap, gnomAD) [B18, B19, B13, B26]. This is the strongest self-claim in the set.

So both method papers agree DNA LMs *can* predict variant effects well — but only GPN-MSA claims this for humans, and even it hedges in important ways (see below).

### Internal caveats already present in the method papers

The method papers are not uniformly triumphant. GPN-MSA itself cautions that "none of the models performs exceedingly well relative to ClinVar results," that sequence models like GPN-MSA are "less interpretable than functional genomics models such as SpliceAI and Enformer," and that jointly modeling sequence and functional genomics "may have the biggest impact" [B7, B5]. It also warns its scores are not "calibrated fitness estimates" and should only be used to compare variants relatively [B13]. These are meaningful concessions that DNA LMs are not a clear, universal winner.

### How the benchmarking paper compares

*Benchmarking DNA Sequence Models for Causal Regulatory Variant Prediction* (TraitGym) **partially corroborates and partially tempers** the method claims, and is more granular than any single-method paper:

- It confirms alignment-based models — **CADD and GPN-MSA — compare favorably for Mendelian traits and complex disease traits** [B25], consistent with GPN-MSA's self-claims.
- But it finds **functional-genomics-supervised models (Enformer, Borzoi) perform better for complex non-disease traits** [B25] — i.e., the self-supervised DNA LM advantage is trait-class dependent, not blanket. The benchmark explains this via mechanism: Mendelian variants are easier because they have larger effects and stronger purifying selection (the signal self-supervised models pick up), while complex traits differ [B8].
- It reports that **Evo2 shows gains with scale but still lags alignment-based models, struggling with enhancer variants** [B25], which complicates the method papers' implicit "scaling will keep improving DNA LMs" narrative [B6, B26].

The benchmark also implicitly critiques prior evaluation practice: it argues existing benchmarks (BEND, GV-Rep) lack adequate baselines — notably omitting conservation-based models that "are usually strong for this task" — and that ClinVar is skewed toward coding/splice variants with almost no intergenic/promoter variants [B23, B34]. This frames the method papers' headline numbers as needing stronger, conservation-aware baselines to be trusted.

### Bottom line

There is broad agreement that DNA LMs (especially alignment-based GPN-MSA) predict variant effects well **for Mendelian and disease variants**. The disagreement/tension is in scope: the original GPN claim was plant-only [B6]; GPN-MSA's "outstanding" human claim is real but trait-class- and baseline-dependent, and the benchmark shows DNA LMs are *not* uniformly best — functional-genomics models win for complex non-disease traits and conservation baselines remain strong [B25, B23]. Note three of these papers share Benegas/Song authorship [B10, B19, B25], so the "benchmark vs. method" comparison is partly self-critique rather than fully independent.

*(The promoter-mutation and protein-conformation entries [B11, B17, B21, B32] do not bear on DNA-LM variant-effect claims and are not used here.)*
