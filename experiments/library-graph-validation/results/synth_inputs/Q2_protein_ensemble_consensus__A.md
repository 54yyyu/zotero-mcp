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
Do my protein-ensemble-generation methods (AlphaFlow, P2DFlow, Distributional Graphormer, BioEmu, ESMAdam, force-guided SE(3) diffusion) actually claim to recover the true Boltzmann/equilibrium distribution, or just generate diverse conformations? Where do they disagree?

=== CONTEXT ===
[A1] Protein Conformation Generation via Force-Guided SE(3) Diffusion Models (Gu, Quanquan; Wang, Yan; Wang, Lihao; Shen, Yuning; Wang, Yiqun; Yuan, Huizhuo; )
Abstract: The conformational landscape of proteins is crucial to understanding their functionality in complex biological processes. Traditional physics-based computational methods, such as molecular dynamics (MD) simulations, suffer from rare event sampling and long equilibration time problems, hindering their applications in general protein systems. Recently, deep generative modeling techniques, especially diffusion models, have been employed to generate novel protein conformations. However, existing score-based diffusion methods cannot properly incorporate important physical prior knowledge to guide the generation process, causing large deviations in the sampled protein conformations from the equilibrium distribution. In this paper, to overcome these limitations, we propose a force-guided SE(3) diffusion model, ConfDiff, for protein conformation generation. By incorporating a force-guided network with a mixture of data-based score models, ConfDiff can can generate protein conformations with rich diversity while preserving high fidelity. Experiments on a variety of protein conformation prediction tasks, including 12 fast-folding proteins and the Bovine Pancreatic Trypsin Inhibitor (BPTI), demonstrate that our method surpasses the state-of-the-art method.
Excerpt: Protein Conformation Generation via Force-Guided SE(3) Diffusion Models

arXiv:2403.14088v1 [q-bio.BM] 21 Mar 2024

Yan Wang * 1 2 Lihao Wang * 1 Yuning Shen 1 Yiqun Wang 1 Huizhuo Yuan 3 Yue Wu 3 Quanquan Gu 1

Abstract
The conformational landscape of proteins is crucial to understanding their functionality in complex biological processes. Traditional physicsbased computational methods, such as molecular dynamics (MD) simulations, suffer from rare event sampling and long equilibration time problems, hindering their applications in general protein systems. Recently, deep generative modeling techniques, especially diffusion models, have been employed to generate novel protein conformations. However, existing score-based diffusion methods cannot properly incorporate important physical prior knowledge to guide the generation process, causing large deviations in the sampled protein conformations from the equilibrium distribution. In this paper, to overcome these limitations, we propose a force-guided SE(3) diffusion model, CONFDIFF, for protein conformation generation. By incorporating a force-guided network with a mixture of data-based score models, CONFDIFF can can generate protein conformations with rich diversity while preserving high fidelity. Experiments on a variety of protein conformation prediction tasks, including 12 fast-folding proteins and the Bovine Pancreatic Trypsin Inhibitor (BPTI), demonstrate that our method surpasses the state-of-the-art method.
1 Introduction
Proteins are dynamic macromolecules that play pivotal roles in various biological processes. Their functionality is realized primarily through conformational changes– structural alterations that enable proteins to interact with other molecules. Depicting the protein conformational landscape provides vital insights for (1) identifying potential
*Equal contribution 1ByteDance Research 2School of Mathematical Sciences, Tongji University, Shanghai (this work was done during Yan’s internship at ByteDance Research) 3Department of Computer Science, University of California, Los Angeles (this work was done during Huizhuo and Yue’s internship at ByteDance Research). Correspondence to: Quanquan Gu <quanquan.gu@bytedance.com>.
Preprint. Copyright 2024 by the author(s).

druggable sites hidden beneath the protein surface, and (2) revealing transition pathways between multiple metastable states. A comprehensive understanding of protein conformations facilitates the elucidation of biological reaction mechanisms, thereby empowering researchers to design targeted inhibitors and therapeutic agents with improved specificity and efficacy.
Traditional physics-based simulation methods such as molecular dynamics (MD) simulations have been extensively studied for protein conformation sampling. With a well-designed empirical force field and numerical integrator, the model propagates the 3D structure of a protein system over time following Newtonian mechanics. MD simulations converge towards the equilibrium distribution (i.e., the Boltzmann distribution) given sufficient time, which facilitates estimation of significant thermodynamic properties, e.g., binding free energy change (Wang et al., 2015). However, to preserve energy conservation and ensure numerical stability, the time step of MD simulations is typically only a few femtoseconds. This poses a challenge as certain biological processes of interest, such as protein folding, span much longer timescales, ranging from microseconds to seconds (Shaw et al., 2021). This results in limited sampling efficiency within conventional MD simulations, further compounded by the rare event sampling problem (TrendelkampSchroer & Noe´, 2016), impeding the research community to widely adopt MD for high throughput studies.
Building upon the cornerstone of powerful folding models (e.g., AlphaFold (Jumper et al., 2021), RoseTTAFold (Baek et al., 2021), OmegaFold (Wu et al., 2022b), etc.), several attempts have been made to tailor these deep neural networks for protein conformation sampling. By perturbing the model input, such as multiple sequence alignment (MSA) masking (Stein & Mchaourab, 2022) or clustering (Wayment-Steele et al., 2023), the folding model provides a more diverse set of possible folded structures, i.e., alternative conformations. However, this heuristic approach cannot guarantee the predicted structure to be a low energy state of the target sequence. More recently, several works have incorporated diffusion models (Song et al., 2020; Ho et al., 2020) for protein conformation generation (Jing et al., 2023; Lu et al., 2024; Zheng et al., 2023). By pretraining on a large amount of known protein structures and efficient sampling through a predefined stochastic process, these models have shown

1

Protein Conformation Generation via Force-Guided SE(3) Diffusion Models

promise in exploring diverse protein conformational states. Nevertheless, existing diffusion models fall short in utilizing important physical prior information, such as the MD force field, to guide their diffusion process, hampering their capability to faithfully sample diverse protein conformations complying with the Boltzmann distribution.
To address the aforementioned challenges, we propose a novel force-guided diffusion model, CONFDIFF, aiming to generate high fidelity protein conformations that better adhere to the Boltzmann distribution. Drawing inspiration from the contrastive energy prediction (CEP) technique (Lu et al., 2023), we employ the MD energy prior as a physicsbased preference function. By introducing an additional force guidance network during the diffusion sampling process, it prioritizes generating conformations with lower potential energy, which effectively enhances sampling quality. Our model is trained on general protein structures from the Protein Data Bank (PDB) (Berman et al., 2000) as well as self-generated conformation samples, without relying on MD simulation data (Zheng et al., 2023). To sum up, the

[A2] AlphaFold Meets Flow Matching for Generating Protein Ensembles (Jing, Bowen; Berger, Bonnie; Jaakkola, Tommi)
Abstract: The biological functions of proteins often depend on dynamic structural ensembles. In this work, we develop a flow-based generative modeling approach for learning and sampling the conformational landscapes of proteins. We repurpose highly accurate single-state predictors such as AlphaFold and ESMFold and fine-tune them under a custom flow matching framework to obtain sequence-conditoned generative models of protein structure called AlphaFLOW and ESMFLOW. When trained and evaluated on the PDB, our method provides a superior combination of precision and diversity compared to AlphaFold with MSA subsampling. When further trained on ensembles from all-atom MD, our method accurately captures conformational flexibility, positional distributions, and higherorder ensemble observables for unseen proteins. Moreover, our method can diversify a static PDB structure with faster wall-clock convergence to certain equilibrium properties than replicate MD trajectories, demonstrating its potential as a proxy for expensive physics-based simulations. Code is available at https://github.com/ bjing2016/alphaflow.
Excerpt: AlphaFold Meets Flow Matching for Generating Protein Ensembles

arXiv:2402.04845v1 [q-bio.BM] 7 Feb 2024

Bowen Jing 1 Bonnie Berger 1 2 Tommi Jaakkola 1

Abstract
The biological functions of proteins often depend on dynamic structural ensembles. In this work, we develop a flow-based generative modeling approach for learning and sampling the conformational landscapes of proteins. We repurpose highly accurate single-state predictors such as AlphaFold and ESMFold and fine-tune them under a custom flow matching framework to obtain sequence-conditoned generative models of protein structure called AlphaFLOW and ESMFLOW. When trained and evaluated on the PDB, our method provides a superior combination of precision and diversity compared to AlphaFold with MSA subsampling. When further trained on ensembles from all-atom MD, our method accurately captures conformational flexibility, positional distributions, and higherorder ensemble observables for unseen proteins. Moreover, our method can diversify a static PDB structure with faster wall-clock convergence to certain equilibrium properties than replicate MD trajectories, demonstrating its potential as a proxy for expensive physics-based simulations. Code is available at https://github.com/ bjing2016/alphaflow.
1. Introduction
Proteins adopt complex three-dimensional structures, often as members of structural ensembles with distinct states, collective motions, and disordered fluctuations, to carry out their biological functions. For example, conformational changes are critical in the function of transporters, channels, and enzymes, and the properties of equilibrium ensembles help govern the strength and selectivity of molecular interactions (Meller et al., 2023; Vo¨gele et al., 2023). While deep learning methods such as AlphaFold (Jumper et al., 2021) have excelled in the single-state modeling of experimental protein structures, they fail to account for this conformational heterogeneity (Lane, 2023; Ourmazd et al., 2022).
1CSAIL, Massachusetts Institute of Technology 2Department of Mathematics, Massachusetts Institute of Technology. Correspondence to: Bowen Jing <bjing@mit.edu>.

Hence, a method which builds upon the level of accuracy of single-structure predictors, but reveals underlying structural ensembles, would be of great value to structural biologists.
Existing machine learning approaches for generating structural ensembles have focused on inference-time interventions in AlphaFold that modify the multiple sequence alignment (MSA) input (Del Alamo et al., 2022; Stein & Mchaourab, 2022; Wayment-Steele et al., 2023), resulting in a different structure prediction for each version of the MSA. While these approaches have demonstrated some success, they suffer from two key limitations. First, by operating on the MSA, they cannot be generalized to structure predictors based on protein language models (PLMs) such as ESMFold (Lin et al., 2023) or OmegaFold (Wu et al., 2022), which have grown in popularity due to their fast runtime and ease of use. Secondly, these inference-time interventions do not provide the capability to train on protein ensembles from beyond the PDB—for example, ensembles from molecular dynamics, which are of significant scientific interest but can be extremely expensive to simulate (Shaw et al., 2010).
To address these limitations, in this work we combine AlphaFold and ESMFold with flow matching, a recent generative modeling framework (Lipman et al., 2022; Albergo & Vanden-Eijnden, 2022), to propose a principled method for sampling the conformational landscape of proteins. While AlphaFold and ESMFold were originally developed and trained as regression models that predict a single best protein structure for a given MSA or sequence input, we develop a strategy for repurposing them as (sequence-conditioned) generative models of protein structure. This synthesis relies on the key insight that iterative denoising frameworks (such as diffusion and flow-matching) provide a general recipe for converting regression models to generative models with relatively little modification to the architecture and training objective. Unlike inference-time MSA ablation, this strategy applies equally well to PLM-based predictors and can be used to train or fine-tune on arbitrary ensembles.
While flow matching has been well established for images, its application to protein structures remains nascent (Bose et al., 2023). Hence, we develop a custom flow matching framework tailored to the architecture and training practices of AlphaFold and ESMFold. Our framework leverages the polymer-structured prior distribution from harmonic diffu-

1

A q(x)

AlphaFold Meets Flow Matching for Generating Protein Ensembles

AlphaFold2

C

/ ESMFold

B

D

Figure 1. Conceptual overview of AlphaFLOW / ESMFLOW. (A) Samples are drawn from a harmonic (polymer-like) prior. (B) The sample is progressively refined or denoised under a flow field controlled by the structure prediction model (AlphaFold or ESMFold). (C) At each step, the denoised structure prediction parameterizes the direction of the flow and we interpolate the current sample towards it. (D) The final prediction is a sample from the learned distribution of structures.

sion (Jing et al., 2023), but improves over it by defining a scale-invariant noising process resilient to missing and cropped residues. These improvements directly result from the increased modeling flexibility offered by flow matching and contribute to the performance of our method.
We demonstrate the performance of our flow-matching variants of AlphaFold and ESMFold—named AlphaFLOW and ESMFLOW—in two distinct settings. First, after fine-tuning these models only on structures from the PDB, we substantially surpass the precision-diversity Pareto frontier of MSA ablation baselines on a test set of recently deposited conformationally heterogeneous proteins. Second, we showcase the ability to learn from ensembles beyond the PDB by further training on t

[A3] P2DFlow: A Protein Ensemble Generative Model with SE(3) Flow Matching (Jin, Yaowei; Huang, Qi; Song, Ziyang; Zheng, Mingyue; Teng, Dan; Shi, Qian)
Abstract: Biological processes, functions, and properties are intricately linked to the ensemble of protein conformations, rather than being solely determined by a single stable conformation. In this study, we have developed P2DFlow, a generative model based on SE(3) flow matching, to predict the structural ensembles of proteins. We specifically designed a valuable prior for the flow process and enhanced the model's ability to distinguish each intermediate state by incorporating an additional dimension to describe the ensemble data, which can reflect the physical laws governing the distribution of ensembles, so that the prior knowledge can effectively guide the generation process. When trained and evaluated on the MD datasets of ATLAS, P2DFlow outperforms other baseline models on extensive experiments, successfully capturing the observable dynamic fluctuations as evidenced in crystal structure and MD simulations. As a potential proxy agent for protein molecular simulation, the high-quality ensembles generated by P2DFlow could significantly aid in understanding protein functions across various scenarios. Code is available at https://github.com/BLEACH366/P2DFlow
Excerpt: P2DFlow: A Protein Ensemble Generative
Model with SE(3) Flow Matching
Yaowei Jin1, Qi Huang5, Ziyang Song6, Mingyue Zheng2,3,4, Dan Teng2, *, Qian Shi1, *
1 Lingang Laboratory, Shanghai 200031, China.
2 Drug Discovery and Design Center, State Key Laboratory of Drug Research,
Shanghai Institute of Materia Medica, Chinese Academy of Sciences, 555 Zuchongzhi
Road, Shanghai 201203, China.
3 University of Chinese Academy of Sciences, No.19A Yuquan Road, Beijing 100049,
China.
4 School of Chinese Materia Medica, Nanjing University of Chinese Medicine, Nanjing
210023, China.
5 Institute for Electric Light Sources, School of Information Science and Technology,
Fudan University, Shanghai 200438, P. R. China.
6 Shanghai Key Lab of Chemical Assessment and Sustainability, School of Chemical
Science and Engineering, Tongji University, Shanghai 200092, P. R. China.
*To whom correspondence should be addressed.
*(Qian Shi) E-mail: shiqian@lglab.ac.cn
*(Dan Teng) E-mail: tengdan@simm.ac.cn


Abstract
Biological processes, functions, and properties are intricately linked to the ensemble of
protein conformations, rather than being solely determined by a single stable
conformation. In this study, we have developed P2DFlow, a generative model based on
SE(3) flow matching, to predict the structural ensembles of proteins. We specifically
designed a valuable prior for the flow process and enhanced the model’s ability to
distinguish each intermediate state by incorporating an additional dimension to describe
the ensemble data, which can reflect the physical laws governing the distribution of
ensembles, so that the prior knowledge can effectively guide the generation process.
When trained and evaluated on the MD datasets of ATLAS, P2DFlow outperforms
other baseline models on extensive experiments, successfully capturing the observable
dynamic fluctuations as evidenced in crystal structure and MD simulations. As a
potential proxy agent for protein molecular simulation, the high-quality ensembles
generated by P2DFlow could significantly aid in understanding protein functions across
various scenarios. Code is available at https://github.com/BLEACH366/P2DFlow.
Key words: protein ensembles, molecular dynamics, flow matching, equivariant graph
neural network
1. Introduction
Proteins exhibit a dynamic nature that leads to the generation of diverse conformations.
Crucial biological functions are executed by relying on the distinct states, collective
motions, and disordered fluctuations within the protein ensemble[1, 2]. Thus,


A Protein Ensemble Generative Model with SE(3) Flow Matching
understanding the distribution of these ensembles is essential for elucidating the
mechanism by which proteins function in different environments. While experimental
measurements, such as crystallographic B-factors and NMR spectroscopy, can provide
some insight into conformational changes, they are limited in spatial and temporal
scale[3, 4]. Although methods like AlphaFold[5], ESMFold[6], and other deep learning
approaches have demonstrated excellent performance in predicting the crystal structure
of proteins, it is still challenging to offer diverse predictions for protein conformational
ensembles [7-10].
Several computational methods are available for conformational sampling. Traditional
approaches include Monte Carlo (MC) and molecular dynamics (MD). By providing an
initial structure of a molecular, these methods explore its conformational space based
on the forces acting upon it, which can be calculated using molecular mechanics or
quantum mechanics[11]. However, these methods face several challenges: the
computational efficiency declines rapidly as the number of atoms and degrees of
freedom increase, and both MC and MD are reliant on force fields and energy, making
it difficult to overcome high energy barriers. Consequently, these methods often
become trapped in local minima. To address these issues, enhanced sampling methods,
such as umbrella sampling [12] and meta dynamics [13], are employed for broader
exploration,.
An alternative approach involves the use of generative models leveraging machine
learning and deep learning techniques. For instance, Boltzmann generator [14] represents
one of the earliest attempt to use normalizing flow to sample system-specific
conformational distributions from random noise. However, it requires pre-acquired


A Protein Ensemble Generative Model with SE(3) Flow Matching
simulation data for specific protein systems. Multiple sequence alignment (MSA)
subsampling combined with AlphaFold [15-17] can predict different structures for each
MSA subset. While effective in certain contexts, it faces two limitations. First, the
results depend on the partition of the MSA, making it difficult to predict protein
ensembles lacking homologous proteins. Furthermore, it does not provide the approach
for training the model on protein ensemble data, which is critically important.
STR2STR [4] employs a diffusion model to learn the score matching of distributions
from random noise to protein ensemble distributions, combining stochastic perturbation
and score to guide the direction of conformational changes. It does not rely on
simulation data during either training or inference and is capable of performing zero
shot conformation sampling. However, the large variability in its predicted results
makes it difficult to focus on a particular low-energy conformation. This may be due to
inconsistencies in its prior distribution during training and sampling processes.
AlphaFlow [18] integrates AlphaFold and ESMFold with flow matching, fine-tune these
models to convert regression models to generative models. It can generate accurate and
diverse PDB structures, which can be utilized to compute additional dynamic properties.
Nonetheless, it encounters the challenges in generating non-existent intermediate states
between two or more minima, similar to the issues faced by AlphaFold in certain
proteins.
To address these limitatio

[A4] Scalable emulation of protein equilibrium ensembles with generative deep learning (Yim, Jason; Campbell, Andrew; Lewis, Sarah; Hempel, Tim; Luna, José Jiménez; Gas)
Abstract: Following the sequence and structure revolutions, predicting the dynamical mechanisms of proteins that implement biological function remains an outstanding scientific challenge. Several experimental techniques and molecular dynamics (MD) simulations can, in principle, determine conformational states, binding configurations and their probabilities, but suffer from low throughput. Here we develop a Biomolecular Emulator (BioEmu), a generative deep learning system that can generate thousands of statistically independent samples from the protein structure ensemble per hour on a single graphical processing unit. By leveraging novel training methods and vast data of protein structures, over 200 milliseconds of MD simulation, and experimental protein stabilities, BioEmu's protein ensembles represent equilibrium in a range of challenging and practically relevant metrics. Qualitatively, BioEmu samples many functionally relevant conformational changes, ranging from formation of cryptic pockets, over unfolding of specific protein regions, to large-scale domain rearrangements. Quantitatively, BioEmu samples protein conformations with relative free energy errors around 1 kcal/mol, as validated against millisecond-timescale MD simulation and experimentally-measured protein stabilities. By simultaneously emulating structural ensembles and thermodynamic properties, BioEmu reveals mechanistic insights, such as the causes for fold destabilization of mutants, and can efficiently provide experimentally-testable hypotheses.
Excerpt: Scalable emulation of protein equilibrium ensembles with
generative deep learning
Sarah Lewis1†, Tim Hempel1†, Jose ́ Jim ́enez-Luna1†, Michael Gastegger1†, Yu Xie1†, Andrew Y. K. Foong1†, Victor Garcı ́a Satorras1†, Osama Abdin1†, Bastiaan S. Veeling1†, Iryna Zaporozhets1,2, Yaoyi Chen1,2, Soojung Yang1, Arne Schneuing1, Jigyasa Nigam1, Federico Barbero1, Vincent Stimper1, Andrew Campbell1, Jason Yim1, Marten Lienen1, Yu Shi1, Shuxin Zheng1, Hannes Schulz1, Usman Munir1, Cecilia Clementi1,2, Frank No ́e1,*
1AI for Science, Microsoft Research. 2Freie Universita ̈t Berlin, Department of Physics, Arnimallee 12, 14195 Berlin. *Correspondance to franknoe@microsoft.com.
†These authors contributed equally to this work.
Abstract
Following the sequence and structure revolutions, predicting the dynamical mechanisms of proteins that implement biological function remains an outstanding scientific challenge. Several experimental techniques and molecular dynamics (MD) simulations can, in principle, determine conformational states, binding configurations and their probabilities, but suffer from low throughput. Here we develop a Biomolecular Emulator (BioEmu), a generative deep learning system that can generate thousands of statistically independent samples from the protein structure ensemble per hour on a single graphical processing unit. By leveraging novel training methods and vast data of protein structures, over 200 milliseconds of MD simulation, and experimental protein stabilities, BioEmu’s protein ensembles represent equilibrium in a range of challenging and practically relevant metrics. Qualitatively, BioEmu samples many functionally relevant conformational changes, ranging from formation of cryptic pockets, over unfolding of specific protein regions, to large-scale domain rearrangements. Quantitatively, BioEmu samples protein conformations with relative free energy errors around 1 kcal/mol, as validated against millisecond-timescale MD simulation and experimentally-measured protein stabilities. By simultaneously emulating structural ensembles and thermodynamic properties, BioEmu reveals mechanistic insights, such as the causes for fold destabilization of mutants, and can efficiently provide experimentally-testable hypotheses.
1 Introduction
Proteins and protein complexes constitute the functional building blocks of life and are at the center stage of drug development, enzymatic catalysis, biotechnological processes and biomaterials. Consequently, understanding how proteins work and how their function can be regulated or designed is one of the grand challenges in science and technology. Protein science can be characterized by three pillars of understanding: sequence, structure, and function. Next-generation sequencing has made it possible to acquire the protein sequences of entire genomes at low cost, while AlphaFold [1] and similar models [2–4] have built upon the decades of data accumulated in the
1
available under aCC-BY-NC-ND 4.0 International license.
(which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made
bioRxiv preprint doi: https://doi.org/10.1101/2024.12.05.626885; this version posted December 5, 2024. The copyright holder for this preprint


Protein Data Bank (PDB) [5] to predict 3D protein structures that in many cases match experimental accuracy within minutes. For protein function, unfortunately, methods that are both highly accurate and high-throughput are missing, and thus our understanding of how proteins work remains anecdotal. Functional descriptions such as “actin builds up muscle fibers” are human-made attributions that arise from objectively measurable mechanistic properties: (i) What are the conformational states (i.e., sets of different structures) a protein can be in? (ii) Which other molecules can a protein bind to in these different conformations? (iii) What is the probability of these conformational and binding states at a given set of experimental conditions? For example, actin exists in multiple conformational and binding states that are regulated by its cofactors ATP/ADP (Fig. 1a), providing the molecular basis of muscle growth. Available technologies that probe such conformational and binding states and their probabilities at high accuracy are currently not scalable. Single-molecule experiments can provide the full equilibrium distributions of observables such as intramolecular distances [6], but require bespoke molecular constructs and time-consuming data collection. Cryo-electron microscopy can resolve multiple conformational states of biomolecular complexes along with their probabilities [7], but running these experiments is costly both from a monetary and time perspective. Molecular Dynamics (MD) simulation is, in principle, a universal tool that allows both structure and dynamics of biomolecules to be explored at all-atom resolution. However, biomolecular forcefields are far from perfect and the sampling problem renders the study of protein folding or association via MD a feat of epic computational costs for small-sized proteins, even if special-purpose supercomputers or enhanced sampling methods are employed [8, 9]. Machine-learned coarse-grained MD models have an opportunity to achieve similar accuracy as all-atom MD at 2-3 orders of magnitude lower computational cost [10, 11] but are still under development. The grand challenge to complete our understanding of protein function thus motivates the development of a technology that can help elucidate protein conformational states and binding states, as well as their associated probabilities. This technology should ideally achieve an accuracy comparable to a converged MD simulation, or a cryo-EM experiment with multi-conformation analysis, but it should only require a few hours of wall-clock time and cost no more than a few dollars per experiment. Generative systems, such as Boltzmann Generators [12] (BGs), which can e

[A5] ESMAdam: a plug-and-play all-purpose protein ensemble generator (Yu, Zongxin; Liu, Yikai; Lin, Guang; Jiang, Wen; Chen, Ming)
Abstract: Proteins often adopt multiple ensemble conformations to perform essential functions such as catalysis, transport, and signal transduction. Traditional physics-based methods for generating these conformations, including molecular dynamics and Monte Carlo simulations, are computationally expensive and time-consuming, limiting their practicality for high-throughput applications like screening. Recent advances in machine learning, particularly deep generative models, offer a promising alternative for protein conformation ensemble generation. However, these models are often task-specific or rely on strong assumptions to generalize. Here, we introduce ESMAdam, a versatile and efficient framework for protein conformation ensemble generation. Using the ESMFold protein language model ESMFold and ADAM stochastic optimization in the continuous protein embedding space, ESMAdam addresses a wide range of ensemble generation tasks. In this work, we demonstrate several basic applications of ESMAdam, including conditional ensemble generation and CG-to-all-atom backmapping. In addition, we showcase advanced applications, such as screening alternative binding modes of protein multimers and reconstructing 3D structures from cryo-EM images. Compared to traditional physics-based methods, ESMAdam significantly reduces computational time. Unlike deep-generative-model-based approaches, it requires no retraining and easily adapts to diverse ensemble restraint conditions, making it exceptionally suited for various structure prediction and screening tasks. This plug-and-play framework represents a step toward efficient and flexible protein ensemble generation for applications in structural biology and drug discovery.
Excerpt: 1 ESMAdam: a plug-and-play all-purpose protein ensemble
2 generator
Zongxin Yu 1,+, Yikai Liu 2,+, Guang Lin 2, Wen Jiang 4, and Ming Chen *3
3
4 1Department of Engineering Sciences and Applied Math, Northwestern University, Evanston, IL, 60201
5 2Department of Mechanical Engineering, Purdue University, West Lafayette, IN, 47907
6 3Department of Chemistry, Purdue University, West Lafayette, IN, 47907
7 4Department of Biochemistry and Molecular Biology, The Pennsylvania State University, University Park, PA 16802
8 USA
9 +these authors contributed equally to this work
10 Abstract
11 Proteins often adopt multiple ensemble conformations to perform essential functions
12 such as catalysis, transport, and signal transduction. Traditional physics-based methods
13 for generating these conformations, including molecular dynamics and Monte Carlo sim
14 ulations, are computationally expensive and time-consuming, limiting their practicality for
15 high-throughput applications like screening. Recent advances in machine learning, particu
16 larly deep generative models, offer a promising alternative for protein conformation ensem
17 ble generation. However, these models are often task-specific or rely on strong assump
18 tions to generalize. Here, we introduce ESMAdam, a versatile and efficient framework for
19 protein conformation ensemble generation. Using the ESMFold protein language model
20 ESMFold and ADAM stochastic optimization in the continuous protein embedding space,
21 ESMAdam addresses a wide range of ensemble generation tasks. In this work, we demon
22 strate several basic applications of ESMAdam, including conditional ensemble generation
23 and CG-to-all-atom backmapping. In addition, we showcase advanced applications, such as
24 screening alternative binding modes of protein multimers and reconstructing 3D structures
25 from cryo-EM images. Compared to traditional physics-based methods, ESMAdam signif
26 icantly reduces computational time. Unlike deep-generative-model-based approaches, it
27 requires no retraining and easily adapts to diverse ensemble restraint conditions, making it
28 exceptionally suited for various structure prediction and screening tasks. This plug-and-play
*chen4116@purdue.edu
1
perpetuity. It is made available under aCC-BY 4.0 International license.
preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in
bioRxiv preprint doi: https://doi.org/10.1101/2025.01.19.633818; this version posted January 21, 2025. The copyright holder for this


29 framework represents a step toward efficient and flexible protein ensemble generation for
30 applications in structural biology and drug discovery.
31 Introduction
32 The inherent dynamic nature of proteins is crucial for modulating their activity, facilitating in
33 teractions with other biomolecules, and regulating intricate biological processes such as signal
34 transduction and molecular transport. Generating and analyzing protein conformational ensem
35 bles is essential for capturing this dynamism, as single static structures fail to represent the full
36 spectrum of functional states. Accurate ensemble generation provides critical insights into the
37 mechanisms underlying protein behavior in various states, enabling advancements in structural
38 biology, drug discovery, and protein design.
39 Protein conformation ensemble generation consists of a diverse tasks, each tailored to address
40 specific scientific or practical needs. One fundamental objective is to generate protein confor
41 mational ensembles that follows the Boltzmann distribution, reflecting the thermodynamic sta
42 bility of the system under physiological conditions.1–4 Beyond thermodynamic considerations,
43 many tasks focus on generating conformation ensembles that satisfy specific geometric or func
44 tional constraints. For instance, these may involve generating conformations that facilitate in
45 teractions with ligands5,6 or protein clustering,7,8 stabilize particular secondary structures,9 or
46 adopt specific topologies critical for biological function.10 Another critical task is protein con
47 formation inpainting, which involves reconstructing missing sections of a protein structure by
48 simultaneously conditioning on its sequence and the surrounding structural context. This task
49 is particularly relevant in coarse-grained molecular modeling, where atomistic details are of
50 ten simplified for computational efficiency, leading to a gap between coarse-grained structures
51 and atomistic-detail requirements in downstream applications, such as force field refinement,
52 functional analysis, and experimental validation. Acurately recovering missing atomistic de
53 tails, named as a “backmapping” process,11–13 ensures the integrity of the resulting structures
54 and enhances their utility for downstream applications. Finally, generating protein conforma
55 tion ensembles from cryo-electron microscopy (cryo-EM) images represents another significant
56 challenge.14–17 Cryo-EM records proteins and complexes in various functional states. However,
57 interpreting cryo-EM data often focuses on resolving the most stable structure and recovering
58 the underlying conformational ensemble remains complex. Accurate ensemble generation from
2
perpetuity. It is made available under aCC-BY 4.0 International license.
preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in
bioRxiv preprint doi: https://doi.org/10.1101/2025.01.19.633818; this version posted January 21, 2025. The copyright holder for this


59 cryo-EM images involves mapping 2D electron density projections of individual molecules to 3D
60 structures while accounting for experimental noise and heterogeneity. This capability is trans
61 formative for understanding flexible and multi-state proteins, as it enables the reconstruction
62 of complete

[A6] Structure Language Models for Protein Conformation Generation (Lu, Jiarui; Chen, Xiaoyin; Lu, Stephen Zhewen; Shi, Chence; Guo, Hongyu; Bengio,)
Abstract: Proteins adopt multiple structural conformations to perform their diverse biological functions, and understanding these conformations is crucial for advancing drug discovery. Traditional physics-based simulation methods often struggle with sampling equilibrium conformations and are computationally expensive. Recently, deep generative models have shown promise in generating protein conformations as a more efficient alternative. However, these methods predominantly rely on the diffusion process within a 3D geometric space, which typically centers around the vicinity of metastable states and is often inefficient in terms of runtime. In this paper, we introduce Structure Language Modeling (SLM) as a novel framework for efficient protein conformation generation. Specifically, the protein structures are first encoded into a compact latent space using a discrete variational auto-encoder, followed by conditional language modeling that effectively captures sequence-specific conformation distributions. This enables a more efficient and interpretable exploration of diverse ensemble modes compared to existing methods. Based on this general framework, we instantiate SLM with various popular LM architectures as well as proposing the ESMDiff, a novel BERT-like structure language model fine-tuned from ESM3 with masked diffusion. We verify our approach in various scenarios, including the equilibrium dynamics of BPTI, conformational change pairs, and intrinsically disordered proteins. SLM provides a highly efficient solution, offering a 20-100x speedup than existing methods in generating diverse conformations, shedding light on promising avenues for future research.
Excerpt: Preprint. Under review.
STRUCTURE LANGUAGE MODELS FOR PROTEIN
CONFORMATION GENERATION
Jiarui Lu1,2,†, Xiaoyin Chen1,2,†, Stephen Z. Lu1,3, Chence Shi1,2, Hongyu Guo4,5, Yoshua Bengio1,2,6 & Jian Tang1,6,7
1Mila - Que ́bec AI Institute, 2Universite ́ de Montre ́al, 3McGill University, 4University of Ottawa, 5National Research Council Canada, 6CIFAR AI Chair, 7HEC Montre ́al
ABSTRACT
Proteins adopt multiple structural conformations to perform their diverse biological functions, and understanding these conformations is crucial for advancing drug discovery. Traditional physics-based simulation methods often struggle with sampling equilibrium conformations and are computationally expensive. Recently, deep generative models have shown promise in generating protein conformations as a more efficient alternative. However, these methods predominantly rely on the diffusion process within a 3D geometric space, which typically centers around the vicinity of metastable states and is often inefficient in terms of runtime. In this paper, we introduce Structure Language Modeling (SLM) as a novel framework for efficient protein conformation generation. Specifically, the protein structures are first encoded into a compact latent space using a discrete variational auto-encoder, followed by conditional language modeling that effectively captures sequencespecific conformation distributions. This enables a more efficient and interpretable exploration of diverse ensemble modes compared to existing methods. Based on this general framework, we instantiate SLM with various popular LM architectures as well as proposing the ESMDiff, a novel BERT-like structure language model fine-tuned from ESM3 with masked diffusion. We verify our approach in various scenarios, including the equilibrium dynamics of BPTI, conformational change pairs, and intrinsically disordered proteins. SLM provides a highly efficient solution, offering a 20-100x speedup than existing methods in generating diverse conformations, shedding light on promising avenues for future research.
1 INTRODUCTION
Protein structure dynamics are fundamental to understanding the biological functions of proteins. The ability of proteins to adopt multiple conformations is crucial for their function in influencing interactions with other biomolecules and the environment. Traditional computational methods, such as molecular dynamics (MD) simulations, have long been used to explore these dynamics. However, these methods are computationally expensive and time-consuming. St
