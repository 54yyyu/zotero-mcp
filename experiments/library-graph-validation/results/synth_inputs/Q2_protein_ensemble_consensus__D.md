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
=== AlphaFold Meets Flow Matching for Generating Protein Ensembles (Jing, Bowen; Berger, Bonnie; Jaakkola, Tommi) ===
AlphaFold Meets Flow Matching for Generating Protein Ensembles

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
We demonstrate the performance of our flow-matching variants of AlphaFold and ESMFold—named AlphaFLOW and ESMFLOW—in two distinct settings. First, after fine-tuning these models only on structures from the PDB, we substantially surpass the precision-diversity Pareto frontier of MSA ablation baselines on a test set of recently deposited conformationally heterogeneous proteins. Second, we showcase the ability to learn from ensembles beyond the PDB by further training on the ATLAS dataset (Vander Meersche et al., 2023) of molecular dynamics simulations. When evaluated on test proteins structurally dissimilar from the training set, AlphaFLOW substantially surpasses the MSA baselines in the prediction of conformational flexibility, distributional modeling of atomic positions, and replication of higherorder ensemble observables such as intermittent contacts and solvent exposure. Furthermore, when a static PDB structure is provided as a template, sampling from AlphaFLOW provides faster wall-clock convergence to many equilibrium properties than running molecular dynamics (MD) simulation starting from that structure. Thus, our method can be used in place of expensive simulations to diversify and obtain equilibrium ensembles of solved protein structures.
2. Background
Protein structure prediction. The modern approach for protein structure prediction was pioneered by AlphaFold (Jumper et al., 2021), which takes as input (1) the protein

sequence, (2) a MSA of evolutionarily related sequences, and optionally (3) a template structure of a related protein, and predicts the all-atom 3D coordinates of single protein structure. AlphaFold was developed and trained in an endto-end fashion under a regression-like FAPE loss with structures from the PDB. Later works, such as ESMFold (Lin et al., 2023) and OmegaFold (Wu et al., 2022), modified the pipeline by substituting the MSA with embeddings from a protein language model (PLM) and eschewing the template input, but otherwise kept the same architecture and training framework as AlphaFold.
Modeling protein ensembles. In the post-AlphaFold era, several works have emphasized diversifying highly accurate single-structure predictions to reflect underlying conformational heterogeneity (Lane, 2023; Chakravarty & Porter, 2022; Saldan˜o et al., 2022; Xie & Huang, 2023; Brotzakis et al., 2023; Bryant, 2023; Porter et al., 2023). Most prominently, Del Alamo et al. (2022) demonstrated that multiple functional states could be obtained by subsampling the MSA input to AlphaFold. Since then, MSA subsampling has become the de-facto standard methodology and has been employed to study conformational states of kinases (Faezov & Dunbrack Jr, 2023; Herrington et al., 2023; Casadevall et al., 2023), variant effects on conformational states (da Silva et al., 2023), and to seed molecular dynamics simulations (Vani et al., 2023). Alternative approaches have also been proposed in the form of point mutations to the MSA (Stein & Mchaourab, 2022; 2023) and MSA clustering (Wayment-Steele et al., 2023). .
An emerging line of work seeks to directly train sequenceto-structure generative models of protein ensembles. EigenFold (Jing et al., 2023) and Distributional Graphormer (Zheng et al., 2023) use harmonic diffusion and SE(3) diffu-

2

AlphaFold Meets Flow Matching for Generating Protein Ensembles

sion (Yim et al., 2023), respectively, to generate ensembles. SENS (Lu et al., 2023) is a local generative model that diversifies single starting structures via local exploration of the conformational landscape. However, these models have yet to show convincing validations or comparisons with MSA subsampling methods on PDB test sets.
A related but separate line of work has focused on learning generative models of Boltzmann distributions as proxies for expensive molecular dynamics simulation. These models were initially conceived as normalizing flows that provided exact likelihoods and thus a means to train with energies and reweigh samples at inference time (Noe´ et al., 2019; Ko¨hler et al., 2021; Midgley et al., 2022; Abdin & Kim, 2023; Felardos et al., 2023). However, these normalizing flows have proven difficult to scale beyond small molecules and toy systems. More recently, the proliferation of diffusion models has shifted the focus of this line of work towards scalability and generalization (Arts et al., 2023; Zheng et al., 2023) rather than exact likelihoods. Our method, when trained on MD ensembles, can be viewed as belonging to this new generation of Boltzmann-targeting generative models.
Flow matching (Lipman et al., 2022; Albergo & VandenEijnden, 2022; Albergo et al., 2023; Liu et al., 2022) is a generative modeling paradigm that resembles and builds upon the significant success of diffusion models (Ho et al., 2020; Song et al., 2021) in image and molecule domains. The fundamental object in flow matching is a conditional probability path pt(x | x1), t ∈ [0, 1]: a family of densities conditioned on a data point x1 ∼ pdata which interpolates between a shared prior distribution p0(x | x1) = q(x) and an approximate Dirac p1(x | x1) ≈ δ(x − x1). Given a conditional vector field ut(x | x1) that generates the time evolution of pt(x | x1), one then learns the marginal vector field with a neural network:
vˆ(x, t; θ) ≈ v(x, t) := Ex1∼pt(x1|x)[ut(x | x1)] (1)
At convergence, the learned vector field vˆ(x, t; θ) is a neural ODE that evolves the prior distribution q(x) to the data distribution pdata(x). Score-matching in diffusion models can be seen as a special case of flow matching; however, as discussed in Section 3.3, flow matching circumvents certain difficulties that would otherwise arise with diffusion.
3. Method
3.1. AlphaFold as a Denoising Model
Given a protein sequence A of amino acid tokens, our objective is to model the distribution p(x | A) over 3D coordinates x ∈ R3×N which represents the structural ensemble of that protein sequence. Considering the enormous intellectual efforts that went into a deterministic sequenceto-structure model (i.e., AlphaFold), developing a distri-

Text-to-image generative model

yorkshire terrier

UNet

Sequence-to-structure generative model

MEEKLKKTKIIFVVGG…

AlphaFold

Figure 2. AlphaFold as a denoising model. Just as (diffusionbased) text-to-image generative models are simply neural networks that denoise images (with text input), a modified AlphaFold that ingests noisy structures and predicts clean structures (with sequence input) immediately provides a sequence-to-structure generative model—when trained under an appropriate framework.

butional model of equivalent accuracy and generalization ability would appear to pose a considerable challenge. Our solution is to leverage recent conceptual advances in generative modeling in order to simply repurpose AlphaFold— nearly out of the box—as a generative model.
Consider, for example, the (simplified) architecture of prototypical text-to-image diffusion models (Ho et al., 2020; Rombach et al., 2022), which aim to model conditional distributions p(x | s) of images x conditioned on text prompt s. At the heart of these models lies a denoising neural network (e.g., a UNet) which ingests a noisy image, along with a text prompt, to predict a clean image. Conditioned on these inputs, such models are otherwise are trained with simple, regression-like MSE objectives. Analogously, a protein structure predictor trained on a regression-like loss—like AlphaFold or ESMFold—can be converted to a denoising model simply by supplying an additional, noisy structure input (Figure 2). Not coincidentally, this is reminiscent of the idea of template structures employed by certain AlphaFold workflows. Thus, we develop an input embedding module very similar to AlphaFold’s template embedding stack and prepend it to the pairwise folding trunks of AlphaFold and ESMFold (details in Appendix A.1). By doing so, we obtain structure denoising architectures that are thin wrappers around well-validated single-structure predictors.
With these architectural modifications, we are ready to plug AlphaFold and ESMFold into any iterative denoising-based generative modeling framework. Next, we will see how this concretely applies to flow matching for protein ensembles.

3

AlphaFold Meets Flow Matching for Generating Protein Ensembles

3.2. Flow Matching for Protein Ensembles
Designing a flow-matching generative framework amounts to the choice of a conditional probability path pt(x | x1) and its corresponding vector field ut(x | x1). Inspired by the interpolant-based perspective on flow matching (Albergo & Vanden-Eijnden, 2022), we define the conditional probability path by sampling noise x0 from the prior q(x0) and interpolating linearly with the data point x1:

x | x1, t = (1 − t) · x0 + t · x1, x0 ∼ q(x0) (2)

This probability path is associated with the vector field

ut(x | x1) = (x1 − x)/(1 − t)

(3)

which matches the CondOT path and field proposed in (for example) Pooladian et al. (2023). Customarily, we then learn a neural network to approximate the marginal vector field according to Equation 1. However, if we instead define a neural network xˆ1(x, t; θ) and reparameterize via

vˆ(x, t; θ) = (xˆ1(x, t; θ) − x)/(1 − t)

(4)

then rearrangements of Equations 1 and 4 reveal that we can equivalently learn the expectation of x1:

xˆ1(x, t; θ) ≈ Ex1∼pt(x1|x)[x1]

(5)

This reparameterization is identical—up to the choice of probability path pt(x1 | x)—to that employed for image diffusion models (Ho et al., 2020). In our setting, since x1 refers to samples from the data distribution (i.e., protein structures), this allows the AlphaFold-based architectures discussed previously to be immediately used as the the denoising model xˆ1(x, t; θ), with x as the noisy input and t as an additional time embedding.

To apply flow matching to protein structures, we describe a structure by the 3D coordinates of its β-carbons (α-carbon for glycine): x ∈ RN×3. (We choose β-carbons because these are the inputs to the template embedding stack.) We then define the prior distribution q(x) over the positions of these β-carbons to be a harmonic prior (Jing et al., 2023):

q(x) ∝ exp

α −
2

N −1
∥xi

−

xi+1∥2

(6)

i=1

This prior ensures that samples along the conditional probability path, and hence inputs to the neural network, always remain polymer-like, physically plausible 3D structures.

The parameterization of learning the conditional expectation of x1 (Equation 5) suggests that the neural network should be trained with an MSE loss. However, there are several issues with this direct approach. (1) The structure prediction networks not only predict β-carbon coordinates,

but also all-atom coordinates and residue frames. (2) The input to the network is SE(3)-invariant by design, which makes training with MSE loss unsuitable without further correction (Appendix A.2). Finally, (3) the networks obtain best performance (and were orginally trained) with the SE(3)-invariant Frame Aligned Point Error (FAPE) loss. To reconcile these issues with the flow-matching framework, we redefine the space of protein structures to be the quotient space R3×N /SE(3), with the prior distribution projected to this space. We redefine the interpolation between two points in this space to be linear interpolation in R3 after RMSDalignment. Further, because the quotient space is no longer a vector space, there is no longer a notion of “expectation” of a distribution; instead, we aim to learn the more general Fre´chet mean of the conditional distribution p(x1 | x):

xˆ1(x,

t;

θ)

≈

min
xˆ1

Ex1 ∼pt (x1 |x)

FAPE2(x1, xˆ1)

(7)

where we leverage the property that FAPE is a valid metric (Jumper et al., 2021) to define a Fre´chet mean. To learn this target, we use a training loss identical to the original FAPE, except now squared. The final result for the training and inference procedures are provided in Algorithms 1 and 2. An important implication of this modified framework is that while our model is faithfully supervised on all-atom coordinates, it technically is learning the distribution only over β-carbon coordinates. These procedures and their subtleties are more fully discussed in Appendix A.2.

Algorithm 1 TRAINING
Input: Training examples of structures, sequences, and MSAs {(Si, Ai, Mi)} for all (Si, Ai, Mi) do
Extract x1 ← BetaCarbons(Si) Sample x0 ∼ HarmonicPrior(length(Ai)) Align x0 ← RMSDAlign(x0, x1) Sample t ∼ Uniform[0, 1] In

=== P2DFlow: A Protein Ensemble Generative Model with SE(3) Flow Matching (Jin, Yaowei; Huang, Qi; Song, Ziyang; Zheng, Mingyue; Teng, Dan; Shi, Qian) ===
P2DFlow: A Protein Ensemble Generative
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
To address these limitations, we propose a new framework that samples protein
ensembles via a SE(3) equivariant flow matching model named P2DFlow (Fig. 1).
P2DFlow is trained on MD simulation data, which contains different macro states of
proteins (as seen in Fig. 4), allowing it to learn the conformational changes within
protein ensembles. To enhance the model's generalization ability while reducing the


A Protein Ensemble Generative Model with SE(3) Flow Matching
difficulty of training, we use a strong prior, ESMFold predictions with perturbations of
coordinates (detailed in Sec. 2.1), to obtain an estimated structure. Compared to
Gaussian prior and harmonic prior, the stronger prior that we use guides the model to
generate accurate structures by introducing more precise biases in bond lengths and
bond angles. This approach differs significantly from AlphaFlow, which uses harmonic
prior and relies on AlphaFold or ESMFold as the main component to control structure.
In contrast, P2DFlow employs ESMFold predictions with perturbations as a stronger
prior and utilizes a new SE(3) equivariant block to adjust the structure. To distinguish
between various conformations of the same protein, we introduce a new dimension
called ‘approximate energy’, which maps MD results onto a low-dimensional plane and
uses the density to represent probability (detailed in Sec. 2.2). It guides the model in
generating conformations with distinct energies, thereby avoiding the generation of
non-existent intermediate states.
We evaluate the performance of P2DFlow against AlphaFlow and STR2STR, two
representative and advanced models for protein ensemble generation. Based on
extensive experiments on the test dataset of ATLAS[19], it is evident that P2DFlow
outperforms other baseline models on the metrics which can reflect the fidelity and
dynamic properties of the generated ensembles compared to the ground truth.
Visualization results indicate that P2DFlow effectively captures important changes in
residue contact and more accurately recovers protein ensembles distributions.
Moreover, the ablation experiment shows that the ‘approximate energy’ significantly
aids in constructing the conformation distribution of proteins. Thus, P2DFlow can be
used to predict protein ensembles without the need for expensive MD simulations.


A Protein Ensemble Generative Model with SE(3) Flow Matching
Figure 1. Conceptual overview of P2DFlow. Protein structures are firstly sampled
from a prior based on perturbed ESMFold predictions. Subsequently, an ‘approximate
energy’ sampling serves as a condition to guide the generation process to produce
ensemble distribution.
2. Method
2.1 SE(3) Flow Matching
A flow-matching generative framework[20] aims to learn continuous normalizing flows
(CNFs) [21], generating target distributions by predicting a vector field and integrating
the ordinary differential equation (ODE).
For a flow φ(x, t), there is a related vector field v(x, t), defined by the following ODE:
dφ(x, t)
dt = v(x, t) , φ(x, 0) = x0 (1)
where t ∈ [0, 1] is a continuous parameter, x0 is a sample of prior distribution p0.
We can use the flow to transform the simple prior p0 towards the data distribution p1
by integrating the vector field v(x, t):


A Protein Ensemble Generative Model with SE(3) Flow Matching
x1 − x0 = ∫ v(x, t) dt
1
0
(2)
For the flow, since it only fixes the distribution at t = 1 and the prior at t = 0 , the
interpolation process between them can be manually defined. In our method, we choose
ESMFold prediction with perturbations as the prior. It means that, for a given protein
sequence requiring ensemble generation, we first use ESMFold to predict the static
structure. Then, Gaussian noise with specific variance is added to the coordinates of
each residue to generate the prior distribution. Previous studies has shown that the prior
distribution closer to data distribution which needs to be learned can lead to superior
performance[22], so we identify this as an especially valuable inductive bias since it will
offer a suitable bond length and torsion angle as the initial value of the iteration,
compared with Gaussian prior and harmonic prior [23, 24].
For the interpolation process, following previous work [25-27], the backbone atom
positions of each residue are parameterized by an orientation preserving rigid
transformation T ∈ SE(3), that maps from fixed coordinates of N∗ , Cα∗ , C∗ , O∗
centered at Cα∗ = (0,0,0). Each frame T = (r, x) consists of a rotation r ∈ SO(3) and
a translation vector x ∈ R3. As for the side chain of the residue, torsion angles θ are
used to define the twists. Since our focus lies on the backbone of protein, we only
interpolate for translation and rotation accompanied by the use of optimal transport (OT)
path [28]:
Translations (R3): xt = (1 − t)x0 + tx1 (3)
Rotations ( SO(3) ): rt = expr0 (t logr0(r1)) (4)


A Protein Ensemble Generative Model with SE(3) Flow Matching
We calculate expr0 and logr0 using Rodrigues’ formula [29]. The vector field for
translation and rotation are then computed as follow:
ẋt = x1 − xt
1 − t , ṙ = logrt( r1)
1 − t (5)
These vector fields are learned with SE(3) equivariant neural network (mentioned in
Section 2.2). For the torsion angle within residues, we predict it using the node
embedding of the corresponding residues with a Multi-Layer Perceptron (MLP) due to
its SE(3) invariant[30], which is similar to the approach utilized in AlphaFold.
With these considerations and calculations mentioned above, the training loss of
P2DFlow is defined as:
L = Εt,p1,p0{ 1
(1 − t)2 (‖ ̂x1 − x1‖2 + ‖r̂1 − r1‖2)
+ α(‖ ̂C1( ̂x1, r̂1, ̂θ) − C1‖2 + ‖ ̂D1( ̂x1, r̂1, ̂θ) − D1‖2)} (6)
Where ( ̂x1, r̂1, ̂θ) refer to predicted translation, rotation, and torsion at t = 1 .
( ̂C1, ̂D1) refer to the atom coordinates and distance matrix of all atoms, which is
recovered using ( ̂x1, r̂1, ̂θ) and standard frames proposed in AlphaFold at t = 1. α is
a hyperparameter to adjust the auxiliary loss with the vector field loss. Furthermore, we
subtract the Center of Mass (CoM) from the prior sample and each protein data to
achieve a zero CoM. This step is crucial for ensuring that the distribution of the sampled
results remains SE(3) equivariant.
2.2 Network Architecture
To learn the aforementioned vector field, we developed P2DFlow, a flow-matching
model that is SE(3) equivariant. The model's equivariance is achieved by the stacking
of Invariant Point Attention (IPA) module [5] and E(n) equivariant Graph Neural


A Protein Ensemble Generative Model with SE(3) Flow Matching
Network (EGNN) module [31]. These components are utilized to encode the spatial
features of the protein. In the molecular graph representation, residues are considered
as nodes, and edges are defined by two criteria: the cutoff coordinate distance between
adjacent residues and the cutoff sequence distance. We utilize the union of these criteria
to establish the edges.
Figure 2. Workflow of P2DFlow. Protein sequence and ‘approximate energy’ are
inputs of P2DFlow. We utilize ESM-2 to get embeddings from sequence, and use
ESMFold + Perturb to get the noised structure as the prior distribution, then apply a
SE(3) equivariant block composed with IPA and EGNN to predict the vector filed for
the flow process. The differences between training and sampling are marked with
different border colors.
The inputs of the model include sequence and ‘approximate energy’ (Figure 2). To
improve the generalization ability of the model, we utilize the ESM-2 protein language
model to generate the initial Node Representation (Node Repr.) and Pair Representation
(Pair Repr.) for residues. After successive aggregations and updates of the Node Repr.


A Protein Ensemble Generative Model with SE(3) Flow Matching
and Pair Repr. using the SE(3) Equivariant Block, we obtain a compressed, full-graph
representation of the protein structure, which is sampled from prior of ESMFold
prediction with perturbation. Subsequently, we employ the representation to predict the
vector fields associated with translation and rotation.
To distinguish structures with different energies in the conformation set, we introduce
the concept of ‘approximate energy’. We project the molecular dynamics (MD)
simulation ensembles onto a two-dimensional plane defined by the radius of gyration
(RG) and the root-mean-square deviation (RMSD) relative to the crystal structure. We
then compute the Gaussian kernel density of this 2D map and apply the Boltzmann
Equation to convert the density values into ‘approximate energy’ values after
normalization.
The sampling process is illustrated in Figure 2. Given a protein sequence as input, the
process begins with the prediction of a stable structure using ESMFold. Subsequently,
we add Gaussian noise with the same variance used during training to perturb the
structure, so that we can get the initial perturbed structure from the specialized prior
distribution. We then sample the 'approximate energy' from Boltzmann Distribution,
and keep it unchanged throughout the entire autoregressive sampling process to guide
the generation. Following the prediction of vector fields, a Euler integration step is
applied to solve the ODE. Ultimately, we reconstruct the coordinates of each atom
based on the translation, rotation, and torsion angles, utilizing the frame representation
provided by AlphaFold, to derive the protein conformation structure. After conducting
a sufficient number of samples, we obtain the predicted protein ensembles.


A Protein Ensemble Generative Model with SE(3) Flow Matching
Other operational details are in line with those of FrameFlow [27]. We modify the loss
function by substituting 1/(1 − t)2 with 1/(1 − min{ t, 0.9})2 to prevent the loss
from exploding. Furthermore, we conduct a pre-alignment[32, 33] to align x0 and x1
using the Kabsch algorithm. Specifically, we solve r∗ = argminr∈SO(3)‖rx0 − x1‖2
and use the aligned position r∗x0 during training.
3. Experiments
3.1 Setup
Training To learn the dynamic distribution of protein ensembles, we utilize ATLAS
dataset[19], which contains ~1300 MD simulation results for each kind of protein. To
select representative structures from the conformations, we compute the ‘approximate
energy’ (mentioned in section 2.2) for each of them, then choose 11 structures at equal
intervals of ‘approximate energy’. For evaluation, we randomly choose ~100 ensembles
from ATLAS dataset excluding the training set. Our model is trained for 2 days using
4 NVIDIA A100-80G GPUs.
Metrics To assess the performance of P2DFlow, we use evaluation metrics which can
be categorized into: (a) Fidelity. Reflects the distributional gap between sampled
ensemble and reference MD simulation, which contains Jensen-Shannon divergence of
pairwise distance[4] (PWD J) and radius of gyration (RG J), root mean Wasserstein
distance (RMWD W2) and root mean square fluctuation (RMSF), intersection over
union of Ramachandran plots (RP IoU); (b) Dynamics. Shows the observable structural
change which is often associated with thermal fluctuations, such as the Weak Contacts
Ja and Transient Contacts Ja [18].


A Protein Ensemble Generative Model with SE(3) Flow Matching
Baseline We compare our results with Alphaflow and STR2STR. Alphaflow is a flow
matching model based on fine-tuning of AlphaFold and ESMFold, and STR2STR is a
diffusion model which changes inference stage by adding AlphaFold prediction as the
initial structure. We use their pretrained weights offered at GitHub to sample ensembles
on the test set of ~100 proteins from ATLAS dataset.
3.2 Results
Table 1 shows the evaluation results for P2DFlow, AlphaFlow and STR2STR. For each
method, we compare the predicted ensemble with the ground truth MD ensembles and
evaluate them on various metrics. We can see that P2DFlow performs better than
AlphaFlow and STR2STR in PWD J , RMWD W2 , RMSF, RP IoU and Weak
Contacts Ja. This indicates that P2DFlow more realistically reflect the physical space
distrib

=== Ab initio characterization of protein molecular dynamics with AI2BMD (Wang, Tong; He, Xinheng; Li, Mingyu; Li, Yatao; Bi, Ran; Wang, Yusong; Cheng, Ch) ===
Nature | www.nature.com | 1
Article
Ab initio characterization of protein
molecular dynamics with AI2BMD
Tong Wang1,2,3 ✉, Xinheng He1,2, Mingyu Li1,2, Yatao Li1,2, Ran Bi1, Yusong Wang1, Chaoran Cheng1, Xiangzhen Shen1, Jiawei Meng1, He Zhang1, Haiguang Liu1, Zun Wang1, Shaoning Li1, Bin Shao1,3✉ & Tie-Yan Liu1
Biomolecular dynamics simulation is a fundamental technology for life sciences research, and its usefulness depends on its accuracy and efficiency1–3. Classical molecular dynamics simulation is fast but lacks chemical accuracy4,5. Quantum chemistry methods such as density functional theory can reach chemical accuracy but cannot scale to support large biomolecules6. Here we introduce an artificial intelligence-based ab initio biomolecular dynamics system (AI2BMD) that can efficiently simulate full-atom large biomolecules with ab initio accuracy. AI2BMD uses a protein fragmentation scheme and a machine learning force field7 to achieve generalizable ab initio accuracy for energy and force calculations for various proteins comprising more than 10,000 atoms. Compared to density functional theory, it reduces the computational time by several orders of magnitude. With several hundred nanoseconds of dynamics simulations, AI2BMD demonstrated its ability to efficiently explore the conformational space of peptides and proteins, deriving accurate 3J couplings that match nuclear magnetic resonance experiments, and showing protein folding and unfolding processes. Furthermore, AI2BMD enables precise free-energy calculations for protein folding, and the estimated thermodynamic properties are well aligned with experiments. AI2BMD could potentially complement wet-lab experiments, detect the dynamic processes of bioactivities and enable biomedical research that is impossible to conduct at present.
The research paradigm of life sciences is shifting as the accuracy of computational simulation models is becoming indistinguishable from that of wet-lab experiments1,2. Among the computational models, molecular dynamics (MD) simulation, as the ‘computational microscope’, is of particular interest for understanding how life works3,5,8. MD simulations study the dynamic evolution of molecules by moving the atoms in a molecular system. They differ in the way that the forces are calculated3. In classical MD, forces are calculated using a prescribed interatomic potential function, whereas in ab initio MD (AIMD), forces are calculated using the potential derived from the electronic structure of molecules6. AIMD provides accurate characterization of molecules; the main challenge of applying AIMD to biomolecular simulation is scalability. On the one hand, the widely used quantum chemistry methods for AIMD are computationally expensive; for example, with the system size N,the time complexity of density functional theory (DFT) is about O(N3), and that of the coupled cluster method with the inclusion of single, double and perturbative triple excitations (CCSD(T)) is O(N7). On the other hand, observing important conformational changes for biomolecules such as proteins usually requires billions of steps with at least cubic time complexity for thousands of atoms4. Until now, scalable and accurate AIMD for biomolecules has not existed. To alleviate the dilemma, machine learning force fields (MLFFs) trained on data generated at the DFT level provide accurate force
calculations at a much lower cost and can be applied to small peptides and proteins7,9,10. The ability to generalize is the key challenge for the applicability and robustness for biomolecule simulations11. First, as the conformational space of a molecule is enormous, training on limited conformations of one kind of molecule and adapting it for conformational space exploration of other kinds of molecule is difficult5. Second, as the time and cost for generating data with DFT increase cubically with the size of the molecules, the lack of training data hinders the application of MLFFs for large biomolecules11. Furthermore, it is impossible to train a specific model for each kind of protein, and a unified solution with good generalization ability is needed. In this study, we propose AI2BMD, a generalizable solution for efficiently simulating a wide range of full-atom proteins with ab initio accuracy, surrounded by an explicit solvent modelled by a polarizable force field (Fig. 1). A generalizable protein fragmentation approach splits proteins into overlapped protein units. Simulations are performed by the AI2BMD simulation system. At each simulation step, the AI2BMD potential, based on ViSNet7, calculates the energy and atomic forces for the protein with ab initio accuracy. Through comprehensive analysis from both kinetics and thermodynamics perspectives, AI2BMD exhibits good alignment with wet-lab experimental data, such as the melting temperature of fast-folding proteins, and detects different phenomena than molecular mechanics (MM).
https://doi.org/10.1038/s41586-024-08127-z
Received: 31 March 2023
Accepted: 26 September 2024
Published online: xx xx xxxx
Open access
Check for updates
1Microsoft Research, Beijing, China. 2These authors contributed equally: Tong Wang, Xinheng He, Mingyu Li, Yatao Li. 3These authors jointly supervised this work: Tong Wang, Bin Shao. ✉e-mail: tongwang.bio@outlook.com; binshao@live.com


2 | Nature | www.nature.com
Article
Energy and force calculations
To provide a generalizable solution for accurately simulating proteins, AI2BMD adopts a universal protein fragmentation approach. Although generating samples for a specific kind of protein and training MLFF on them is straightforward, simulating other kinds of protein with the MLFF usually leads to simulation collapse12 (Supplementary Fig. 1). Furthermore, it is computationally prohibitive to generate training data at the DFT level for large proteins. Thus, we fragment proteins into smaller units, specifically dipeptides, calculate intra- and inter-unit interactions, and then assemble them to determine the protein energy and forces acting on the atoms (see Methods for more details). Our fragmentation approach contains only 21 kinds of protein unit, and all protein units have similar and moderate numbers of atoms (range from 12 to 36), which is convenient for DFT data generation and MLFF training. Moreover, all kinds of protein can be broken down into the 21 kinds of protein unit, indicating that this is a generalizable fragmentation approach. We built a comprehensively sampled protein unit dataset. During dataset construction, we scanned the main-chain dihedrals of all protein units to cover a wide range of conformations and ran AIMD simulations with the 6-31g* basis set and the M06-2X functional13, as this functional models dispersion and weak interactions well and has been widely used for biomolecules14,15. We obtained 20.88 million samples (see Methods for more details). The whole dataset was split into training, validation and test sets to train ViSNet7 models as the AI2BMD potential. The model encodes physics-informed molecular representations and calculates four-body interactions with linear time complexity. The model subsequently generates precise force and energy estimations based on the atom types and the coordinates as inputs (Methods and Extended Data Fig. 1). The performance of the AI2BMD potential was compared with that of the conventional MM force field on the test set, with the results presented in Supplementary Table 1.
In terms of energy mean absolute error (MAE), the AI2BMD potential outperformed the MM force field by approximately two orders of magnitude (AI2BMD: 0.045 kcal mol−1, MM: 3.198 kcal mol−1). The AI2BMD potential also demonstrated superior performance for the force MAE (0.078 kcal mol−1 Å−1) compared to MM (8.125 kcal mol−1 Å−1). Overall, the AI2BMD potential offers accurate predictions for both potential energy and atomic forces for protein units. On the basis of the AI2BMD potential, we developed an MD simulation system with a polarizable solvent described by the AMOEBA force field16 (see the Methods for further details). Then we conducted simulations for 9 proteins with the number of atoms ranging from 175 to 13,728 (Fig. 2a; see the Methods for more details). Each protein was assessed with 5 folded, 5 unfolded and 10 intermediate structures derived from replica-exchange MD simulations as the initial conformations, and 10 AI2BMD simulation steps were run resulting in 200 structures per protein. The AI2BMD simulation system’s ability to reach ab initio accuracy was evaluated by comparing its results to those calculated by DFT. Calculations by MM act as a control (Fig. 2b–e). For evaluation on potential energy (Fig. 2b,c), MM exhibited a broader error distribution and a much higher upper bound of error (that is, the maximum error) than AI2BMD. The average MAE of the MM potential energy consistently hovered around 0.2 kcal mol−1 per atom, whereas AI2BMD achieved a much lower value (0.038 kcal mol −1 per atom, averaged over the five proteins) (Fig. 2b). As the protein size increased from chignolin (175 atoms) to PACSIN3 (1,040 atoms), the increase of energy errors could be attributed to insufficient modelling for the escalating many-body interactions among protein units. For proteins from SSO0941 with 2,450 atoms to aminopeptidase N with 13,728 atoms, the reference value could be determined only through fragmented DFT (Fig. 2c). For these four proteins, AI2BMD’s performance (MAE of 7.18 × 10−3 kcal mol−1 per atom) was substantially superior to that of MM (0.214 kcal mol−1 per atom). In terms of force (Fig. 2d,e), compared with the MM force field, AI2BMD aligned much more closely
Fragmentation
AI2BMD potential
Energy and atomic forces
Proteins
Simulation
Trajectories:
t + t ...  t + nt
Thermodynamics
Ab initio accuracy
Kinetics
Wet-lab experiment alignment
Protein units Datasets
Modelling Calculation
...
+
DFT
High
Low Time
Dissociation
fraction
RMSD
Low High
Tm
Fig. 1 | The overall pipeline of AI2BMD. Proteins are divided into protein units by a fragmentation process. The AI2BMD potential is designed on the basis of ViSNet, and the datasets are generated at the DFT level. It calculates the energy and atomic forces for the whole protein. The AI2BMD simulation system is built on these components and provides a generalizable solution for simulating the
MD of proteins. It achieves ab initio accuracy in energy and force calculations. Through comprehensive analysis from both kinetics and thermodynamics perspectives, AI2BMD exhibits good alignment with wet-lab experimental data and detects different phenomena than MM.


Nature | www.nature.com | 3
with DFT results. For the first five proteins directly calculated by DFT, AI2BMD had an average MAE of 1.974 kcal mol−1 Å−1 compared to MM’s 8.094 kcal mol−1 Å−1 (Fig. 2d). For the last four large proteins, AI2BMD achieved an average MAE of 1.056 kcal mol−1 Å−1, whereas MM’s value was 8.392 kcal mol−1 Å−1 across four systems (Fig. 2e). We further compared the performance of AI2BMD for different conformations. As shown in Supplementary Figs. 2–4, the MAE values of the potential energy for unfolded, intermediate and folded conformations of each kind of protein were analysed. The MAE values of the potential energies of different
conformations fluctuated among different proteins, whereas those of the atomic forces were slightly increased from unfolded conformations to folded conformations. The minimal MAE across different proteins and conformations underscores the ab initio accuracy of the AI2BMD system. Furthermore, to examine the efficiency of AI2BMD, we compared the time consumption of the energy calculation for all nine proteins by AI2BMD and DFT calculation software with graphics processing unit (GPU) support. In Fig. 2f, we present the computation time for
0 2,000 4,000 6,000 8,000 10,000 12,000 14,000 Atoms
0
50
100
150
200
250
Time consumption (days)
AI2BMD DFT
Chignolin Trp-cage WW domain
ABD PACSIN3
0
0.1
0.2
0.3
0.4
0.5
0.6
MAE of energy (kcal mol–1)
a
Chignolin 175 atoms
Trp-cage 281 atoms
WW domain 571 atoms
ABD 746 atoms
PACSIN3 1,040 atoms
SSO0941 2,450 atoms
APC 5,292 atoms
Polyphophate kinase 11,404 atoms
Aminopeptidase N 13,728 atoms
bc
de
f
200 300 400 500 600 700 Atoms
0
20
40
60
80
Time consumption (min)
SSO0941 APC Polyphosphate kinase
Aminopeptidase N
Chignolin Trp-cage WW domain
ABD PACSIN3 SSO0941 APC Polyphosphate kinase
Aminopeptidase N
0
0.1
0.2
0.3
0.4
0.5
0.6
Protein
0
2
4
6
8
10
12
MAE of force (kcal mol–1 Å–1)
MAE of energy (kcal mol–1)
MAE of force (kcal mol–1 Å–1)
MM
Proteins
0
2
4
6
8
10
12
AI2BMD
MM
AI2BMD
MM
AI2BMD
MM
AI2BMD
Fig. 2 | Evaluation of energy and force calculations by AI2BMD and MM.
a, Folded structures of nine evaluated proteins. For these proteins, the number of atoms ranges from 175 to 13,728. b–e, The MAE of potential energy (b,c) and atomic force (d,e). For each protein, we conducted replica-exchange MD and structure clustering to select representative structures, including folded, unfolded or intermediate states. AI2BMD simulations were conducted for the representative structures, and 200 samples in total were selected for evaluation. For the first 5 proteins within 1,040 atoms shown in b,d, DFT calculation for the whole protein performed by ORCA with the same settings in dataset generation is set as the reference value, whereas for the last 4 proteins shown in c,e, the
reference value is set as the fragment DFT calculation owing to prohibitive computational cost. In b,c, the potential energy of each structure has that of the initial folded structure subtracted, and then is normalized by the number of atoms. The error bars in b–e indicate the standard deviations of the potential energy and atomic force of 200 different samples of the protein (n = 200), with each sample shown as a filled circle. f, Comparison of time consumption of energy calculation for nine proteins. DFT calculations were carried out on a GPU. For the last five proteins, the time consumption by DFT was estimated by the fitting curve from those of the first four proteins and is shown with a dashed line and circles. The inset shows a comparison for the first four proteins.


4 | Nature | www.nature.com
Article
AI2BMD and DFT on a desktop with an A6000 GPU card (48-GB GPU memory) and 32 central processing unit cores. It is obvious that AI2BMD achieved ab initio accuracy much faster than DFT. The computational time for AI2BMD exhibited a near-linear increase. AI2BMD took 0.072 s to perform a simulation step for Trp-cage with 281 atoms, compared to 21 min by DFT. For the albumin-binding domain with 746 atoms, the time slightly increased to 0.125 s for AI2BMD compared to 92 min for DFT. For a larger protein, aminopeptidase N with 13,728 atoms, it was 2.610 s, and DFT calculations were not feasible with the estimated time exceeding 254 days, which would be more than 6 orders slower than AI2BMD. We further compared AI2BMD’s simulation speed with that of other AI-driven simulation systems, including DPMD17 and Allegro18, as well as the AMOEBA force field implemented in Tinker 8 and ff19SB implemented in Amber. As shown in Extended Data Table 1, AI2BMD exhibits a faster simulation speed, except for the smallest protein chignolin, than DPMD, even though DPMD uses a simpler model architecture. AI2BMD’s simulation speed substantially surpassed Allegro and AMOEBA for all cases. Furthermore, both DPMD and Allegro encountered an ‘out-of-memory’ error on an A6000 GPU card for some large proteins, whereas AI2BMD worked well. In addition, a non-polarizable force field exhibits the fastest simulation, being about one order faster than AI2BMD. In summary, AI2BMD is versatile, isgeneralizable to various proteins and offers both ab initio accuracy and highly efficient calculation for MD simulation.
Conformational space exploration
To demonstrate the capabilities of AI2BMD for conformational space exploration and protein kinetics, we carried out AI2BMD simulations for both protein dipeptides and proteins. We initially constructed an asparagine dipeptide (Ace-N-Nme) in which the amino acid is capped with acetyl and N-methylamino groups at its amino and carboxy termini, respectively, in a 5-Å water box and sampled the hydrogen bonds between the solute and the solvent by carrying out a 500-ps simulation using quantum mechanics (QM)–MM, AI2BMD with polarizable embedding and MM with Amber ff19SB. Then we scanned the distance between the oxygen in the water molecule and the acceptor on the dipeptide and calculated the energy fluctuations for the entire system by pure QM, AI2BMD and MM. As depicted in Extended Data Fig. 2a,b, the distance distributions between the oxygen in the water molecule and the hydrogen-bond acceptor on the main chain, as sampled by QM–MM and AI2BMD, exhibited high similarity. AI2BMD also demonstrated an energy distribution much more consistent with QM–MM than MM in the hydrogen-bond scanning (Extended Data Fig. 2c). Furthermore, AI2BMD showed consistent O–O distance distributions in comparison to QM–MM for the side-chain hydrogen bond with water (Extended Data Fig. 2d–f), with the peaks of both AI2BMD and QM–MM located at identical positions. In conclusion, the hydrogen-bond sampling and scanning experiments suggest that AI2BMD can accurately model the solvent effect and the interactions between the solute and the solvent. Then we comprehensively sampled the conformation space of different protein units. We first evaluated the accuracy of potential energy and atomic force calculations during the simulations produced by the AI2BMD system. AI2BMD simulations of 10 ns were carried out for each kind of dipeptide with a 10-Å water box, and 200 snapshots with solvent were evenly picked from the simulation trajectory. The energy and force were calculated by QM for the whole protein and the AMOEBA force field for the solvent part as the reference value. The MM calcu

=== Structure Language Models for Protein Conformation Generation (Lu, Jiarui; Chen, Xiaoyin; Lu, Stephen Zhewen; Shi, Chence; Guo, Hongyu; Bengio,) ===
Preprint. Under review.
STRUCTURE LANGUAGE MODELS FOR PROTEIN
CONFORMATION GENERATION
Jiarui Lu1,2,†, Xiaoyin Chen1,2,†, Stephen Z. Lu1,3, Chence Shi1,2, Hongyu Guo4,5, Yoshua Bengio1,2,6 & Jian Tang1,6,7
1Mila - Que ́bec AI Institute, 2Universite ́ de Montre ́al, 3McGill University, 4University of Ottawa, 5National Research Council Canada, 6CIFAR AI Chair, 7HEC Montre ́al
ABSTRACT
Proteins adopt multiple structural conformations to perform their diverse biological functions, and understanding these conformations is crucial for advancing drug discovery. Traditional physics-based simulation methods often struggle with sampling equilibrium conformations and are computationally expensive. Recently, deep generative models have shown promise in generating protein conformations as a more efficient alternative. However, these methods predominantly rely on the diffusion process within a 3D geometric space, which typically centers around the vicinity of metastable states and is often inefficient in terms of runtime. In this paper, we introduce Structure Language Modeling (SLM) as a novel framework for efficient protein conformation generation. Specifically, the protein structures are first encoded into a compact latent space using a discrete variational auto-encoder, followed by conditional language modeling that effectively captures sequencespecific conformation distributions. This enables a more efficient and interpretable exploration of diverse ensemble modes compared to existing methods. Based on this general framework, we instantiate SLM with various popular LM architectures as well as proposing the ESMDiff, a novel BERT-like structure language model fine-tuned from ESM3 with masked diffusion. We verify our approach in various scenarios, including the equilibrium dynamics of BPTI, conformational change pairs, and intrinsically disordered proteins. SLM provides a highly efficient solution, offering a 20-100x speedup than existing methods in generating diverse conformations, shedding light on promising avenues for future research.
1 INTRODUCTION
Protein structure dynamics are fundamental to understanding the biological functions of proteins. The ability of proteins to adopt multiple conformations is crucial for their function in influencing interactions with other biomolecules and the environment. Traditional computational methods, such as molecular dynamics (MD) simulations, have long been used to explore these dynamics. However, these methods are computationally expensive and time-consuming. Structure prediction models, such as AlphaFold 2 (Jumper et al., 2021) and RosettaFold (Baek et al., 2021), have made significant strides in predicting static protein structures, yet often fail to accurately capture the dynamic nature of proteins and their multiple conformations (Chakravarty & Porter, 2022).
Recently, significant progress has been made by adopting deep generative models as conformation samplers to efficiently explore the complicated protein conformational space. For example, Noe ́ et al. (2019) adopts normalizing flow to match the underlying Boltzmann distribution by learning from simulation data. Despite their potential, normalizing flow-based methods (Noe ́ et al., 2019; Klein et al., 2023) face significant challenges in modeling large protein systems with hundreds of amino acids, as the invertibility constraint becomes a major obstacle when scaling up model parameters. As a remedy, denoising diffusion approaches (Jing et al., 2023; Lu et al., 2024; Wang et al., 2024; Zheng et al., 2024) can efficiently learn from structural data, achieve good generalization, and
†Equal contribution. Code available at https://github.com/lujiarui/esmdiff. Correspondence to: jiarui.lu@mila.quebec
1
arXiv:2410.18403v1 [q-bio.BM] 24 Oct 2024


Preprint. Under review.
perform amortized inference. However, modeling high-dimensional protein structures explicitly in their 3D Euclidean space can demand intensive computation and usually requires accounting for special equivariant properties (Ko ̈hler et al., 2020). Furthermore, L2-based training objectives such as denoising score matching (Song et al., 2020) tend to predict local perturbations rather than capturing remote modes of alternative conformations (Wang et al., 2024). Consequently, these models may overallocate their capacity to learn structural noises in the training data instead of focusing on low-frequency structural changes (Chou, 1985).
In complement with existing approaches, we present Structure Language Modeling (SLM), a novel framework for protein conformation generation that performs generative modeling in the latent space of protein structures. Inspired by the recent progress in developing structural vocabularies for protein representation learning (Su et al., 2023; Hayes et al., 2024), our approach first encodes structural flexibility into a distribution over latent tokens using a discrete variational autoencoder, as illustrated in Fig. 1. The discrete latent encoding removes high-frequency details of protein structures, forming “structure languages” that effectively capture the uncertainty of complex protein conformations (Fig. 2a); Conditional language modeling is then applied to these latent structure tokens, using amino acid types as context to capture sequence-specific conformation distributions (Fig. 2b); Protein conformations can finally be reconstructed by mapping structure tokens into 3D space with a learned decoder (Fig. 2c). By leveraging generative language modeling in the discrete latent space, SLM bypasses the complexity of equivariant constraints associated with geometric symmetries and benefits from enhanced model capacity. As a general framework, SLM is fully compatible with any existing language model (LM) architectures and shows promising scalability. To further demonstrate the versatility of our approach, we introduce ESMDiff, a novel BERT-like structure language model instantiation fine-tuned from ESM3 (Hayes et al., 2024) with masked discrete diffusion (Austin et al., 2021; Zhao et al., 2024) grounded in the SLM framework. Experimental results across various conformation generation scenarios demonstrate the state-of-the-art performance of SLM including the representative ESMDiff model, achieving orders of magnitude faster speeds compared to existing generative methods. The proposed framework paves the way for new research avenues in addressing the protein conformation sampling challenge.
We summarize our key contributions as follows.
• We comprehensively explore an innovative conformation generation framework based on language modeling in the latent space, which opens up potential research avenues.
• We introduce ESMDiff, a novel fine-tuned variant of a state-of-the-art protein language model, built on masked discrete diffusion.
• We demonstrate the superior capability of structure language models by evaluating them on various conformation generation settings and comparing them with existing methods.
2 RELATED WORK
Flexibility
Flexibility
GLU49
GLY12
Figure 1: Residue flexibility (BPTI clusters, Shaw et al. (2010)) reflected by the categorical distribution over latent structure tokens. Different tokens (colored in different shades ) are used to encode varying local structural patterns.
Protein language models. In recent years, several language models of protein sequence have been built. Among these, the ESM-series (Rives et al., 2021; Lin et al., 2023; Hayes et al., 2024) and other similar models (Elnaggar et al., 2021; Alley et al., 2019) have garnered great attention because of their wide range of downstream applications such as protein engineering (Meier et al., 2021). On the other hand, auto-regressive protein language models, based on either recurrent neural networks (Alley et al., 2019), or Transformer including ProGen (Madani et al., 2023) and ProtGPT2 (Ferruz et al., 2022), are able to generate de novo sequences with input controlling tokens. Specially, inverse folding models (Ingraham et al., 2019; Jing et al., 2020; Hsu et al., 2022; Dauparas et al., 2022; Gao et al., 2022) learn to perform structure-based protein design with geometric-aware encoders.
Generative conformation sampling. Given the intensive computation of traditional MD simulations, generative models have been used to learn conformation distributions in a data-driven fashion. The Boltzmann generator (Noe ́ et al., 2019) uses normalizing flow to fit the Boltzmann dis
2


Preprint. Under review.
Encoder Decoder
Structure Language Model
AA-type tokens
Structure tokens
...
...
...
Structure Language Model
Structure Language Model
Input
Structure tokens
Conformation ensemble (size = )
...
Encoder
Decoder
Neighborhoodaware invariant encoding
Residue embeddings
AA-type tokens
...
Codebook
Structure tokens
Sample times
...
dVAE for Structure
3D structure coords
(a) Encoding (b) Training (c) Inference
Figure 2: An illustration of the proposed SLM framework.
tribution from target-specific simulation data. Arts et al. (2023) extends this by using denoising diffusion models for coarse-grained protein conformations. Furthermore, EigenFold (Jing et al., 2023), Str2Str (Lu et al., 2024), AlphaFlow (Jing et al., 2024), ConfDiff (Wang et al., 2024), and DiG (Zheng et al., 2024) leverage diffusion or flow matching to conditionally sample protein conformations by learning from PDB data. Recently, AlphaFold3 (Abramson et al., 2024) revised the structure decoder of AlphaFold2 to a diffusion-based module for diversified structure prediction.
Quantized representation for protein structures. Beside the prevailing diffusion models for protein structure, representation learning of protein structures using discrete variational autoencoders (dVAE) has gained increasing attention in recent years. FoldSeek (van Kempen et al., 2022) is one of the earliest attempt to build dVAE for fast structure search and alignment. Based on this, SaProt (Su et al., 2023) constructs learned representations with both sequence and structure tokens as input, while ProtT5 (Heinzinger et al., 2023) fine-tuned an existing language model to accept structure tokens as input. PVQD (Haiyan et al., 2023) applied latent diffusion in the embedding space of dVAE for conditional protein structure generation. ProSST (Li et al., 2024) trained an autoencoder with Kmeans clustering applied in the latent space. Gaujac et al. (2024) and Gao et al. (2024) respectively build dVAE with large vocabularies for learning protein structure representations.
Remarks: Our work is closely related to these concurrent research directions by leveraging LMs to model and efficiently perform conformation generation over the quantized representation of protein structures. We refer to this framework as “structure language models” and describe them in detail.
3 PROTEIN CONFORMATION GENERATION WITH LANGUAGE MODELING
Notation. A protein with L residues is identified by its sequence of amino acid types c ∈ |S|L where S is the vocabulary of 20 standard amino acids. The protein (backbone) structure is represented by its composing 3D atom positions x ∈ X ≡ RL×4×3 including all backbone heavy atoms. Through an encoder q(z|x), the structure x is encoded to a sequence of latent codes z ∼ q(z|x) where z ≡ (z1, . . . , zL) ∈ |V |L and V is the pre-specified vocabulary of latent codes; the structure tokens z are decoded by first mapping to embedding vectors and then to the 3D structure x.
3.1 LEARNING THE SEQUENCE-STRUCTURE DISTRIBUTION
To address the conformation generation problem, we start with modeling the sequence-to-structure translation distribution p(x|c) of interest and derive the learning objective in this section. To circumvent explicitly learning in the structure space, the roto-translation invariant* latent representation z is introduced to encode 3D atomic protein structure. Given this, the target distribution
*For example, features like distance and angle are roto-translation invariant. This relationship can be formally written as q(z|T ◦ x) ≜ q(z|R ◦ x + t) = q(z|x), ∀T .
3


Preprint. Under review.
p(x|c) can be derived by marginalizing the joint distribution p(x|c) = R
z p(x, z|c). We fur
ther factorize this joint distribution according to the Bayes’ rule by isolating the latent variable z: pθ,φ(x, z|c) = pφ(x|c, z)pθ(z|c), where pφ denotes the (decoding) distribution over the 3D protein structures given the structure token and sequence, and pθ denotes the conditional distribution over the structure tokens, respectively modeled by neural networks with parameter set φ, θ. This gives rise to the evidence lower bound on the likelihood of model distribution over protein structures conditioned on sequence:
log pθ,ψ(x|c) ≥ Eqψ(z|x) [log pφ(x|c, z)] − DKL(qψ(z|x)∥pθ(z|c)) ≜ L(φ, θ), (1)
where ψ is introduced to parameterized the posterior distribution over latent representation z. Please refer to the Appendix G.1 for the full derivation of Eq. 1. Directly optimizing the right-hand side of Eq. 1 can be intractable and difficult since we have unknown posterior qψ. As a result, we adopt an one-step expectation–maximization (EM) approach (Dempster et al., 1977) by first jointly learning pφ and qψ with a simple and parameter-free prior distribution p(z|c), followed by optimization on
pθ with the learned p∗
φ and q∗
ψ. This yields the overall two-stage and separable training pipeline:
Learning quantized representation for structure. With the prior p(z|c) fixed, we begin by maximizing the ELBO L(φ, θ) with respect to the encoder ψ and decoder φ, using protein structure samples D = c, x. In the context of discrete latent spaces, this process is analogous to training a discrete VAE (dVAE) (Van Den Oord et al., 2017) to learn quantized representations for protein structures. Here, the encoder qψ(z|x) maps structures to latent tokens, while the decoder pφ(x|z, c) reconstructs structures from these tokens†. The prior p(z|c) is fixed to be uniform during this stage.
Learning the prior over latent tokens. In this stage, we fix the learned parameters φ∗ and ψ∗, and train the prior pθ by maximizing the ELBO: arg maxθ L(φ∗, θ). Since both φ∗ and ψ∗ are fixed, the reconstruction term in the ELBO cancels out, and training reduces to minimizing the KL divergence DKL(qψ∗ ∥pθ). This is equivalent to performing maximum likelihood estimation, as E(c,x)∼DEz∼qψ(z|x)pθ(z|c) with respect to pθ. Given that both z and c are categorical variables, this formulation resembles a translation task, allowing pθ to be parameterized by language models.
3.2 STRUCTURE LANGUAGE MODELING
The prior learned in the previous stage is now applied to conformation generation, which can be framed as a conditional generative modeling problem for sequence-to-structure (seq2str) translation. Given an input condition c ∈ |S|L which determines the molecular topology, the goal is to sample a conformation ensemble from p(x|c). To do this, we first sample a set of latent variables from the prior distribution learned earlier, z ∼ pθ(z|c), and then decode these latents using the decoder pφ(x|c, z). The decoder is jointly trained with the encoder qψ(z|x) in the first stage, ensuring that the sampled latents align with the reconstruction. This framework supports roto-translation invariant inference and is described in Algorithm 1. Next, we illustrate this approach with two straightforward examples of structure language models (SLM): the encoder-decoder and decoder-only architectures.
Encoder-decoder. Given the conditional nature of translation, the prior pθ(z|c) can be explicitly modeled by an encoder-decoder architecture like T5 (Raffel et al., 2020). The decoder conditions on the context c and factorizes the structure tokens sequentially: p(z|c) = QL
l=1 p(zl|z<l, c), where
z ∈ Z represents the quantized structure tokens. The training objective is the negative log-likelihood (NLL) loss conditioned on c: L(θ) = −E(c,x)∼DEz∼q(z|x)
PL
l=1 log pθ(zl|z<l, c), z<1 ≡ ∅.
Decoder-only. Alternatively, the latent prior pθ(z|c) ∝ pθ(c, z) can be modeled autoregressively using a decoder-only architecture, such as GPT (Radford et al., 2019), where c serves as the “prompt”. We define y ≜ [c, z] = [c1, . . . , cL, z1, . . . , zL], and the training involves maximizing the likelihood over y via the NLL minimization: L(θ) =
−E(c,x)∼D Ez∼q(z|x)
P2L
l=1 log pθ(yl|y<l), where c, x ∼ D is the i.i.d. samples from the data dis
tribution over structures each with the associated amino-acid sequence as condition c. In practice, we add an additional special token [sep] to differentiate between these two modalities.
Inference involves sampling with a left-to-right decoding order, as defined by the autoregressive factorization of both language models. Figure 3 briefly illustrates these two modeling strategies.
†We assume that x is conditionally independent of c given latent variable z.
4


Preprint. Under review.
(a) Encoder-decoder
Encoder
Structure tokens
...
...
...
...
AA-type tokens
cross attention
Decoder
?
Structure tokens
...
...
...
...
AA-type tokens <sep>
(b) Decoder-only
Decoder
?
Figure 3: Autoregressive prior modeling the for latent structure tokens discussed in Section 3.2.
4 ESMDIFF: A MASKED DIFFUSION INSTANTIATION
Building on the foundation of SLM, we here propose ESMDiff as an instantiation based on discrete diffusion models (Austin et al., 2021). ESMDiff incorporates the inductive bias of seq2str translation and leverages the protein foundation model ESM3 (Hayes et al., 2024) through masked diffusion fine-tuning. The effectively fine-tuning of ESMDiff also exemplifies how a large pretrained BERTlike masked language model can be adapted to acquire additional generative capabilities, making it well-suited for broader downstream tasks such as conformation generation.
4.1 REVISITING DISCRETE DIFFUSION AS DISTRIBUTION INTERPOLATION
The discrete diffusion models (Austin et al., 2021; Lou et al., 2023; Sun et al., 2022; Campbell et al., 2022; Zheng et al., 2023) can be gene

=== Protein Conformation Generation via Force-Guided SE(3) Diffusion Models (Gu, Quanquan; Wang, Yan; Wang, Lihao; Shen, Yuning; Wang, Yiqun; Yuan, Huizhuo; ) ===
Protein Conformation Generation via Force-Guided SE(3) Diffusion Models

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
To address the aforementioned challenges, we propose a novel force-guided diffusion model, CONFDIFF, aiming to generate high fidelity protein conformations that better adhere to the Boltzmann distribution. Drawing inspiration from the contrastive energy prediction (CEP) technique (Lu et al., 2023), we employ the MD energy prior as a physicsbased preference function. By introducing an additional force guidance network during the diffusion sampling process, it prioritizes generating conformations with lower potential energy, which effectively enhances sampling quality. Our model is trained on general protein structures from the Protein Data Bank (PDB) (Berman et al., 2000) as well as self-generated conformation samples, without relying on MD simulation data (Zheng et al., 2023). To sum up, the main contributions of this work are highlighted as follows:
• Employing a sequence-conditional model to guide the unconditional model, we use classifier-free guidance on SE(3) to find a better trade-off between conformation quality and diversity. Compared with DiG (Zheng et al., 2023), our method does not rely on MD data during training; compared with Str2Str (Lu et al., 2024), the guidance intensity coefficient provides a higher degree of freedom for balancing sample diversity and quality.
• We utilize the MD energy function as a physics-based reward to guide the generation of protein conformations. In addition, we propose an intermediate force guidance strategy during the diffusion sampling process. To the best of our knowledge, this is the first force-guided network suitable for protein conformation generation, contributing to the alignment of diverse conformation predictions with the equilibrium distribution.
• Experiments on a variety of benchmarks demonstrate that our method outperforms the state-of-the-art approaches. In particular, energy and force guidance effectively guide the model to sample conformations with lower energy, leading to diverse samples more truthful to the underlying Boltzmann distribution.
2 Related Work
Protein Conformation Prediction. Perturbing pretrained folding models, such as AlphaFold (Jumper et al., 2021), to obtain a diverse set of alternative conformations marks the first attempt to use deep neural networks for multiconformation predictions. Stein & Mchaourab (2022) introduced mutations to the MSA representations to obtain different folded structures from AlphaFold. Similarly, WaymentSteele et al. (2023) clusters MSA by sequence similarity

to enable AlphaFold to discover alternative folding states of known metamorphic proteins. Reducing the MSA depth can also unlock multi-conformation prediction capability of AlphaFold (Del Alamo et al., 2022). In addition, Vani et al. (2022) proposed to use the outputs from AlphaFold as initialization for AI-augmented MD simulations. Noe´ et al. (2019), Janson et al. (2023) and Mansoor et al. (2023) utilized MD simulation data to generate protein conformation ensembles.
Recently, diffusion models have been employed for protein conformation generation. Zheng et al. (2023) proposed Distributional Graphormer (DiG), which is trained on both protein structures from the PDB and MD simulation data. Unlike our proposed method, DiG incorporates an additional regularization into its loss function to align the learned score with MD force field at small diffusion time, which then extends over the entire pathway by the Fokker-Planck equation. Jing et al. (2023) introduced EIGENFOLD, a harmonic diffusion model with physics-inspired prior to sample protein conformations. The cascading-resolution generative process allows efficient sampling across proteins of varying length. The model achieves remarkable performance in a number of benchmark tasks, yet the advantage of using a harmonic prior over the conventional isotropic Gaussian prior is not evidently clear. Inspired by simulated annealing, Lu et al. (2024) proposed Str2Str, a heating-annealing generative framework using an unconditional score model. By adjusting the duration Tδ of heating (i.e., forward diffusion) process, a certain degree of sample diversity could be achieved. However, during the model’s training/inference phase, the absence of sequence or energy information evokes a question of whether the diverse generated outcomes adhere to the Boltzmann distribution.
Diffusion Models for Protein Design. Another line of work focuses on developing diffusion models for protein design (Trippe et al., 2022; Anand & Achim, 2022; Wu et al., 2022a). In particular, Watson et al. (2022) repurposed RoseTTAFold (Baek et al., 2021) to generate novel protein-binder backbones with successful experimental validation. Ingraham et al. (2023) proposed Chroma, which introduces a diffusion process that respects the conformational statistics of polymer ensembles, and can be effectively conditioned on protein semantics or even natural language to generate structures with desired properties. Yim et al. (2023b) proposed framediff, an innovative diffusion process on SE(3) for equivariant protein backbone generation, and has been recently extended to the flow-matching paradigm (Yim et al., 2023a; Bose et al., 2023). Diffusion models have also been applied to antibody sequence-structure co-design (Martinkus et al., 2023; Luo et al., 2022). The effectiveness of these approaches underscores the potential for further advancements of diffusion modeling in protein studies.
Controllable Generation via Guided Diffusion. Control-

2

Protein Conformation Generation via Force-Guided SE(3) Diffusion Models

lable generative modeling is key to aligning diffusion models with human preference in many real-world tasks. Both classifier guidance (Dhariwal & Nichol, 2021; Song et al., 2020) and classifier-free guidance (Ho & Salimans, 2022) have been proposed to guide an unconditional model with a preferred conditional variable, showing remarkable performance in a wide range of applications including text-toimage generation (Saharia et al., 2022; Nichol et al., 2021), video generation (Ho et al., 2022), etc. Recently, (Lu et al., 2023) proposed a novel energy guidance policy using a scalar reward function rather than fixed conditioning variables. By optimizing a contrastive energy prediction (CEP) objective, the model is guaranteed to converge to the exact guidance under enough model capacity and data samples. CEP shows great performance in image synthesis and reinforcement learning tasks. However, it is the gradient of energy function that is utilized as guidance in reverse sampling. This inspires us to propose a force guidance strategy which employs an equivariant network to directly approximate the intermediate force vector.
3 Preliminaries
3.1 Protein Backbone Diffusion on SE(3)
We adopt the protein backbone representation from AlphaFold (Jumper et al., 2021): for a protein with N amino acid residues, its backbone atomic coordinates can be parameterized by a collection of N orientation preserving rigid transformations (i.e., frames) to the local [N,Cα,C,O] backbone atoms in each residue. We collectively denote the positions of all N frames by x0 = [T0, R0] ∈ SE(3)N , where T0 ∈ R3N and R0 ∈ SO(3)N denote the corresponding translation and rotation operations, respectively. With an additional backbone torsion angle ψ describing the rotation of the oxygen atom around the C–Cα bond within each residue, we can reconstruct the protein backbone structure from the frame representations.
Following Yim et al. (2023b), diffusion modeling on manifold SE(3)N is employed for protein backbone generation. Two independent diffusion processes are defined for the translation and rotation subspaces, respectively:

1 dTt = − 2 βtPTt dt + βtPdwt,

dRt =

d dt

σt2

dwtSO(3),

(1)

where subscript t denotes the diffusion time variable in

[0, 1], βt and σt are predefined time-dependent noise sched-

ules, P is a projection operator removing the center of

mass, and [wt, wtSO(3)] represents the standard Wiener pro-

cess in [N (0, I3)⊗N , U (SO(3))⊗N ]. T√he transition kernel of T satisfies pt(Tt|T0) = N (Tt; αtT0, (1 − αt)I),

where αt

=

e−

. t
0

βs ds

The rotational transition kernel

satisfies pt(Rt|R0) = IGSO3(Rt; R0, t), where IGSO3 is

the isotropic Gaussian distribution on SO(3) (Leach et al., 2022).
The associated reverse-time stochastic differential equation (SDE) is as follows:

1 dTt = P − 2 βtTt − βt∇ log pt(Tt) dt + βtPdw¯ t,

dRt

=

−

d dt

σt2

∇

log

pt(Rt)dt

+

d dt

σt2dw¯ tSO(3),

(2)

where [w¯ t, w¯ tSO(3)] denotes another standard Wiener process in reverse time.

3.2 Classifier-free Guidance for Diffusion Sampling

Guided sampling has emerged as a critical strategy in devel-

oping diffusion models capable of generating samples com-

plying with human instructions. Consider existing paired

data x0 ∼ p0(x0|c) with a conditioning variable c (subscript

denotes diffusion time, where t = 0 corresponds to the orig-

inal data), we typically perceive the conditional probability

density at time t as pt(xt|c). Applying Bayes’ rule we can

obtain

pt(xt|c)

=

pt

(xt )pt (c|xt p(c)

)

.

Therefore,

one

can

train

a classifier to predict the conditioning probability pt(c|xt)

with given noisy data xt, and use the score of the classifier

output as guidance (Dhariwal & Nichol, 2021). Instead of

training a separate classifier to estimate ∇ log pt(c|xt), Ho

& Salimans (2022) proposed to utilize an implicit classi-

fier pγt (c|xt) ∝

pt (xt |c)p(c) pt (xt )

γ, which then leads to a linear

combination of an unconditional score estimator sθ(xt) and

a conditional score estimator sθ(xt, c) to jointly estimate

the target score function:

∇xt log pt(xt|c) =∇xt log pt(xt) + γ∇xt log pt(c|xt) =γ∇xt log pt(xt, c) + (1 − γ)∇xt log pt(xt) ≈γsθ(xt, c) + (1 − γ)sθ(xt),

where γ is a hyperparameter controlling the guidance strength. When γ = 0, it reduces to an unconditional model, while at γ = 1, it becomes a pure conditional model. These two models can be simultaneously trained under the same hood, where the unconditional model receives masked conditioning variable c during training.
4 Force-Guided Diffusion for Protein Conformation Generation
In this section, we propose force-guided CONFDIFF, a diffusion model targeting multi-conformation generation for proteins. Employing a sequence-based conditional score network to guide an unconditional score model in Section 4.1, CONFDIFF achieves reasonable conformation diversity while ensuring sample quality. Building upon the energy guidance foundations in Section 4.2, a novel force-guided sampling strategy is proposed to estimate the intermediate

3

Protein Conformation Generation via Force-Guided SE(3) Diffusion Models

CONFDIFF

Classifier-free Guidance

Energy Improved

∇xt log pt(xt|seq) is estimate by sθ(xt, t|seq) = γscθ(xt, t, seq) + (1 − γ)suθ (xt, t).

Prior

+ Intermediate Force Guidance

Figure 1. Protein conformation generation with multiple guidance strategies. Upper: With a mixture of sequence-conditional and unconditional score models, CONFDIFF in Section 4.1 samples diverse conformations with reasonable quality. Lower: Incorporating force guidance in Section 4.3, the model generates structures with lower energy, better comply with the Boltzmann distribution.
force function, which is then embedded within reverse time sampling process in Section 4.3. Utilizing prior information from the MD force field, our model successfully reweights the generated conformations to ensure they adhere better to the equilibrium distribution. A visual depiction is shown in Figure 1.
4.1 Sequence-Conditional Diffusion on SE(3)
Our baseline model consists of an unconditional score model suθ (xt, t) and a sequence-conditional one scθ(xt, t, seq). The unconditional model is trained on protein structures (from the PDB) without any sequence information, effectively capturing the conformation distribution of general proteins. On the other hand, the sequence-conditional model has access to both protein sequence information (seq) and the corresponding structure xt at time t. We adopt a similar network architecture to FramePred (Yim et al., 2023b) to parameterize the corresponding score functions. The unconditional model takes sinusoidal embedding of the residue index and diffusion time t as its single ({si}) and pair ({zij}) embeddings, where the conditional model additionally concatenates precomputed representations from ESMFold (Lin et al., 2022) to its single embedding. Note that the choice of sequence representation for the conditional model is flexible – it has been shown that using pretrained representations from folding models helps diffusion models generate reasonable protein structures (Jing et al., 2023; Zheng et al., 2023), while the unconditional model can ef

=== ESMAdam: a plug-and-play all-purpose protein ensemble generator (Yu, Zongxin; Liu, Yikai; Lin, Guang; Jiang, Wen; Chen, Ming) ===
1 ESMAdam: a plug-and-play all-purpose protein ensemble
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
62 of complete conformational landscapes.
63 Advancements in AI-driven methods, including AlphaFold,18,19 RoseTTAFold,20 trRosetta,21 and
64 ESMFold,22 have significantly improved the accuracy and efficiency of protein structure pre
65 diction. More recently, the focus has shifted from single-structure predictions to generating
66 protein conformational ensembles and capturing the dynamic range of protein states. Early ap
67 proaches, such as MSA subsampling23 and clustering,24 expanded AlphaFold’s output by mod
68 ifying model inputs to produce a more diverse set of conformations. Advanced methods have
69 used deep generative models, particularly diffusion models,25,26 to address the challenge of en
70 semble generation.27–30 These models utilize stochastic perturbation processes to effectively
71 explore the conformational landscape, enhancing both accuracy and diversity, while predict
72 ing the conformational dynamics that govern protein behavior. Deep generative models have
73 also been broadly applied to other conformation generation tasks, including protein structure
74 inpainting,31–35 protein-ligand docking prediction,36–38 and protein-protein interaction mod
75 eling.39 However, these models are typically designed for one or a few specific conformation
76 generation tasks. Recent advances have demonstrated that diffusion models can be adapted
77 for general-purpose ensemble conformation generation tasks with arbitrarily defined ensemble
78 constraints.40 Despite this versatility, the performance of such models declines significantly as
79 the nonlinearity and dimensionality of the ensemble constraints increase, limiting their ability
to generalize across complex tasks.41
80
81 In this study, we introduce ESMAdam, a simple yet efficient method for general-purpose pro
82 tein conformation ensemble generation. Building upon the pretrained protein language model
83 ESMFold,22 ESMAdam utilizes Adam stochastic optimization42 on the high-dimensional em
84 bedding space of protein sequences. This approach is based on the assumption that pro
85 tein conformation ensembles are latently embedded near the native structures within the high
86 dimensional embedding space. Unlike other single-purpose protein conformation generation
87 models, ESMAdam is highly flexible and can accommodate a wide range of tasks. We demon
88 strate its efficacy through extensive protein conformation generation experiments, focusing on
89 four key applications: (1) controllable conditional conformation ensemble generation for a vari
3
perpetuity. It is made available under aCC-BY 4.0 International license.
preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in
bioRxiv preprint doi: https://doi.org/10.1101/2025.01.19.633818; this version posted January 21, 2025. The copyright holder for this


90 ety of stable, fast-folding, and intrinsically disordered proteins, (2) CG-to-all-atom configuration
91 backmapping, (3) screening alternative binding modes of flexible protein-protein interactions,
92 and (4) reconstructing protein 3D structures from cryo-EM images. Numerical evaluations reveal
93 that ESMAdam consistently achieves high performance. We propose that ESMAdam serves as
94 a versatile computational tool for rapidly generating protein ensembles, with broad applications
95 in structural biology and drug discovery.
96 Results
97 Overview of the ESMAdam model Fig. 1 summarizes a high-level methodology framework of
98 ESMAdam. The core philosophy of ESMAdam is that protein conformation ensembles can be
99 encoded in the latent space near the embedding of the native structure. if a protein conforma
100 tional ensemble is characterized by low-dimensional features such as experimental ensemble
101 measurement, stable protein structures constrained by low-dimensional features are encoded
102 in the latent space. By exploring the latent space variable with constraints from low-dimensional
103 features, it is possible to generate reasonable protein conformation ensembles. With the re
104 cently developed protein language model ESMFold, which excels in predicting native protein
105 structures, ESMAdam leverages the latent space as a trainable variable. For any given ensem
106 ble constraint, ESMAdam iteratively updates this trainable latent space variable using stochastic
107 gradient descent methods, such as Adam, ensuring the generated protein conformations align
108 with the desired constraints. Despite its simplicity, this method is highly adaptable to a wide
109 range of tasks.
110 Conditional protein conformation generation We demonstrate the effectiveness of ESMAdam
111 on conditional protein conformation ensemble generation on a comprehensive benchmark dataset,
112 which consists of a diverse set of proteins, including ordered (BPTI, gb3, Ubq), fast-folding (BBA,
113 BBL, Homeodomain, ProteinB, TrpCage and WWDomain), and intrinsically disordered proteins
114 (PaaA2, drkN, RS-peptide). This dataset spans a wide range of structural characteristics, includ
115 ing varying degrees of order, secondary structure compositions, and sequence lengths. Protein
116 ensembles representing the ground truth Boltzmann distribution were obtained from long, un
117 biased molecular dynamics (MD) simulations. Previous studies40 have shown that diffusion
118 model-based protein ensemble generation models can generate reasonable protein conforma
119 tion ensembles when guided by both global and local feature distributions. Following this frame
4
perpetuity. It is made available under aCC-BY 4.0 International license.
preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in
bioRxiv preprint doi: https://doi.org/10.1101/2025.01.19.633818; this version posted January 21, 2025. The copyright holder for this


Y Y D P E
G T W Y
Protein sequence
T
Pretrained ESM embedding
5 1 2 3 4 6 4 3 0 2
1 7 2 9 9 6 1 2 1 9
3 4 8 1 4 6 5 3 0 3
3 1 2 7 4 1 2 3 0 1
3 2 2 6 8 5 0 1 2 2
3 4 6 2 1 6 4 2 0 4
Updated embedding
Ensemble constraint
• Experimental measurement • Coarse-grained representa�on • Cryo-EM density image • Mul�mer rela�ve posi�on ...
Adam Op�miza�on
MSE Loss
ESMFold
Figure 1: The high level framework of ESMAdam. In ESMAdam, the protein sequence of interest is first embedded using the pretrained protein language model ESMFold. This embedding represents the value in the latent space corresponding to the native structure of the sequence. ESMAdam treats the embedding space as a trainable parameter. The embedding is passed through the trunk of ESMFold to generate the corresponding 3D protein structure. To generate conformation ensembles under specific constraints, the embedding parameter is optimized using a Mean Square Error (MSE) loss function. Optimization is performed with a stochastic gradient descent method, such as Adam. The embedding parameter is updated iteratively until the loss converges below a defined threshold, resulting in a physically plausible ensemble of protein conformations that align with the desired constraints.
120 work, we generated protein conformation ensembles guided by two key features: radius of gyra
121 tion and secondary structure. These features are experimentally accessible through techniques
122 such as small-angle X-ray scattering (SAXS),43–47 nuclear magnetic resonance (NMR),48–50 and
123 circular dichroism spectroscopy51–55 experiments. For fair comparison, in this experiment,
124 the distributions of these features are directly obtained from the reference MD simulation.
125 To evaluate the generated ensembles quantitatively, we compare the equilibrium free energy
126 surfaces between ESMAdam-generated conformation ensembles and ground-truth MD simula
127 tions. These free energy surfaces were generated by projecting protein configurations to UMAP
128 collective variables derived from ground truth MD simulations. The result, as shown in Fig. 2,
129 highlights the ability of ESMAdam to accurately capture the conformational diversity and ther
130 modynamic properties of the protein ensembles.
131 CG-to-all-atom configuration backmapping Coarse-grained (CG) models are important for
5
perpetuity. It is made available under aCC-BY 4.0 International license.
preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in
bioRxiv preprint doi: https://doi.org/10.1101/2025.01.19.633818; this version posted January 21, 2025. The copyright holder for this


−4 −2 0 2 4
UMAP-0
−6
−4
−2
0
2
4
6
8
UMAP-1
True (MD)
−4 −2 0 2 4
UMAP-0
−4
−2
0
2
4
6
8
ESMAdam
0
1
2
3
4
5
6
7
8
9
Free energy (kJ/mol)
bba
−4 −2 0 2 4 6 8 10
UMAP-0
−6
−4
−2
0
2
4
UMAP-1
True (MD)
−5.0 −2.5 0.0 2.5 5.0 7.5
UMAP-0
−4
−2
0
2
4
ESMAdam
0
1
2
3
4
5
6
7
8
9
Free energy (kJ/mol)
bbl
−8 −6 −4 −2 0 2 4
UMAP-0
−10
−8
−6
−4
−2
0
UMAP-1
True (MD)
−7.5 −5.0 −2.5 0.0 2.5
UMAP-0
−10
−8
−6
−4
−2
0
ESMAdam
0
1
2
3
4
5
6
7
8
9
Free energy (kJ/mol)
BPTI
−4 −2 0 2 4 6
UMAP-0
−6
−4
−2
0
2
4
6
UMAP-1
True (MD)
−2 0 2 4 6
UMAP-0
−4
−2
0
2
4
6
ESMAdam
0
1
2
3
4
5
6
7
8
9
Free energy (kJ/mol)
drkN
−8 −6 −4 −2 0
UMAP-0
−8
−6
−4
−2
0
2
UMAP-1
True (MD)
−8 −6 −4 −2 0
UMAP-0
−6
−4
−2
0
2
ESMAdam
0
1
2
3
4
5
6
7
8
9
Free energy (kJ/mol)
gb3
−10.0 −7.5 −5.0 −2.5 0.0 2.5 5.0
UMAP-0
−5
0
5
10
UMAP-1
True (MD)
−5 0 5
UMAP-0
−7.5
−5.0
−2.5
0.0
2.5
5.0
7.5
10.0
12.5
ESMAdam
0
1
2
3
4
5
6
7
8
9
Free energy (kJ/mol)
homeodomain
−5.0 −2.5 0.0 2.5 5.0 7.5 10.0 12.5
UMAP-0
−5
−4
−3
−2
−1
0
1
2
3
UMAP-1
True (MD)
−5 0 5 10
UMAP-0
−4
−3
−2
−1
0
1
2
3
ESMAdam
0
1
2
3
4
5
6
7
8
9
Free energy (kJ/mol)
PaaA2
−5 0 5 10
UMAP-0
−6
−4
−2
0
2
4
6
8
UMAP-1
True (MD)
−5 0 5 10
UMAP-0
−4
−2
0
2
4
6
ESMAdam
0
1
2
3
4
5
6
7
8
9
Free energy (kJ/mol)
proteinb
−3 −2 −1 0 1 2
UMAP-0
−3
−2
−1
0
1
2
3
UMAP-1
True (MD)
−3 −2 −1 0 1 2
UMAP-0
−2
−1
0
1
2
3
ESMAdam
0
1
2
3
4
5
6
7
8
9
Free energy (kJ/mol)
rspeptide
−4 −2 0 2 4
UMAP-0
−5
−4
−3
−2
−1
0
1
2
3
UMAP-1
True (MD)
−4 −2 0 2 4
UMAP-0
−4
−3
−2
−1
0
1
2
3
ESMAdam
0
1
2
3
4
5
6
7
8
9
Free energy (kJ/mol)
Ubq
−8 −6 −4 −2 0 2 4
UMAP-0
−6
−4
−2
0
2
4
UMAP-1
True (MD)
−7.5 −5.0 −2.5 0.0 2.5
UMAP-0
−6
−4
−2
0
2
4 ESMAdam
0
1
2
3
4
5
6
7
8
9
Free energy (kJ/mol)
WWdomain
−7.5 −5.0 −2.5 0.0 2.5 5.0
UMAP-0
−2
0
2
4
6
8
UMAP-1
True (MD)
−5 0 5
UMAP-0
−2
0
2
4
6
8
ESMAdam
0
1
2
3
4
5
6
7
8
9
Free energy (kJ/mol)
trpcage
Figure 2: Comparison between the reference MD simulation (left) and ESMAdam (right) guided by radius of gyration distribution and secondary structure free energy surface across the two dimensional UMAP for each protein. The UMAP mapping function was parameterized with the backbone torsion angles of the conformation ensembles from the reference MD simulation. The red triangle represents the native structure predicted by the ESMFold.
132 studying protein structures, thermodynamic properties, and conformation dynamics. However,
133 the coarse-graining process inherently results in the loss of detailed atomic information, mak
134 ing protein backmapping, which reconstructs an all-atom ensemble from a CG configuration,
135 a critical step in downstream applications. Recent advances in data-driven backmapping, par
136 ticularly with generative models, have facilitated efficient methods that bypass computationally
expensive physical simulations. Despite these advancements, the diversity of CG models,56–62
137
138 ranging from high-resolution multi-bead-per-residue representations58 to low-resolution ultra
139 coarse-grained (ultraCG) models63 where a single bead represents multiple residues, presents a
140 significant challenge for backmapping. No universal backmapping approach currently combines
141 high accuracy with adaptability across different CG resolutions. In this work, we demonstrate
142 that ESMAdam addresses this gap by adapting to CG models of varying scales without retrain
143 ing. We conducted two CG-to-all-atom

=== Predicting equilibrium distributions for molecular systems with deep learning (Zhang, He; Liu, Haiguang; Liu, Tie-Yan; Shi, Yu; Zheng, Shuxin; Noé, Frank; He, ) ===
Nature Machine Intelligence | Volume 6 | May 2024 | 558–567 558
nature machine intelligence
Article https://doi.org/10.1038/s42256-024-00837-3
Predicting equilibrium distributions for molecular systems with deep learning
Shuxin Zheng 1,5 , Jiyan He 2,5, Chang Liu 1,5 , Yu Shi1,5, Ziheng Lu1,5, Weitao Feng1,2, Fusong Ju 1, Jiaxi Wang1, Jianwei Zhu 1, Yaosen Min1, He Zhang1, Shidi Tang 1, Hongxia Hao 1, Peiran Jin1, Chi Chen 3, Frank Noé4, Haiguang Liu 1 & Tie-Yan Liu 1
Advances in deep learning have greatly improved structure prediction of molecules. However, many macroscopic observations that are important for real-world applications are not functions of a single molecular structure but rather determined from the equilibrium distribution of structures. Conventional methods for obtaining these distributions, such as molecular dynamics simulation, are computationally expensive and often intractable. Here we introduce a deep learning framework, called Distributional Graphormer (DiG), in an attempt to predict the equilibrium distribution of molecular systems. Inspired by the annealing process in thermodynamics, DiG uses deep neural networks to transform a simple distribution towards the equilibrium distribution, conditioned on a descriptor of a molecular system such as a chemical graph or a protein sequence. This framework enables the efficient generation of diverse conformations and provides estimations of state densities, orders of magnitude faster than conventional methods. We demonstrate applications of DiG on several molecular tasks, including protein conformation sampling, ligand structure sampling, catalyst–adsorbate sampling and property-guided structure generation. DiG presents a substantial advancement in methodology for statistically understanding molecular systems, opening up new research opportunities in the molecular sciences.
Deep learning methods excel at predicting molecular structures with high efficiency. For example, AlphaFold predicts protein structures with atomic accuracy1, enabling new structural biology applications2–4; neural network-based docking methods predict ligand binding structures5,6, supporting drug discovery virtual screening7,8; and deep learning models predict adsorbate structures on catalyst surfaces9–12. These developments demonstrate the potential of deep learning in modelling molecular structures and states. However, predicting the most probable structure only reveals a fraction of the information about a molecular system in equilibrium.
Molecules can be very flexible, and the equilibrium distribution is essential for the accurate calculation of macroscopic properties. For example, biomolecule functions can be inferred from structure probabilities to identify metastable states; and thermodynamic properties, such as entropy and free energies, can be computed from probabilistic densities in the structure space using statistical mechanics. Figure 1a shows the difference between conventional structure prediction and distribution prediction of molecular systems. Adenylate kinase has two distinct functional conformations (open and closed states), both experimentally determined, but a predicted structure
Received: 2 August 2023
Accepted: 10 April 2024
Published online: 8 May 2024
Check for updates
1Microsoft Research AI4Science, Beijing, China. 2University of Science and Technology of China, Hefei, China. 3Microsoft Quantum, Redmond, WA, USA. 4Microsoft Research AI4Science, Berlin, Germany. 5These authors contributed equally: Shuxin Zheng, Jiyan He, Chang Liu, Yu Shi, Ziheng Lu. e-mail: shuxin.zheng@microsoft.com; chang.liu@microsoft.com; haiguang.liu@microsoft.com; tie-yan.liu@microsoft.com


Nature Machine Intelligence | Volume 6 | May 2024 | 558–567 559
Article https://doi.org/10.1038/s42256-024-00837-3
high-probability regions. This diffusion process is implemented by a deep learning model based upon the Graphormer architecture10 (Fig. 1b), conditioned on a descriptor of the target molecule, such as a chemical graph or a protein sequence. DiG can be trained with structure data from experiments and MD simulations. For data-scarce cases, we develop a physics-informed diffusion pre-training (PIDP) method to train DiG with energy functions (such as force fields) of the systems. In both data-based or energy-supervised modes, the model gets a training signal in each diffusion step independently (Fig. 1b, left arrow symbol), enabling efficient training that avoids long-chain back-propagation. We evaluate DiG on three predictive tasks: protein structure distribution, the ligand conformation distribution in binding pockets and the molecular adsorption distribution on catalyst surfaces. DiG generates realistic and diverse molecular structures in these tasks. For the proteins in this Article, DiG efficiently generated structures resembling major functional states. We further demonstrate that DiG can facilitate the inverse design of molecular structures by applying biased distributions that favour structures with desired properties. This capability can expand molecular design for properties that lack enough data. These results indicate that DiG advances deep learning for molecules from predicting a single structure towards predicting structure distributions, paving the way for efficient prediction of the thermodynamic properties of molecules.
Results
Here, we demonstrate that DiG can be applied to study protein conformations, protein–ligand interactions and molecule adsorption on catalyst surfaces. In addition, we investigate the inverse design capability of DiG through its application to carbon allotrope generation for desired electronic band gaps.
Protein conformation sampling
At physiological conditions, most protein molecules exhibit multiple functional states that are linked via dynamical processes. Sampling of these conformations is crucial for the understanding of protein properties and their interactions with other molecules. Recently, it was
usually corresponds to a highly probable metastable state or an intermediate state (as shown in this figure). A method is desired to sample the equilibrium distribution of proteins with multiple functional states, such as adenylate kinase. Unlike single structure prediction, equilibrium distribution research still depends on classical and costly simulation methods, while deep learning methods are underdeveloped. Commonly, equilibrium distributions are sampled with molecular dynamics (MD) simulations, which are expensive or infeasible13. Enhanced sampling simulations14,15 and Markov state modelling16 can accelerate rare event sampling but need system-specific collective variables and are not easily generalized. Another approach is coarse-grained MD17,18, where deep learning approaches have been proposed19,20. These deep learning coarse-grained methods have worked well for individual molecular systems but have not yet demonstrated generalization. Boltzmann generators21 are a deep learning approach to generate equilibrium distributions by creating a probability flow from a simple reference state, but this also hard to generalize to different molecules. Generalization has been demonstrated for flows generating simulations with longer time steps for small peptides but has not yet been scaled to large proteins22. In this Article, we develop DiG, a deep learning approach to approximately predict the equilibrium distribution and efficiently sample diverse and function-relevant structures of molecular systems. We show that DiG can generalize across molecular systems and propose diverse structures that resemble observations in experiments. DiG draws inspiration from simulated annealing23–26, which transforms a uniform distribution to a complex one through a simulated annealing process. DiG simulates a diffusion process that gradually transforms a simple distribution to the target one, approximating the equilibrium distribution of the given molecular system27,28 (Fig. 1b, right arrow symbol). As the simple distribution is chosen to enable independent sampling and have a closed-form density function, DiG enables independent sampling of the equilibrium distribution and also provides a density function for the distribution by tracking the process. The diffusion process can also be biased towards a desired property for inverse design and allows interpolation between structures that passes through
Supervision from energy function
Structural representation
Pair representation
Node representation
Diffusion step tN – 1
qN – 1
pN = psimple
Diffusion step tN
Diffusion step tN – 2
Diffusion step tN – 3
Supervision from data samples
DiG
Distribution prediction
Descriptor
a
b
Descriptor
Structure prediction Semi-closed
Closed Open
Transition paths
qN – 2 qN – 3
pN – 1 pN – 2 pN – 3 p0
Graphormer
Fig. 1 | Predicting conformational distributions with the DiG framework. a, DiG takes the basic descriptor DD of a target molecular system as input—for example, an amino acid sequence—to generate a probability distribution of structures that aims at approximating the equilibrium distribution and sampling different metastable or intermediate states. In contrast, static structure prediction methods, such as AlphaFold1, aim at predicting one single highprobability structure of a molecule. b, The DiG framework for predicting distributions of molecular structures. A deep learning model (Graphormer10) is
used as modules to predict a diffusion process (→) that gradually transforms a simple distribution towards the target distribution. The model is trained so that the derived distribution pi in each intermediate diffusion time step i matches the corresponding distribution qi in a predefined diffusion process (←) that is set to transform the target distribution to the simple distribution. Supervision can be obtained from both samples (workflow in the top row) and a molecular energy function (workflow shown in the bottom row).


Nature Machine Intelligence | Volume 6 | May 2024 | 558–567 560
Article https://doi.org/10.1038/s42256-024-00837-3
reported that AlphaFold1 can generate alternative conformations for certain proteins by manipulating input information such as multiple sequence alignments (MSAs)29. However, this approach is developed on the basis of varying the depth of MSAs, and it is hard to generalize to all proteins (especially those with a small number of homologous sequences). Therefore, it is highly desirable to develop advanced artificial intelligence (AI) models that can sample diverse structures consistent with the energy landscape in the conformational space29. Here, we show that DiG is capable of generating diverse and functionally relevant protein structures, which is a key capability for being able to efficiently sample equilibrium distributions. Because the equilibrium distribution of protein conformations is difficult to obtain experimentally or computationally, there is a lack of high-quality data for training or benchmarking. To train this model, we collect experimental and simulated structures from public databases. To mitigate the data scarcity, we generated an MD simulation dataset and developed the PIDP training method (see Supplementary Information sections A.1.1 and D.1 for the training procedure and the dataset). The performance of DiG was assessed at two levels: (1) by comparing the conformational distributions against those obtained from extensive (millisecond timescale) atomistic MD simulations and (2) by validating on proteins with multiple conformations. As shown in Fig. 2a, the conformational distributions are obtained from MD simulations for two proteins from the SARS-CoV-2 virus30 (the receptor-binding domain (RBD) of the spike protein and the main protease, also known as 3CL protease; see Supplementary Information section A.7 for details on the MD simulation data). These two proteins are the crucial components of the SARS-CoV-2 and key targets for drug development in treating COVID-1931,32. The millisecond-timescale MD simulations extensively sample conformation space, and we therefore regard the resulting distribution as a proxy for the equilibrium distribution. Taking protein sequences as the descriptor inputs for DiG, structures were generated and compared with simulation data. Although simulation data of RBD and the main protease were not used for DiG training, generated structures resemble the conformational distributions (Fig. 2a). In the two-dimensional (2D) projection space of RBD conformations, MD simulations populate four regions, which are all sampled by DiG (Fig. 2a, left). Four representative structures are well reproduced by DiG. Similarly, three representative structures from main protease simulations are predicted by DiG (Fig. 2a). We noticed that conformations in cluster I are not well recovered by DiG, indicating room for improvement. In terms of conformational coverage, we compared the regions sampled by DiG with those from simulations in the 2D manifold (Fig. 2a), observing that about 70% of the RBD conformations sampled by simulations can be covered with just 10,000 DiG-generated structures (Supplementary Fig. 1). Atomistic MD simulations are computationally expensive, therefore millisecond-timescale simulations of proteins are rarely executed, except for simulations on special-purpose hardware such as the Anton supercomputer13 or extensive distributed simulations combined in Markov state models16. To obtain an additional assessment on the diverse structures generated by DiG, we turn to proteins with multiple structures that have been experimentally determined. In Fig. 2b, we show the capability of DiG in generating multiple conformations for four proteins. Experimental structures are shown in cylinder cartoons, each aligned with two structures generated by DiG (thin ribbons). For example, DiG generated structures similar to either open or closed states of the adenylate kinase protein (for example, backbone root mean square difference (r.m.s.d.) < 1.0 Å to the closed state, 1ake). Similarly, for the drug transport protein LmrP, DiG generated structures covering both states (r.m.s.d. < 2.0 ): one structure is experimentally determined, and the other (denoted as DEER-AF) is the AlphaFold prediction29 supported by double electron electron resonance (DEER) experiments33. For human BRAF kinase, the overall structural difference between the two states is less pronounced. The major difference is in
the A-loop region and a nearby helix (the αC-helix, indicated in the figure)34. Structures generated by DiG accurately capture such regional structural differences. For D-ribose binding protein, the packing of two domains is the major source of structural difference. DiG correctly generates structures corresponding to both the straight-up conformation
Human BRAF kinase αC-helix
6uan 3skc 2dri 1urp
4ake 1ake DEER-AF 6t1z
Adenylate kinase
Cluster representative AlphaFold
Cluster representative AlphaFold
III
III
IV
I
I
II
–1.0 1.0 2.0 IC 1
IC 2
–2.0
–8.0
–6.0
–4.0
–2.0
IC 2
0
3.0
2.0
1.0
–1.0
–2.0
–3.0
0
a
b
0
IV
III
II
II
III
I
I
–1.5 –1.0 –0.5 0.5 1.0 IC 1
0
II
D-ribose binding protein
LmrP membrane protein
Fig. 2 | Distribution and sampling results for protein conformations. a, Structures generated by DiG resemble the diverse conformations of millisecond MD simulations. MD-simulated structures are projected onto the reduced space spanned by two time-lagged independent component analysis (TICA) coordinates (that is, independent component (IC) 1 and 2), and the probability densities are depicted using contour lines. Left: for the RBD protein, MD simulation reveals four highly populated regions in the 2D space spanned by TICA coordinates. DiG-generated structures are mapped to this 2D space (shown as orange dots), with a distribution reflected by the colour intensity. Under the distribution plot, structures generated by DiG (thin ribbons) are superposed on representative structures. AlphaFold-predicted structures (stars) are shown in the plot. Right: the results for the SARS-CoV-2 main protease, compared with MD simulation and AlphaFold prediction results. The contour map reveals three clusters, DiG-generated structures overlap with clusters II and III, whereas structures in cluster I are underrepresented. b, The performance of DiG on generating multiple conformations of proteins. Structures generated by DiG (thin ribbons) are compared with the experimentally determined structures (each structure is labelled by its PDB ID, except DEER-AF, which is an AlphaFold predicted model, shown as cylindrical cartoons). For the four proteins (adenylate kinase, LmrP membrane protein, human BRAF kinase and D-ribose binding protein), structures in two functional states (distinguished by cyan and brown) are well reproduced by DiG (ribbons).


Nature Machine Intelligence | Volume 6 | May 2024 | 558–567 561
Article https://doi.org/10.1038/s42256-024-00837-3
(cylinder cartoon) and the twisted or tilted conformation. If we align one domain of D-ribose binding protein, the other domain only partially matches the twisted conformation as an ‘intermediate’ state. Furthermore, DiG can generate plausible conformation transition pathways by latent space interpolations (see demonstration cases in Supplementary Videos 1 and 2). In summary, beyond static structure prediction for proteins, DiG generates diverse structures corresponding to different functional states.
Ligand structure sampling around binding sites
An immediate extension of protein conformational sampling is to predict ligand structures in druggable pockets. To model the interactions between proteins and ligands, we conducted MD simulations for 1,500 complexes to train the DiG model (see Supplementary Information section D.1 for the dataset). We evaluated the performance of DiG with 409 protein–ligand systems35,36 that are not in the training dataset. The inputs of DiG include protein pocket information (atomic type and position) and the ligan

=== AFsample2: Predicting multiple conformations and ensembles with AlphaFold2 (Wallner, Björn; Kalakoti, Yogesh) ===
bioRxiv preprint doi: https://doi.org/10.1101/2024.05.28.596195; this version posted June 2, 2024. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC 4.0 International license.

AFsample2: Predicting multiple conformations and ensembles with AlphaFold2

Yogesh Kalakotia and Björn Wallnera
aDivision of Bioinformatics, Department of Physics, Chemistry and Biology, Linköping University, 581 83 Linköping, Sweden

1 Abstract

48 driven by the type and extent of structural dynamics asso-

2 Understanding protein dynamics and conformational states car- 49 ciated with the protein system. Conventional experimen3 ries profound scientiﬁc and practical implications for several ar- 50 tal structural biology methods such as X-ray crystallogra4 eas of research, ranging from a general understanding of biolog- 51 phy and cryogenic electron microscopy (cryo-EM) can pro5 ical processes at the molecular level to a detailed understanding 52 vide a few highly accurate snapshots of the overall confor-
6 of disease mechanisms, which in turn can open up new avenues 53 mational ensemble of the protein system (4–6). However,

7 in drug development. Multiple solutions have been recently de- 54 these snapshots only represent a fraction of possible states

8 veloped to widen the conformational landscape of predictions 55 and have to be supplemented by molecular dynamics or other

9 made by Alphafold2 (AF2). Here, we introduce AFsample2, a 56 similar solutions to infer molecular mechanisms. Further-

10

method employing random MSA column masking to reduce the 57

more, computational costs related to MD at biologically rel-

11

inﬂuence of co-evolutionary signals to enhance the structural di58

evant timescales are not viable in practice. Other experimen-

12 13 14

versity of models generated by the AF2 neural network. AFsample2 improves the prediction of alternative states for a broad 59 range of proteins, yielding high-quality end states and diverse 60

tal methods such as Nuclear Magnetic Resonance (NMR) could potentially proﬁle the dynamic nature of the protein

15 conformational ensembles. In the data set of open-closed con- 61 molecule, but are limited by scale (7).

16 formations (OC23), alternate state models improved in 17 out of 62

Recent advancements in in-silico protein structure de-

17 23 cases without compromising the generation of the preferred 63 termination have largely been an outcome of intelligent data

18

state. Consistent results were observed in 16 membrane protein 64

processing and generative artiﬁcial intelligence (AI). Meth-

19

transporters, with improvements in 12 out of 16 targets. TM65

ods like AlphaFold2 (8) (AF2) and RosettaFold (9) have

20 21 22

score improvements to experimental end states were substantial, sometimes exceeding 50%, elevating mediocre scores from 66 0.58 to nearly perfect 0.98. Furthermore, AFsample2 increased 67

demonstrated exceptional levels of success in determining accurate protein structure from evolutionary sequence infor-

23 the diversity of intermediate conformations by 70% compared 68 mation provided as Multiple Sequence Alignments (MSAs).

24 to the standard AF2 system, producing highly conﬁdent mod- 69 However, the default versions of these workﬂows are trained

25 els, that could potentially be on-path between the two states. In 70 to estimate a single high-conﬁdent model of the structure

26 addition, we also propose a way of selecting the end-states in 71 of a protein. This is a limitation since the entire confor-

27 generated model ensembles. These solutions could potentially 72 mational landscape has to be considered in order to get in-

28 enhance the generation and identiﬁcation of alternative protein 73 sights into the mechanistic basis of protein function. There-

29 conformations, thereby providing a more comprehensive under- 74 fore, an ideal sequence-to-structure prediction system should

30

standing of protein function and dynamics. Future work will 75

have the ability to model the entire conformational ensemble

31

focus on validating the accuracy of these intermediate confor76

for a given protein, identify states, and trace physically vi-

32 33

mations and exploring their relevance to functional transitions

in proteins.

77

78

able paths in estimated ensembles. Our recently developed AFsample method (10) captured different conformations of

34

Sampling | Alphafold | Conformations | Ensembles | Diversity

79 multimeric proteins by increasing the sampling rate and in-

35 Correspondence: bjorn.wallner@liu.se

80 troducing noise by enabling dropout layers at inference. The

81 method achieved state-of-the-art performance and was one of

36 Introduction

82 the top-ranked at CASP15 (11) (2022). Additional strategies 83 have also been proposed to induce conformational diversity

37 Proteins are the workhorses of life, serving as the build- 84 in AF2 predictions by subsampling the MSA, using shallow 38 ing blocks of cells and playing crucial roles in almost ev- 85 MSAs (12), in-silico alanine scan as in SPEACH_AF (13), or 39 ery biological process. They exhibit a wide range of func- 86 clustering the MSA as in AFcluster (14). All of these meth40 tions, including catalyzing biochemical reactions, providing 87 ods work by effectively reducing the information to AF2 to 41 structural reinforcement, and even acting as conduits in in- 88 allow the system to explore alternative solutions.

42 tracellular communication (1). Proteins adopt intricate three- 89

In this work, we present AFsample2, which employs

43 dimensional conﬁgurations, often existing within structural 90 random MSA column masking to diminish the constraints

44 ensembles that exhibit various states, collective movements, 91 exerted by co-evolutionary signals in MSA. Thereby increas-

45 and dynamic ﬂuctuations, all essential for executing their 92 ing the structural heterogeneity of models generated with the

46 function (2, 3). Processes such as folding, signal transduc- 93 AF2 neural network. AFsample2 was able to improve the

47 tion, enzyme catalysis, and molecular recognition are all 94 prediction of alternative states for a wide range of proteins.

Kalakoti et al. | bioRχiv | May 28, 2024 | 1–22

bioRxiv preprint doi: https://doi.org/10.1101/2024.05.28.596195; this version posted June 2, 2024. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC 4.0 International license.

Co-evolving residues

Retained

Contact Close conformation

PGSRADV

PGSRADV

HGGR ANR

HGGR ANR

K L QRMN A

XXXXXXX

L GQR AQA

XXXXXXX

Shallow MSA, MSA clustering and subsampling (Co-variance information retained)
Co-evolving residues

Masked
XX

d1
Open conformation d1<d2

PGSRADV

PGSRXDX

HGGR ANR

HGGR XNX

d2

Contact

K L QRMN A L GQR AQA

K LQRXNX L GQR XQX

Masked co-evolution

(a) Two strategies to alter the MSA, (top) MSA subsampling act on the rows of the MSAs, (bottom) column masking, break co-evolving residues and potentially contact networks.

(AvTerM-asgceordeoovferbe2s3t prmootdeielns) Model confidence

0.92

0.90

0.88

0.86

0.84

0.82

0.80

0.78

Best open Best close

0.76 00 05 10 15 20 25 30 35 40 50

MSA randomization (%)

90

85

80

75

70

Mean confidence

65

Best open confidence Best close confidence

00 05 10 15 20 25 30 35 40 50 MSA randomization (%)

(b) Average TM-scores and conﬁdences of best models for OC23 dataset with increasing MSA randomization

543322115000505050 543322115000505050
TM score (open conformation)
109988776655443322110505050505050505050500000000000000000000
TM score (close conformation)
109988776655443322110505050505050505050500000000000000000000

Best open models

Best close models

A0A07QQQQQQQAQOQABPPPPPPP5A5B7596903919730426739Q237DFZUXQEE8X61001213SR0WI9A4Y6RAV71515442SE9TWJMNNUR5682353894819ETTP82641933508381957412870

0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60

0.95 0.90 0.90 0.88 0.85 0.86 0.80 0.84 0.75 0.82 0.70 0.80 0.65 0.78
0.60

MSA randomisation (%)

Number of samples

%
00 05 10 15 20 25 30 35 40 50
Number of samples

(c) Per target best open and closed models for different MSA randomizations

(d) Highest TM-scores for open and close conformation with number of samples

Fig. 1. Overall summary and analysis of MSA randomization strategy in AFsample2. (a) A general outline of the modiﬁcations to the MSA for the AFsample2 pipeline (bottom). Traditional methods retain the information on co-variation, which in turn constrains the inference system to generate structures. AFsample2, on the other hand, remove those constraints by masking columns in the MSA to partially remove this co-variance information, leading to the generation of alternate conformations. (b) Effectiveness of the randomization strategy in terms of generating high-quality models and aggregate conﬁdence for both open and closed states. The results indicate 15% randomization to have the highest TM-scores in the OC23 dataset. (c) Optimal level of randomization on a per target protein. (d) Sampling more models increases the chances of generating better models, and is signiﬁcantly more potent with the proposed randomizations.

95 The improvement was quantiﬁed based on the ability of the 118 been presented in this study that enhance the capability of

96 inference system to generate high-quality end states and di- 119 MSA-based generative models in capturing the conforma-

97 verse conformational ensembles. The models for in particular 120 tional landscape of a given protein system.

98 the alternate state is improved for most of the cases (17/23)

99 in the open-closed data set (OC23) without sacriﬁcing per-
100 formance for the preferred state. The performance is main- 121 Results

101

tain on an additional set of 16 membrane protein transporters, 122

Method Development. The primary objective of this study

102

with the alternate state improved for 12/16 targets. The im123

was to improve the sampling of conformational states by

103

provement as measured by TM-score to experimental end 124

introducing more noise than simply turning on the dropout

104

states is sometimes massive with improvements over 50%, 125

layers at inference. In AFsample2, the noise is introduced

105

basically going from mediocre TM-scores of 0.58 to almost 126

by randomly masking columns in the MSA, with the ra-

106

perfect 0.98. However, the improvements are not only in the 127

tionale to break covariance constraints in the MSA, see

107

end-states but AFsample2 also improves the diversity by gen128

Fig 1a. By breaking covariance signals, the inference sys-

108

erating 70% more conformations in-between the end-states, 129

tem is allowed to explore and arrive at different solutions

109

when compared to the the vanilla AF2 system. While it re130

for the given protein, ultimately increasing the diversity of

110

mains to be demonstrated whether these intermediate confor131

the generated protein ensemble. A similar strategy for intro-

111

mations are accurate on-path representations between states, 132

ducing noise to MSAs has previously been attempted with

112

they are highly conﬁdent models generated by the AF2 infer133

SPEACH_AF (13), where a sliding window of alanines was

113 ence system.

134 introduced at speciﬁc columns in the MSA to break interact-

114

Furthermore, a novel strategy to identify conformational 135 ing residues. Although effective, this strategy was dependent

115 states from a pool of generated models without the aid of 136 on in-silico mutagenesis of the MSA, requiring prior knowl-

116 any experimental reference structures was also developed. 137 edge of the interacting residues. In their implementation,

117 In summary, signiﬁcant methodological improvements have 138 these residues were based on either prior structural informa-

2 | bioRχiv

Kalakoti et al. | AFsample2

bioRxiv preprint doi: https://doi.org/10.1101/2024.05.28.596195; this version posted June 2, 2024. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC 4.0 International license.

139 tion or/and contacts in generated models. AFsample2 does

Protein sequence

AlphaFold inference system

140 not have this limitation and provides a general solution even 141 for cases where such information is unavailable.
142 Effect of MSA masking on generated models. The amount

Genetic database search hhblits/MMseqs
Multiple sequence alignment (MSA)

RAW MSA
PAIRING

MSA track Randomized MSA masking
Pair representation track

Default AF2 architecture
Evoformer>Structure module -> recycles

143 of MSA masking, i.e., the fraction of randomized positions, 144 was observed to be the most important factor in the ability 145 of the inference system to generate alternate conformational 146 states. It was observed that increasing the MSA masking in147 creased the chances of generating end-state conformations

Templates

Reference-free State identification

Generated models (1000)
Rank by confidence

Screen 1: Confidence

Screen 2: Extremity Coverage

x1000
No References available ? Yes Ensemble diversity analysis

Model confidence

148 for a given protein. This trend is summarized in Fig 1b, 149 where MSA masking generates signiﬁcantly better models 150 compared to no masking (0%) for the alternate state (open in 151 these cases) across a set of diverse proteins with well-deﬁned 152 open and closed conformations (the OC23 set, see Methods). 153 The aggregate TM-score for the best alternate (open) confor-

Compute Similarity (TMscore) of top ranked model with all models

TM-score with best

Confidence threshold
TM-score with best

State2

Confidence and extremity-based screening

Identified states

State-guided ensemble analysis for
diversity plots and fill ratio

154 mation increases from 0.795 for no masking to 0.878 with 155 15% masking while showing a marginal improvement from

Fig. 2. Overall workﬂow of the AFsample2 pipeline. It starts by generating MSAs for a given protein sequence. This is followed by randomized MSA masking

156 0.89 to 0.90 for the closed conformation. Beyond 30% mask157 ing, performance drops ﬁrst for the open conformations and 158 subsequently for the closed conformations.

in a way such that a unique MSA proﬁle is fed into the system at every instance of the inference run. The generated ensemble is either passed to the diversity analysis workﬂow or the state-identiﬁcation workﬂow, depending on the availability of reference states.

159

In addition, it has previously been reported that the

160 model conﬁdence of AF2 predictions deteriorates with in- 195 ﬂecting the fact that AF2 has a preference for predicting the

161 creased sub-sampling (15). A similar trend was also observed 196 closed conformation in this case, leaving more room for im-

162 here, where the mean conﬁdence gradually decreased with 197 provement of the open conformation. The increasing trend

163 increasing MSA masking (Fig. 1b). The decrease is linear 198 is most pronounced for fewer samples, reﬂecting the switch

164 from 0% to 35% with a 2% drop in conﬁdence for every 5 199 from no sampling to actual sampling, but it is still increas-

165 percentage points of masking, followed by a rapid drop in 200 ing even up to 1000 samples, indicating that sampling more

166 model conﬁdence beyond 35% masking. Since MSA mask- 201 is always better. However, considering the trade-off between

167 ing essentially removes information, this trend is expected. 202 speed and performance, 1000 samples at 15% masking is a

168 However, it is important to realize that the decrease in model 203 reasonable default.

169 conﬁdence up to 20% masking is not coupled to lower qual170 ity models. It is most likely an effect that the mask itself 204 Overview of the AFsample2. Given a protein sequence, AF171 renders more uncertainty in the prediction, which in turn re- 205 sample2 follows a four-step process to generate diverse pro172 sults in lower model conﬁdence. Overall, 15% randomiza- 206 tein structures using a modiﬁed version of the AF2 inference 173 tion seems to perform marginally better than other settings. 207 system. It starts by (i) querying sequence databases to gen174 However, by analyzing the per target performance (Fig. 1c), 208 erate multiple sequence alignments (MSAs), (ii) Randomly 175 it can be seen that different levels of masking yield the best 209 masking MSA columns with a pre-deﬁned probability (e.g. 176 performance for different target proteins, e.g., 20% masking 210 15%), (iii) running inference on a uniquely masked MSA for 177 generates the best model for P40131, while 5% masking is 211 each model and lastly, (iv) depending on the availability of 178 optimal for P71147. The best TM-scores for each protein at 212 reference states, identifying state representatives with clus179 various masking levels can be visualized in Fig. S1. Even 213 tering, conﬁdence and extremity selection, followed by en180 though the exact magnitude of masking might differ between 214 semble analysis. A schematic representation of the workﬂow 181 targets, it is true that in all cases, masking is always better 215 is summarized in Fig. 2.

182 than no masking for the same level of sampling. For compar-

183 ative analysis and simplicity, AFsample2 using 15% masking 216 Comparing AFsample2 to AFvanilla, AFdropout and

184 was used for the downstream analysis.

217 AFcluster. The performance of AFsample2 was compared to

218 standard AF2 (AFvanilla), standard AF2 with dropout (AF-

185 The importance of sampling. It has be

=== Scalable emulation of protein equilibrium ensembles with generative deep learning (Yim, Jason; Campbell, Andrew; Lewis, Sarah; Hempel, Tim; Luna, José Jiménez; Gas) ===
Scalable emulation of protein equilibrium ensembles with
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


Protein Data Bank (PDB) [5] to predict 3D protein structures that in many cases match experimental accuracy within minutes. For protein function, unfortunately, methods that are both highly accurate and high-throughput are missing, and thus our understanding of how proteins work remains anecdotal. Functional descriptions such as “actin builds up muscle fibers” are human-made attributions that arise from objectively measurable mechanistic properties: (i) What are the conformational states (i.e., sets of different structures) a protein can be in? (ii) Which other molecules can a protein bind to in these different conformations? (iii) What is the probability of these conformational and binding states at a given set of experimental conditions? For example, actin exists in multiple conformational and binding states that are regulated by its cofactors ATP/ADP (Fig. 1a), providing the molecular basis of muscle growth. Available technologies that probe such conformational and binding states and their probabilities at high accuracy are currently not scalable. Single-molecule experiments can provide the full equilibrium distributions of observables such as intramolecular distances [6], but require bespoke molecular constructs and time-consuming data collection. Cryo-electron microscopy can resolve multiple conformational states of biomolecular complexes along with their probabilities [7], but running these experiments is costly both from a monetary and time perspective. Molecular Dynamics (MD) simulation is, in principle, a universal tool that allows both structure and dynamics of biomolecules to be explored at all-atom resolution. However, biomolecular forcefields are far from perfect and the sampling problem renders the study of protein folding or association via MD a feat of epic computational costs for small-sized proteins, even if special-purpose supercomputers or enhanced sampling methods are employed [8, 9]. Machine-learned coarse-grained MD models have an opportunity to achieve similar accuracy as all-atom MD at 2-3 orders of magnitude lower computational cost [10, 11] but are still under development. The grand challenge to complete our understanding of protein function thus motivates the development of a technology that can help elucidate protein conformational states and binding states, as well as their associated probabilities. This technology should ideally achieve an accuracy comparable to a converged MD simulation, or a cryo-EM experiment with multi-conformation analysis, but it should only require a few hours of wall-clock time and cost no more than a few dollars per experiment. Generative systems, such as Boltzmann Generators [12] (BGs), which can efficiently sample arbitrarily-defined equilibrium distributions, indicate that such technologies may be within reach, but are difficult to scale to large proteins. Concurrently, diffusion models and similar approaches are now widely used in protein structure prediction and design [2, 3]. Such models [13–15], as well as perturbation-based derivatives of AlphaFold [16, 17] have also been shown to be capable of generating distinct protein structures and can be combined with MD simulation to alleviate the sampling problem [18]. As yet, generative ML systems have mainly demonstrated an ability to qualitatively sample distinct protein conformational states. A demonstration that generative ML can quantitatively match equilibrium ensembles and predict experimental observables is critical going forward [19]. Here we set out to develop a first version of an ML system that can approximately sample from the equilibrium distribution of protein conformations within a few GPU-hours per experiment — a biomolecular emulator (BioEmu). The biggest challenge in training such a generative model is that no single high-quality data source for training exists due to the aforementioned challenges with experimental methods and MD. We therefore train BioEmu by combining data ranging from a large set of static protein structures and vast amounts of MD simulation to experimental measurements of protein stabilities. We validate the system on a range of tasks: (i) the prediction of protein conformational changes including large domain motions, local unfolding, and the formation of cryptic binding pockets, (ii) the emulation of equilibrium distributions that can be generated by high-throughput MD simulation, and (iii) the prediction of experimentally-measured stabilities of folded states of small proteins by directly generating equilibrium ensembles and explaining structure-stability relationships of mutants. We demonstrate that free energies can be predicted with errors below 1 kcal/mol and are therefore on the order of experimental accuracy. Given its versatility and efficiency, we believe that BioEmu has a variety of practical use cases, ranging from helping with current MD simulation workflows, the interpretation of protein experiments, identification of binding pockets and allosteric mechanisms in drug discovery, and generation of ensembles for dynamical protein design. Importantly, our demonstration that the large upfront costs of MD simulation and experimental data generation can be amortized and the prediction error decreases with an increasing amount of diverse training data indicates a path forward for predicting biomolecular function at genomic scale.
2
available under aCC-BY-NC-ND 4.0 International license.
(which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made
bioRxiv preprint doi: https://doi.org/10.1101/2024.12.05.626885; this version posted December 5, 2024. The copyright holder for this preprint


Free energy landscape
open closed
Net cycle due to energy input
Single steps are equilibrium processes
ATP
ADP
Binding
Dissociation
Opening
Closing
Pretrained distribution MD distribution Equilibrium distribution
Property- prediction fine-tuning
a) Actin example
d) Pretraining Finetuning
Predict properties DG = 3.5 kcal/mol
Understand molecular mechanisms
b) Biomolecular Emulator (BioEmu)
<latexit sha1_base64="TkWXXp5mH4fVRhBAmqY0VBPnUCU=">AAAB+HicbVBNT8JAEJ36ifhB1aOXjcQEL6Q1Bj0SvXjEKB8JNGS7bGHDdtvsbg1Y+SVePGiMV3+KN/+NC/Sg4EsmeXlvJjPz/JgzpR3n21pZXVvf2Mxt5bd3dvcK9v5BQ0WJJLROIh7Jlo8V5UzQumaa01YsKQ59Tpv+8HrqNx+oVCwS93ocUy/EfcECRrA2UtcujFBHsRDFpRF6QnenXbvolJ0Z0DJxM1KEDLWu/dXpRSQJqdCEY6XarhNrL8VSM8LpJN9JFI0xGeI+bRsqcEiVl84On6ATo/RQEElTQqOZ+nsixaFS49A3nSHWA7XoTcX/vHaig0svZSJONBVkvihIONIRmqaAekxSovnYEEwkM7ciMsASE22yypsQ3MWXl0njrOxWypXb82L1KosjB0dwDCVw4QKqcAM1qAOBBJ7hFd6sR+vFerc+5q0rVjZzCH9gff4A3qiR8Q==</latexit>
x ⇠ p(x|S)
c) Score model
MLP
SDE Integrator
Input sequence S
FSTAVHPL
FSTAVHPL
FATAVHPL FSTAIHPL FTTAVQPL
Genetic database search
Pairing
MSA
FSTAVHPL
FSTAVHPL
Evoformer
48 blocks
Single Repr.
Pair Repr.
Protein sequence encoding Denoising diffusion model
xT xt-dt xt x0
q(xt-dt|xt)
p(xt|xt-dt)
... ...
+ noise
Score model Diffusion timestep t Node feat.
Invariant point attention
8 blocks
Node feat.
Structure-based drug design
score model s(xt)
Score st
Pair Repr.
Single Repr.
Backbone frames xt
Backbone frames xt-dt
Equilibrium distribution
Applications
f) Property-prediction fine-tuning
Reweighting
Pretrained Model
Finetuned Model
Diffusion model training
Diffusion model training
Preprocessing
e) AFDB preprocessing 200 M AFDB structures
> 200 ms MD simulations
750 K exp. protein stabilities
Reweighted MD
Aug. cluster AFDB
mmseq + filtering
foldseek + filtering
200 M AFDB struct.
1.4 M seq. clusters
50 K seq. clusters w. diverse structures
Centroid sequence
multi-struct. augmentation Aug. cluster AFDB
ATP hydrolysis
xT
xT-k dt
x0
... ...
folded unfolded
predict
classify
Experimental data DG = 3.5 kcal/mol
Property prediction loss
backprop
Protein stabilities
Subsampling
Fig. 1 Overview of model and architecture. a) Actin conformational changes and filament formation / dissociation as an example for the mechanistic basis of protein function. b) ML model architecture consisting of protein sequence encoder and denoising diffusion model. The diffusion model samples coarse-grained protein structures from an approximate equilibrium distribution, from which properties such as free energy differences can be computed. c) Architecture of the score model used in the denoising diffusion model. d) Data integration and model training pipeline. e) Data processing pipeline for pretraining. f) Experimental property training for finetuning.
2 Model
BioEmu uses a similar model architecture as Distributional Graphormer [13], but with a significantly different training approach. Starting from the input protein sequence, single and pair representations of the sequence are computed using the AlphaFold2 evoformer [1]. These sequence representations serve as input to a denoising diffusion model that generates protein structures (Fig. 1b,c; Sec. S.2). Sequence encoding is invoked only once per protein, and using a second-order integration scheme we generate protein structures in as few as 100 denoising steps (Sec. S.2.3), leading to high sampling efficiency: 10,000 independent protein structures from the learned equilibrium distribution can be sampled within minutes to a few hours on a single GPU, depending on their size. For model training and testing we have developed several new benchmarks and training methods to integrate the heterogeneous data modalities (Sec. S.1, S.3). BioEmu is pretrained on a clustered version of the AlphaFold database (AFDB), using a data augmentation strategy that incentivizes it to sample diverse conformations (Fig. 1d,e, Sec. S.3.2). Starting from this pretrained model, we then continue to train on a mixture of MD data and experimental measurements of protein stability, plus occasional examples from the pretraining data. We have curated and generated a total of over 200 milliseconds of all-atom MD data for small-to-medium proteins
3
available under aCC-BY-NC-ND 4.0 International license.
(which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made
bioRxiv preprint doi: https://doi.org/10.1101/2024.12.05.626885; this version posted December 5, 2024. The copyright holder for this preprint


(Sec. S.3). To mitigate the sampling problem, MD data was reweighed towards equilibrium using either Markov State Models [20], or weights from experimental data (see S.3.5.3), when possible. The reweighed MD data is used in a second training stage of the model (Fig. 1d). The experimental measurements of protein stability that we train on are a subset of the MEGAscale dataset [21], which comprises on the order of a million protein stability measurements (Fig. 1d). As the MEGAscale dataset does not contain structures, we developed an new algorithm called property-prediction fine-tuning (PPFT) to efficiently incorporate experimental measurements into diffusion model training (Fig. 1f, Sec. S.3.6). Finally, to evaluate generalization, we filter our training set such that no protein has more than 40% sequence similarity to any of the reported test proteins of at least 20 residues or longer. The model name BioEmu denotes the fine-tuned model, trained on AFDB, MD simulations and experimental measurements of protein stability. Subsequent results use this model unless otherwise described.
3 Sampling conformational changes related to protein function
We regard the ability to sample distinct biologically relevant conformations qualitatively as a basis to build a quantitative equilibrium sampler. Therefore we first test qualitatively if BioEmu’s samples include known conformational changes and compare this capability with AFCluster [16] and AlphaFlow [14] as two representative baseline methods. Towards this goal, we defined a challenging test set of conformational changes, called OOD60, with a maximum of 60% and 40% sequence similarity to the AlphaFold2 monomer model and our training sets, respectively. Due to the strict sequence similarity constraints, OOD60 only contains 19 proteins, but it features various challenging cases like large-scale conformational changes caused by binding to other biomolecules (Fig. S1). While it is uncertain if all of these conformational changes can be predicted by a single-domain model, the benchmark tests for strong generalization and we find that our model significantly outperforms the two considered baseline approaches (Fig. S5a). In order to evaluate the multi-conformation capabilities of our model more exhaustively, we have also curated a set of around 100 proteins that engage in experimentally-validated domain motions, local unfolding transitions, or cryptic pocket formation. These include some proteins contained in OOD60 as well as proteins that overlap with the AlphaFold2 training set. We confirmed that the model’s performance is similar for proteins that overlap with the AlphaFold2 training set and those that do not, indicating that the benchmark does not test capabilities that the model trivially extracted from evoformer embeddings (Table S4). Furthermore, BioEmu outperforms other methods except for the apo states in the cryptic pocket benchmark, and the difference is especially large for the proteins outside the AlphaFold2 training set (Fig. S5, b-d). Our curated benchmark furthermore demonstrates that our model qualitatively captures functionally relevant protein conformations. For example, proteins can undergo large-scale domain motions as part of their functional cycle. In the open-close transition of Adenylate Kinase, the closed state brings the substrates together to catalyze the ATP + AMP ⇌ 2ADP reaction. Single-molecule experiments have confirmed that opening and closing occurs reversibly on timescales of tens of microseconds when the substrates are bound [22]. BioEmu predicts a range of open and closed states, including close matches with crystallographic structures (Fig. 2a,i). A second example is the open-close transition of LAO-binding protein which is required to bind and release lysine, arginine and ornithine for transport across membranes as part of the ATP-binding cassette protein family (Fig. 2a,ii). Another interesting example of domain motions is that of the receptor module which regulates the concentration of cyclic di-GMP in bacteria. In this case one domain undergoes a large-scale rotation and repacks to the other domain with a completely different contact pattern (Fig. 2a, iii). See Fig. S2 for 15 further examples. Overall, BioEmu predicts 85% of the reference experimental structures with ≤3  ̊A RMSD (Fig. 2a), indicating the model’s ability to predict which protein regions

