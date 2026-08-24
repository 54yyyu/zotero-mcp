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
Where in my library is optimal transport used, and for what distinct purposes -- generative modeling vs data alignment vs trajectory inference vs Schrodinger bridges?

=== CONTEXT ===
[A1] Diffusion Schrödinger Bridge Matching (Campbell, Andrew; Shi, Yuyang; Bortoli, Valentin De; Doucet, Arnaud)
Abstract: Solving transport problems, i.e. finding a map transporting one given distribution to another, has numerous applications in machine learning. Novel mass transport methods motivated by generative modeling have recently been proposed, e.g. Denoising Diffusion Models (DDMs) and Flow Matching Models (FMMs) implement such a transport through a Stochastic Differential Equation (SDE) or an Ordinary Differential Equation (ODE). However, while it is desirable in many applications to approximate the deterministic dynamic Optimal Transport (OT) map which admits attractive properties, DDMs and FMMs are not guaranteed to provide transports close to the OT map. In contrast, Schr\"odinger bridges (SBs) compute stochastic dynamic mappings which recover entropy-regularized versions of OT. Unfortunately, existing numerical methods approximating SBs either scale poorly with dimension or accumulate errors across iterations. In this work, we introduce Iterative Markovian Fitting (IMF), a new methodology for solving SB problems, and Diffusion Schr\"odinger Bridge Matching (DSBM), a novel numerical algorithm for computing IMF iterates. DSBM significantly improves over previous SB numerics and recovers as special/limiting cases various recent transport methods. We demonstrate the performance of DSBM on a variety of problems.
Excerpt: Diffusion Schrödinger Bridge Matching
Yuyang Shi∗
University of Oxford
Valentin De Bortoli∗
ENS ULM
Andrew Campbell
University of Oxford
Arnaud Doucet
University of Oxford
Abstract
Solving transport problems, i.e. finding a map transporting one given distribution
to another, has numerous applications in machine learning. Novel mass trans
port methods motivated by generative modeling have recently been proposed, e.g.
Denoising Diffusion Models (DDMs) and Flow Matching Models (FMMs) im
plement such a transport through a Stochastic Differential Equation (SDE) or an
Ordinary Differential Equation (ODE). However, while it is desirable in many
applications to approximate the deterministic dynamic Optimal Transport (OT)
map which admits attractive properties, DDMs and FMMs are not guaranteed to
provide transports close to the OT map. In contrast, Schrödinger bridges (SBs)
compute stochastic dynamic mappings which recover entropy-regularized versions
of OT. Unfortunately, existing numerical methods approximating SBs either scale
poorly with dimension or accumulate errors across iterations. In this work, we
introduce Iterative Markovian Fitting (IMF), a new methodology for solving SB
problems, and Diffusion Schrödinger Bridge Matching (DSBM), a novel numerical
algorithm for computing IMF iterates. DSBM significantly improves over previ
ous SB numerics and recovers as special/limiting cases various recent transport
methods. We demonstrate the performance of DSBM on a variety of problems.
1 Introduction
Mass transport problems are ubiquitous in machine learning (Peyré and Cuturi, 2019). For discrete
measures, the Optimal Transport (OT) map can be computed exactly but is computationally inten
sive. In a landmark paper, Cuturi (2013) showed that an entropy-regularized version of OT can be
computed more efficiently using the Sinkhorn algorithm (Sinkhorn, 1967). This has enabled the use
of OT techniques in a variety of applications ranging from biology (Bunne et al., 2022) to shape
correspondence (Feydy et al., 2017). However, applications involving high-dimensional continuous
distributions and/or large datasets remain challenging for these techniques.
One of such data-rich applications is generative modeling, a central transport problem in machine
learning which requires designing a deterministic or stochastic mapping transporting a reference
“noise” distribution to the data distribution. For example, Generative Adversarial Networks (Good
fellow et al., 2014) define a static, deterministic transport map, while Denoising Diffusion Models
(DDMs) (Song et al., 2021b; Ho et al., 2020) build a dynamic, stochastic transport map by simulating
a Stochastic Differential Equation (SDE), whose drift is learned using score matching (Hyvärinen,
2005; Vincent, 2011). The excellent performances of DDMs have motivated recent developments
of Bridge Matching and Flow Matching models, which are dynamic transport maps using SDEs
(Song et al., 2021a; Peluchetti, 2021; Liu, 2022; Albergo et al., 2023) or ODEs (Albergo and Vanden
Eijnden, 2023; Heitz et al., 2023; Lipman et al., 2023; Liu et al., 2023b). Compared to DDMs, Bridge
and Flow Matching methods do not rely on a forward “noising” diffusion converging to the reference
distribution in infinite time, and are also more generally applicable as they can approximate transport
maps between two general distributions based on their samples. Nonetheless, these transport maps
∗Equal contribution.
37th Conference on Neural Information Processing Systems (NeurIPS 2023).
arXiv:2303.16852v3 [stat.ML] 12 Dec 2023


DSBM
Bridge Matching
Denoising
Diffusion Flow
Matching
Figure 1: Relationship between
DSBM and existing methods.
Sets for alternating projections Preserved properties
IPF P0 = π0; PT = πT M, R(Q) IMF M; R(Q) P0 = π0, PT = πT
Table 1: Comparison between Iterative Markovian Fitting (IMF)
and Iterative Proportional Fitting (IPF). The Schrödinger Bridge is
the unique P s.t. P0 = π0, PT = πT , P ∈ M, P ∈ R(Q) simul
taneously by Proposition 5. M is the space of (regular) Markov
measures and R(Q) the space of reciprocal measures of Q.
are not necessarily close to the OT map minimizing the Wasserstein-2 metric, which is appealing for
its many attractive properties (Peyré and Cuturi, 2019; Villani, 2009).
In contrast, the Schrödinger Bridge (SB) problem is a dynamic version of entropy-regularized OT
(EOT) (Föllmer, 1988; Léonard, 2014b). The SB is the finite-time diffusion which admits as initial
and terminal distributions the two distributions of interest and is the closest in Kullback–Leibler
divergence to a reference diffusion. Numerous methods to approximate SBs numerically have been
proposed, see e.g. (Bernton et al., 2019; Chen et al., 2016; Finlay et al., 2020; Caluya and Halder,
2021; Pavon et al., 2021), but these techniques tend to be restricted to low-dimensional settings.
Recently, novel techniques using diffusion-based ideas have been proposed in (De Bortoli et al.,
2021; Vargas et al., 2021; Chen et al., 2022) based on Iterative Proportional Fitting (IPF) (Fortet,
1940; Kullback, 1968; Rüschendorf and Thomsen, 1993), a continuous state-space extension of the
Sinkhorn algorithm (Essid and Pavon, 2019). These approaches have been shown to scale better
empirically, but numerical errors tend to accumulate over iterations (Fernandes et al., 2021).
In this paper, our contributions are three-fold. First, we introduce Iterative Markovian Fitting (IMF), a
new procedure to compute SBs which alternates between projecting on the space of Markov processes
and on the reciprocal class, i.e. the measures which have the same bridge as the reference measure of
SB (Léonard et al., 2014). We establish various theoretical results for IMF. Contrary to IPF, the IMF
iterates always preserve the initial and terminal distributions. The differences between IPF and IMF
are presented in Table 1. Second, we propose Diffusion Schrödinger Bridge Matching (DSBM), a
novel algorithm approximating nu

[A2] Optimal transport mapping via input convex neural networks (Makkuva, Ashok Vardhan; Taghvaei, Amirhossein; Oh, Sewoong; Lee, Jason D.)
Abstract: In this paper, we present a novel and principled approach to learn the optimal transport between two distributions, from samples. Guided by the optimal transport theory, we learn the optimal Kantorovich potential which induces the optimal transport map. This involves learning two convex functions, by solving a novel minimax optimization. Building upon recent advances in the field of input convex neural networks, we propose a new framework where the gradient of one convex function represents the optimal transport mapping. Numerical experiments confirm that we learn the optimal transport mapping. This approach ensures that the transport mapping we find is optimal independent of how we initialize the neural networks. Further, target distributions from a discontinuous support can be easily captured, as gradient of a convex function naturally models a {\em discontinuous} transport mapping.
Excerpt: Optimal transport mapping via input convex neural networks
Ashok Vardhan Makkuva * 1 Amirhossein Taghvaei * 2 Jason D. Lee 3 Sewoong Oh 4
Abstract
In this paper, we present a novel and principled approach to learn the optimal transport between two distributions, from samples. Guided by the optimal transport theory, we learn the optimal Kantorovich potential which induces the optimal transport map. This involves learning two convex functions, by solving a novel minimax optimization. Building upon recent advances in the field of input convex neural networks, we propose a new framework to estimate the optimal transport mapping as the gradient of a convex function that is trained via minimax optimization. Numerical experiments confirm the accuracy of the learned transport map. Our approach can be readily used to train a deep generative model. When trained between a simple distribution in the latent space and a target distribution, the learned optimal transport map acts as a deep generative model. Although scaling this to a large dataset is challenging, we demonstrate two important strengths over standard adversarial training: robustness and discontinuity. As we seek the optimal transport, the learned generative model provides the same mapping regardless of how we initialize the neural networks. Further, a gradient of a neural network can easily represent discontinuous mappings, unlike standard neural networks that are constrained to be continuous. This allows the learned transport map to match any target distribution with many discontinuous supports and achieve sharp boundaries.
*Equal contribution. Order decided by a coin flip. 1Department of Electrical and Computer Engineering, University of Illinois at Urbana-Champaign. 2Department of Mechanical and Aerospace Engineering, University of California, Irvine. 3Department of Electrical Engineering, Princeton University, 4Allen School of Computer Science & Engineering, University of Washington. Correspondence to: Ashok <makkuva2@illinois.edu>, Amir <amirhoseintghv@gmail.com>.
Proceedings of the 37 th International Conference on Machine Learning, Vienna, Austria, PMLR 108, 2020. Copyright 2020 by the author(s).
1. Introduction
Finding a mapping that transports mass from one distribution Q to another distribution P is an important task in various machine learning applications, such as deep generative models (Goodfellow et al., 2014; Kingma & Welling, 2013) and domain adaptation (Gopalan et al., 2011; BenDavid et al., 2010). Among infinitely many transport maps T that can map a random variable X from Q such that T (X) is distributed as P , several recent advances focus on discovering some inductive bias to find a transport map with desirable properties. Research in optimal transport has been leading such efforts, in applications such as color transfer (Ferradans et al., 2014), shape matching (Su et al., 2015), data assimilation (Reich, 2013), and Bayesian inference (El Moselhy & Marzouk, 2012). Searching for an optimal transport encourages a mapping that minimizes the total cost of transporting mass from Q to P , as originally formulated in Monge (1781), and provides the inductive bias needed in many such applications. However, finding the optimal transport map in general is a challenging task, especially in high dimensions where efficient approaches are critical.
Algorithmic solutions are well-established for discrete variables; the optimal transport can be found as a solution to linear program. Building upon this mature area, typical approaches for general distributions use quantization, and this becomes intractable for high-dimensional variables we encounter in modern applications (Evans & Gangbo, 1999; Benamou & Brenier, 2000; Papadakis et al., 2014).
To this end, we propose a novel minimax optimization approach to search for the optimal transport under the quadratic distance (i.e. 2-Wassertstein metric). A major challenge in a minimax formulation of optimal transport is that the constraints in the Kantorovich dual formulation (3) are notoriously challenging. They require the evaluation of the functions at every point in the domain, which is not tractable. A common straightforward heuristics sample some points and add those sampled constraints as regularizers. Such regularizations create biases that hinder learning the true optimal transport.
Our key innovation is to depart from this common practice; we instead eliminate the constraints by restricting our search to the set of all convex functions, building upon the fundamental connection from Theorem 3.1. This leads to
arXiv:1908.10962v2 [cs.LG] 17 Jun 2020


Optimal transport mapping via input convex neural networks
(a) Data samples (b) Our transport map (c) Displacement vector field (d) Level sets
Figure 1. Results on Checkerboard dataset. (a) Samples from the source (orange) and target (green) distributions; (b) The learned transport map and the generated distribution, via Algorithm 1; (c) The learned displacement vector field generated by ∇g(y) − y; (d) The level sets of the original dual variable g(y) − 1
2 |y|2. The experimental details are included in Section 4.1.
a novel minimax formulation in (5). Leveraging on recent advances in input convex neural networks, we propose a new architecture and a training algorithm for solving this minimax optimization. We establish the consistency of our proposed minimax formulation in Theorem 3.3. In particular, we show that the solution to this optimization problems yields the exact optimal transport map. We provide stability analysis for the proposed estimator in Theorem 3.6.
Further, when used to train deep generative models, our approach can be viewed as a novel framework to train a generator that is modeled as a gradient of a convex function. We provide a principled training rule based on the optimal transport theory. This ensures that (i) the generator converges to the optimal transport, independent of how we initialize the neural network; and 

[A3] Wasserstein Flow Matching: Generative modeling over families of distributions (Haviv, Doron; Pe'er, Dana; Pooladian, Aram-Alexandre; Amos, Brandon)
Abstract: Generative modeling typically concerns the transport of a single source distribution to a single target distribution by learning (i.e., regressing onto) simple probability flows. However, in modern data-driven fields such as computer graphics and single-cell genomics, samples (say, point-clouds) from datasets can themselves be viewed as distributions (as, say, discrete measures). In these settings, the standard generative modeling paradigm of flow matching would ignore the relevant geometry of the samples. To remedy this, we propose \emph{Wasserstein flow matching} (WFM), which appropriately lifts flow matching onto families of distributions by appealing to the Riemannian nature of the Wasserstein geometry. Our algorithm leverages theoretical and computational advances in (entropic) optimal transport, as well as the attention mechanism in our neural network architecture. We present two novel algorithmic contributions. First, we demonstrate how to perform generative modeling over Gaussian distributions, where we generate representations of granular cell states from single-cell genomics data. Secondly, we show that WFM can learn flows between high-dimensional and variable sized point-clouds and synthesize cellular microenvironments from spatial transcriptomics datasets. Code is available at [WassersteinFlowMatching](https://github.com/DoronHav/WassersteinFlowMatching).
Excerpt: Wasserstein Flow Matching: Generative modeling over families of distributions
Doron Haviv1,2,∗, Aram-Alexandre Pooladian3,∗, Dana Pe’er1,4, Brandon Amos5,†
1Memorial Sloan–Kettering Cancer Center
2Weill Cornell
3Center for Data Science, New York University
4Howard Hughes Medical Institute
5Meta AI
November 4, 2024
Abstract
Generative modeling typically concerns the transport of a single source distribution to a single target distribution by learning (i.e., regressing onto) simple probability flows. However, in modern data-driven fields such as computer graphics and single-cell genomics, samples (say, point-clouds) from datasets can themselves be viewed as distributions (as, say, discrete measures). In these settings, the standard generative modeling paradigm of flow matching would ignore the relevant geometry of the samples. To remedy this, we propose Wasserstein flow matching (WFM), which appropriately lifts flow matching onto families of distributions by appealing to the Riemannian nature of the Wasserstein geometry. Our algorithm leverages theoretical and computational advances in (entropic) optimal transport, as well as the attention mechanism in our neural network architecture. We present two novel algorithmic contributions. First, we demonstrate how to perform generative modeling over Gaussian distributions, where we generate representations of granular cell states from single-cell genomics data. Secondly, we show that WFM can learn flows between high-dimensional and variable sized point-clouds and synthesize cellular microenvironments from spatial transcriptomics datasets. Code is available at WassersteinFlowMatching.
1 Introduction
Today’s abundance of data and scalability of training massive neural networks has made it possible to generate hyper-realistic images on the basis of training examples (OpenAI, 2022), as well as video and audio clips (Vyas et al., 2023; Xing et al., 2023), and, of course, text (Bubeck et al., 2023). All of these are instances of generative modeling: given access to finitely many samples from a distribution, devise a scheme which generates new samples from the same distribution. Generative modeling has also been revolutionary in the biomedical sciences, for drug design (Jumper et al., 2021) and single-cell genomics (Lopez et al., 2018). Nearly all frameworks exploit the notion that datasets (of, say, genomic profiles of cells, images, videos, or corpora of text documents) are
∗Equal contribution. Correspondance to doron.haviv12@gmail.com and ap6599@nyu.edu. †Meta was involved only in an advisory role. All experimentation and data processing was conducted at MSKCC.
1
arXiv:2411.00698v1 [cs.LG] 1 Nov 2024


Method Data type Source Target
FM over Rd x ∈ Rd x ∼ p0 y ∼ p1 FM over M x ∈ M x ∼ p0 y ∼ p1 FM over ∆d μ ∈ P(∆d) μ ∼ p0 ν ∼ p1
Wasserstein FM μ ∈ P(Rd) μ ∼ p0 ν ∼ p1 → Gaussians N (m, Σ) N (mμ, Σμ) ∼ p0 N (mν, Σν) ∼ p1 → Point-Clouds 1
n
P
i δxi
1 m
P
i δxi ∼ p0 1
n
P
j δyj ∼ p1
Wasserstein FM
p0 p1
μν
N (mμ, Σμ) N (mν , Σν )
1 m
P
i δxi
1 n
P
j δyj
Figure 1: Left: Table contrasting FM methods over Rd, general manifolds M, categorical and Dirichlet distributions on the d-simplex ∆d, and finally, our approach, FM problems defined over P(Rd). Right: WFM overview, which learns flows over distributions over distributions.
instantiations of probability measures, and the task is to transform a point sampled from random noise to generate a data point that obeys the distribution of interest. Among the zoo of available generative models, one approach noted for its flexibility and simplicity is Flow Matching (FM) (Albergo and Vanden-Eijnden, 2022; Lipman et al., 2022; Liu et al., 2022). For a fixed target probability measure, FM learns an implicitly defined vector field that can transform a source measure (e.g., the standard Gaussian) to the target. Unlike discrete time and probabilistic generative models (such as Denoising Diffusion Models by Song et al. (2020)), FM learns a deterministic, continuous normalizing flow by regressing onto a simple conditional probability flow. This approach, while originally designed for Euclidean domains, can be readily adopted to Riemannian geometries (Chen and Lipman, 2023). Riemannian flow matching is widely used for generating samples over geometries such as spheres, tori, translation/rotation groups, simplices, triangular meshes, mazes, and molecular positions and structures. The Wasserstein geometry, a canonical geometry over distributions, does not easily fit into any of these existing frameworks and has not been successfully adapted for flow matching. This geometry is useful, for example, in computational graphics where collections of 3D shapes are represented as empirical distributions (point-clouds). Likewise, recent developments in single-cell genomics analysis have demonstrated that gene-expression profiles from groups of cells aggregated via their mean and covariance can capture cellular microenvironments or highlight fine-grain clusters (Haviv et al., 2024b; Persad et al., 2023). For both point-cloud and Gaussian settings, it is natural to search for a unified generative model that respects the underlying geometry of the data, namely, treating each sample as itself a probability distribution.
Contributions. We introduce Wasserstein Flow Matching (WFM), a principled extension of the FM framework lifted to the space of probability distributions. As illustrated in Figure 1, a single point in our source and target datasets is itself a distribution (e.g., a single discrete measure or a single Gaussian), and our aim is to learn vector fields acting on the space of probability distributions and match the optimal transport map, which is the geodesic in Wasserstein space. WFM is an instantiation of Riemannian FM (Chen and Lipman, 2023), where we train a neural model to learn a continuous normalizing flow (CNF) between distributions over distributions. We demonstrate the effectiveness of our approach for generative modeling

[A4] Flow Matching for Generative Modeling (Lipman, Yaron; Chen, Ricky T. Q.; Ben-Hamu, Heli; Nickel, Maximilian; Le, Matt)
Abstract: We introduce a new paradigm for generative modeling built on Continuous Normalizing Flows (CNFs), allowing us to train CNFs at unprecedented scale. Specifically, we present the notion of Flow Matching (FM), a simulation-free approach for training CNFs based on regressing vector fields of fixed conditional probability paths. Flow Matching is compatible with a general family of Gaussian probability paths for transforming between noise and data samples -- which subsumes existing diffusion paths as specific instances. Interestingly, we find that employing FM with diffusion paths results in a more robust and stable alternative for training diffusion models. Furthermore, Flow Matching opens the door to training CNFs with other, non-diffusion probability paths. An instance of particular interest is using Optimal Transport (OT) displacement interpolation to define the conditional probability paths. These paths are more efficient than diffusion paths, provide faster training and sampling, and result in better generalization. Training CNFs using Flow Matching on ImageNet leads to consistently better performance than alternative diffusion-based methods in terms of both likelihood and sample quality, and allows fast and reliable sample generation using off-the-shelf numerical ODE solvers.
Excerpt: Preprint
FLOW MATCHING FOR GENERATIVE MODELING
Yaron Lipman1,2 Ricky T. Q. Chen1 Heli Ben-Hamu2 Maximilian Nickel1 Matt Le1 1Meta AI (FAIR) 2Weizmann Institute of Science
ABSTRACT
We introduce a new paradigm for generative modeling built on Continuous Normalizing Flows (CNFs), allowing us to train CNFs at unprecedented scale. Specifically, we present the notion of Flow Matching (FM), a simulation-free approach for training CNFs based on regressing vector fields of fixed conditional probability paths. Flow Matching is compatible with a general family of Gaussian probability paths for transforming between noise and data samples—which subsumes existing diffusion paths as specific instances. Interestingly, we find that employing FM with diffusion paths results in a more robust and stable alternative for training diffusion models. Furthermore, Flow Matching opens the door to training CNFs with other, non-diffusion probability paths. An instance of particular interest is using Optimal Transport (OT) displacement interpolation to define the conditional probability paths. These paths are more efficient than diffusion paths, provide faster training and sampling, and result in better generalization. Training CNFs using Flow Matching on ImageNet leads to consistently better performance than alternative diffusion-based methods in terms of both likelihood and sample quality, and allows fast and reliable sample generation using off-the-shelf numerical ODE solvers.
1 INTRODUCTION
Deep generative models are a class of deep learning algorithms aimed at estimating and sampling from an unknown data distribution. The recent influx of amazing advances in generative modeling, e.g., for image generation Ramesh et al. (2022); Rombach et al. (2022), is mostly facilitated by the scalable and relatively stable training of diffusion-based models Ho et al. (2020); Song et al. (2020b). However, the restriction to simple diffusion processes leads to a rather confined space of sampling probability paths, resulting in very long training times and the need to adopt specialized methods (e.g., Song et al. (2020a); Zhang & Chen (2022)) for efficient sampling.
In this work we consider the general and deterministic framework of Continuous Normalizing Flows (CNFs; Chen et al. (2018)). CNFs are capable of modeling arbitrary probability path
Figure 1: Unconditional ImageNet-128 samples of a CNF trained using Flow Matching with Optimal Transport probability paths.
and are in particular known to encompass the probability paths modeled by diffusion processes (Song et al., 2021). However, aside from diffusion that can be trained efficiently via, e.g., denoising score matching (Vincent, 2011), no scalable CNF training algorithms are known. Indeed, maximum likelihood training (e.g., Grathwohl et al. (2018)) require expensive numerical ODE simulations, while existing simulation-free methods either involve intractable integrals (Rozen et al., 2021) or biased gradients (Ben-Hamu et al., 2022).
The goal of this work is to propose Flow Matching (FM), an efficient simulation-free approach to training CNF models, allowing the adoption of general probability paths to supervise CNF training. Importantly, FM breaks the barriers for scalable CNF training beyond diffusion, and sidesteps the need to reason about diffusion processes to directly work with probability paths.
1
arXiv:2210.02747v2 [cs.LG] 8 Feb 2023


Preprint
In particular, we propose the Flow Matching objective (Section 3), a simple and intuitive training objective to regress onto a target vector field that generates a desired probability path. We first show that we can construct such target vector fields through per-example (i.e., conditional) formulations. Then, inspired by denoising score matching, we show that a per-example training objective, termed Conditional Flow Matching (CFM), provides equivalent gradients and does not require explicit knowledge of the intractable target vector field. Furthermore, we discuss a general family of per-example probability paths (Section 4) that can be used for Flow Matching, which subsumes existing diffusion paths as special instances. Even on diffusion paths, we find that using FM provides more robust and stable training, and achieves superior performance compared to score matching. Furthermore, this family of probability paths also includes a particularly interesting case: the vector field that corresponds to an Optimal Transport (OT) displacement interpolant (McCann, 1997). We find that conditional OT paths are simpler than diffusion paths, forming straight line trajectories whereas diffusion paths result in curved paths. These properties seem to empirically translate to faster training, faster generation, and better performance.
We empirically validate Flow Matching and the construction via Optimal Transport paths on ImageNet, a large and highly diverse image dataset. We find that we can easily train models to achieve favorable performance in both likelihood estimation and sample quality amongst competing diffusion-based methods. Furthermore, we find that our models produce better trade-offs between computational cost and sample quality compared to prior methods. Figure 1 depicts selected unconditional ImageNet 128×128 samples from our model.
2 PRELIMINARIES: CONTINUOUS NORMALIZING FLOWS
Let Rd denote the data space with data points x = (x1, . . . , xd) ∈ Rd. Two important objects we use in this paper are: the probability density path p : [0, 1] × Rd → R>0, which is a time
dependent1 probability density function, i.e., ∫ pt(x)dx = 1, and a time-dependent vector field, v : [0, 1] × Rd → Rd. A vector field vt can be used to construct a time-dependent diffeomorphic
map, called a flow, φ : [0, 1] × Rd → Rd, defined via the ordinary differential equation (ODE):
d
dt φt(x) = vt(φt(x)) (1)
φ0(x) = x (2)
Previously, Chen et al. (2018) suggested modeling the vector field vt with a neural network, vt(x; θ),
where θ ∈ Rp are its learnable parameters,

[A5] Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow (Liu, Xingchao; Gong, Chengyue; Liu, Qiang)
Abstract: We present rectified flow, a surprisingly simple approach to learning (neural) ordinary differential equation (ODE) models to transport between two empirically observed distributions \pi_0 and \pi_1, hence providing a unified solution to generative modeling and domain transfer, among various other tasks involving distribution transport. The idea of rectified flow is to learn the ODE to follow the straight paths connecting the points drawn from \pi_0 and \pi_1 as much as possible. This is achieved by solving a straightforward nonlinear least squares optimization problem, which can be easily scaled to large models without introducing extra parameters beyond standard supervised learning. The straight paths are special and preferred because they are the shortest paths between two points, and can be simulated exactly without time discretization and hence yield computationally efficient models. We show that the procedure of learning a rectified flow from data, called rectification, turns an arbitrary coupling of \pi_0 and \pi_1 to a new deterministic coupling with provably non-increasing convex transport costs. In addition, recursively applying rectification allows us to obtain a sequence of flows with increasingly straight paths, which can be simulated accurately with coarse time discretization in the inference phase. In empirical studies, we show that rectified flow performs superbly on image generation, image-to-image translation, and domain adaptation. In particular, on image generation and translation, our method yields nearly straight flows that give high quality results even with a single Euler discretization step.
Excerpt: Flow Straight and Fast:
Learning to Generate and Transfer Data with Rectified Flow
Xingchao Liu* University of Texas at Austin xcliu@utexas.edu
Chengyue Gong* University of Texas at Austin cygong@cs.utexas.edu
Qiang Liu University of Texas at Austin lqiang@cs.utexas.edu
Abstract
We present rectified flow, a surprisingly simple approach to learning (neural) ordinary differential equation (ODE) models to transport between two empirically observed distributions π0 and π1, hence providing a unified solution to generative modeling and domain transfer, among various other tasks involving distribution transport. The idea of rectified flow is to learn the ODE to follow the straight paths connecting the points drawn from π0 and π1 as much as possible. This is achieved by solving a straightforward nonlinear least squares optimization problem, which can be easily scaled to large models without introducing extra parameters beyond standard supervised learning. The straight paths are special and preferred because they are the shortest paths between two points, and can be simulated exactly without time discretization and hence yield computationally efficient models. We show that the procedure of learning a rectified flow from data, called rectification, turns an arbitrary coupling of π0 and π1 to a new deterministic coupling with provably non-increasing convex transport costs. In addition, recursively applying rectification allows us to obtain a sequence of flows with increasingly straight paths, which can be simulated accurately with coarse time discretization in the inference phase. In empirical studies, we show that rectified flow performs superbly on image generation, image-to-image translation, and domain adaptation. In particular, on image generation and translation, our method yields nearly straight flows that give high quality results even with a single Euler discretization step.
1 Introduction
Compared with supervised learning, the shared difficulty of various forms of unsupervised learning is the lack of paired input/output data with which standard regression or classification tasks can be invoked. The gist of most unsupervised methods is to find, in one way or another, meaningful correspondences between points from two distributions. For example, generative models such as generative adversarial networks (GAN) and variational autoencoders (VAE) [e.g., 19, 32, 14] seek to map data points to latent codes following a simple elementary (Gaussian) distribution with which the data can be generated and manipulated. Representation learning rests on the idea that if a sufficiently smooth function can map a structured data
*XL and CG contributed equally to this work.
1
arXiv:2209.03003v1 [cs.LG] 7 Sep 2022


distribution to an elementary distribution, it can (likely) be endowed with certain semantically meaningful interpretation and useful for various downstream learning tasks. On the other hand, domain transfer methods find mappings to transfer points from two different data distributions, both observed empirically, for the purpose of image-to-image translation, style transfer, and domain adaption [e.g., 100, 16, 79, 59]. All these tasks can be framed unifiedly as finding a transport map between two distributions:
The Transport Mapping Problem Given empirical observations of two distributions X0 ∼ π0, X1 ∼ π1 on Rd, find a transport map T : Rd → Rd (hopefully nice or optimal in certain sense), such that Z1 := T (Z0) ∼ π1 when Z0 ∼ π0, that is, (Z0, Z1) is a coupling (a.k.a transport plan) of π0 and π1.
Several lines of techniques have been developed depending on how to represent and train the map T . In traditional generative models, T is parameterized as a neural network, and trained with either GAN-type minimax algorithms or (approximate) maximum likelihood estimation (MLE). However, GANs are known to suffer from numerically instability and mode collapse issues, and require substantial engineering efforts and human tuning, which often do not transfer well across different model architecture and datasets. On the other hand, MLE tends to be intractable for complex models, and hence requires approximate variational or Monte Carlo inference techniques such as those used in variational auto-encoders (VAE), or special model structures such as normalizing flow and auto-regressive models, to yield tractable likelihood, causing difficult trade-offs between expressive power and computational cost.
Recently, advances have been made by representing the transport plan implicitly as a continuous time process, such as flow models with neural ordinary differential equations (ODEs) [e.g., 6, 56] and diffusion models by stochastic differential equations (SDEs) [e.g., 73, 23, 80, 11, 82]; in these models, a neural network is trained to represent the drift force of the processes and a numerical ODE/SDE solver is used to simulate the process during inference. The key idea is that, by leveraging the mathematical structures of ODEs/SDEs, the continuous-time models can be trained efficiently without resorting to minimax or traditional approximate inference techniques. The most notable examples are the recent score-based generative models [71–73] and denoising diffusion probabilistic models (DDPM) [23], which we call denoising diffusion methods collectively. These methods allow us to train large-scale diffusion/SDE-based generative models that surpass GANs on image generation in both image quality and diversity, without the instability and mode collapse issues [e.g., 12, 53, 61, 64]. The learned SDEs can be converted into deterministic ODE models for faster inference with the method of probability flow ODEs [73] and DDIM [70].
However, compared with the traditional one-step models like GAN and VAE, a key drawback of continuoustimes models is the high computational cost in inference time: drawing a single point (e.g., image) requires to solve the ODE/SDE with a numerical solver that needs to repeatedly call the expensive ne

[A6] Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design (Yim, Jason; Barzilay, Regina; Jaakkola, Tommi; Campbell, Andrew; Rainforth, Tom;)
Abstract: Combining discrete and continuous data is an important capability for generative models. We present Discrete Flow Models (DFMs), a new flow-based model of discrete data that provides the missing link in enabling flow-based generative models to be applied to multimodal continuous and discrete data problems. Our key insight is that the discrete equivalent of continuous space flow matching can be realized using Continuous Time Markov Chains. DFMs benefit from a simple derivation that includes discrete diffusion models as a specific instance while allowing improved performance over existing diffusion-based approaches. We utilize our DFMs method to build a multimodal flow-based modeling framework. We apply this capability to the task of protein co-design, wherein we learn a model for jointly generating protein structure and sequence. Our approach achieves state-of-the-art co-design performance while allowing the same multimodal model to be used for flexible generation of the sequence or structure.
Excerpt: Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design

arXiv:2402.04997v1 [stat.ML] 7 Feb 2024

Andrew Campbell * 1 Jason Yim * 2 Regina Barzilay 2 Tom Rainforth 1 Tommi Jaakkola 2

Abstract
Combining discrete and continuous data is an important capability for generative models. We present Discrete Flow Models (DFMs), a new flow-based model of discrete data that provides the missing link in enabling flow-based generative models to be applied to multimodal continuous and discrete data problems. Our key insight is that the discrete equivalent of continuous space flow matching can be realized using Continuous Time Markov Chains. DFMs benefit from a simple derivation that includes discrete diffusion models as a specific instance while allowing improved performance over existing diffusion-based approaches. We utilize our DFMs method to build a multimodal flow-based modeling framework. We apply this capability to the task of protein co-design, wherein we learn a model for jointly generating protein structure and sequence. Our approach achieves state-of-the-art co-design performance while allowing the same multimodal model to be used for flexible generation of the sequence or structure.
1. Introduction
Scientific domains often involve continuous atomic interactions with discrete chemical descriptions. Expanding the capabilities of generative models to handle discrete and continuous data, which we refer to as multimodal, is a fundamental problem to enable their widespread adoption in scientific applications (Wang et al., 2023). One such application requiring a multimodal generative model is protein co-design where the aim is to jointly generate continuous protein structures alongside corresponding discrete amino acid sequences (Shi et al., 2022). Proteins have been wellstudied: the function of the protein is endowed through its
*Equal contribution 1Department of Statistics, University of Oxford, UK 2Department of Electrical Engineering and Computer Science, Massachusetts Institute of Technology, Massachusetts, USA. Correspondence to: Andrew Campbell <campbell@stats.ox.ac.uk>, Jason Yim <jyim@csail.mit.edu>.

structure while the sequence is the blueprint of how the structure is made. This interplay motivates jointly generating the structure and sequence rather than in isolation. To this end, the focus of our work is to develop a multimodal generative framework capable of co-design.
Diffusion models (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020) have achieved state-of-the-art performance across multiple applications. They have potential as a multimodal framework because they can be defined on both continuous and discrete spaces (Hoogeboom et al., 2021; Austin et al., 2021). However, their sample time inflexibility makes them unsuitable for multimodal problems. On even just a single modality, finding optimal sampling parameters requires extensive re-training and evaluations (Karras et al., 2022). This problem is exacerbated for multiple modalities. On the other hand, flow-based models (Liu et al., 2023; Albergo & Vanden-Eijnden, 2023; Lipman et al., 2023) improve over diffusion models with a simpler framework that allows for superior performance through sampling flexibility (Ma et al., 2024). Unfortunately, our current inability to define a flow-based model on discrete spaces holds us back from a multimodal flow model.
We address this by 
