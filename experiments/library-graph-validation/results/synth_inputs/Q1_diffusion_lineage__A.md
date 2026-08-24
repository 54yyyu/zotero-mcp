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
Across my diffusion papers, which frameworks claim to unify or generalize the others (flow matching vs stochastic interpolants vs score-based diffusion vs rectified flow), and what is the actual subsumption hierarchy?

=== CONTEXT ===
[A1] Step-by-Step Diffusion: An Elementary Tutorial (Nakkiran, Preetum; Bradley, Arwen; Zhou, Hattie; Advani, Madhu)
Abstract: We present an accessible first course on diffusion models and flow matching for machine learning, aimed at a technical audience with no diffusion experience. We try to simplify the mathematical details as much as possible (sometimes heuristically), while retaining enough precision to derive correct algorithms.
Excerpt: Step-by-Step Diffusion: An Elementary Tutorial
Preetum Nakkiran1, Arwen Bradley1, Hattie Zhou1,2, Madhu Advani1 1Apple, 2Mila, Université de Montréal
We present an accessible first course on diffusion models and flow matching for machine learning, aimed at a technical audience with no diffusion experience. We try to simplify the mathematical details as much as possible (sometimes heuristically), while retaining enough precision to derive correct algorithms.

arXiv:2406.08929v1 [cs.LG] 13 Jun 2024

Contents

1 Fundamentals of Diffusion

3

1.1 Gaussian Diffusion . . . . . . . . . . . . . . . . . . . . . . 3

1.2 Diffusions in the Abstract . . . . . . . . . . . . . . . . . . 5

1.3 Discretization . . . . . . . . . . . . . . . . . . . . . . . . . 6

2 Stochastic Sampling: DDPM

8

2.1 Correctness of DDPM . . . . . . . . . . . . . . . . . . . . 9

2.2 Algorithms . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

2.3 Variance Reduction: Predicting x0 . . . . . . . . . . . . . 11 2.4 Diffusions as SDEs [Optional] . . . . . . . . . . . . . . . . 13

3 Deterministic Sampling: DDIM

16

3.1 Case 1: Single Point . . . . . . . . . . . . . . . . . . . . . . 16

3.2 Velocity Fields and Gases . . . . . . . . . . . . . . . . . . 18

3.3 Case 2: Two Points . . . . . . . . . . . . . . . . . . . . . . 18

3.4 Case 3: Arbitrary Distributions . . . . . . . . . . . . . . . 20

3.5 The Probability Flow ODE [Optional] . . . . . . . . . . . 21

3.6 Discussion: DDPM vs DDIM . . . . . . . . . . . . . . . . 22

3.7 Remarks on Generalization . . . . . . . . . . . . . . . . . 23

4 Flow Matching

25

4.1 Flows . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

4.2 Pointwise Flows . . . . . . . . . . . . . . . . . . . . . . . . 26

4.3 Marginal Flows . . . . . . . . . . . . . . . . . . . . . . . . 26

4.4 A Simple Choice of Pointwise Flow . . . . . . . . . . . . 27

4.5 Flow Matching . . . . . . . . . . . . . . . . . . . . . . . . 28

4.6 DDIM as Flow Matching [Optional] . . . . . . . . . . . . 30

4.7 Additional Remarks and References [Optional] . . . . . . 31

5 Diffusion in Practice

32

A Additional Resources

36

B Omitted Derivations

38

step-by-step diffusion: an elementary tutorial 2
Preface
There are many existing resources for learning diffusion models. Why did we write another? Our goal was to teach diffusion as simply as possible, with minimal mathematical and machine learning prerequisites, but in enough detail to reason about its correctness. Unlike most tutorials on this subject, we take neither a Variational Auto Encoder (VAE) nor an Stochastic Differential Equations (SDE) approach. In fact, for the core ideas we will not need any SDEs, Evidence-Based-Lower-Bounds (ELBOs), Langevin dynamics, or even the notion of a score. The reader need only be familiar with basic probability, calculus, linear algebra, and multivariate Gaussians. The intended audience for this tutorial is technical readers at the level of at least advanced undergraduate or graduate students, who are learning diffusion for the first time and want a mathematical understanding of the subject.
This tutorial has five parts, each relatively self-contained, but covering closely related topics. Section 1 presents the fundamentals of diffusion: the problem we are trying to solve and an overview of the basic approach. Sections 2 and 3 show how to construct a stochastic and deterministic diffusion sampler, respectively, and give intuitive derivations for why these samplers correctly reverse the forward diffusion process. Section 4 covers the closely-related topic of Flow Matching, which can be thought of as a generalization of diffusion that offers additional flexibility (including what are called rectified flows or linear flows). Finally, in Section 5 we return to diffusion and connect this tutorial to the broader literature while highlighting some of the design choices that matter most in practice, including samplers, noise schedules, and parametrizations.
Acknowledgements
We are grateful for helpful feedback and suggestions from many people, in particular: Josh Susskind, Eugene Ndiaye, Dan Busbridge, Sam Power, De Wang, Russ Webb, Sitan Chen, Vimal Thilak, Etai Littwin, Chenyang Yuan, Alex Schwing, and Miguel Angel Bautista Martin.

step-by-step diffusion: an elementary tutorial 3

1 Fundamentals of Diffusion

The goal of generative modeling is: given i.i.d. samples from some unknown distribution p∗(x), construct a sampler for (approximately) the same distribution. For example, given a training set of dog images from some underlying distribution pdog, we want a method of producing new images of dogs from this distribution.
One way to solve this problem, at a high level, is to learn a transformation from some easy-to-sample distribution (such as Gaussian noise) to our target distribution p∗. Diffusion models offer a general framework for learning such transformations. The clever trick of diffusion is to reduce the problem of sampling from distribution p∗(x) into to a sequence of easier sampling problems.
This idea is best explained via the following Gaussian diffusion example. We’ll sketch the main ideas now, and in later sections we will use this setup to derive what are commonly known as the DDPM and DDIM samplers1, and reason about their correctness.

1.1 Gaussian Diffusion

For Gaussian diffusion, let x0 be a random variable in Rd distributed according to the target distribution p∗ (e.g., images of dogs). Then

construct a sequence of random variables x1, x2, . . . , xT, by successively adding independent Gaussian noise with some small scale

σ:

xt+1 := xt + ηt, ηt ∼ N (0, σ2).

(1)

This is called the forward process2, which transforms the data distribu-
tion into a noise distribution. Equation (1) defines a joint distribution over all (x0, x1, . . . , xT), and we let {pt}t∈[T] denote the marginal distributions of each xt. Notice that at large step count T, the distribution pT is nearly Gaussian3, so we can approxim

[A2] Flow Matching for Generative Modeling (Lipman, Yaron; Chen, Ricky T. Q.; Ben-Hamu, Heli; Nickel, Maximilian; Le, Matt)
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

[A3] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions (Albergo, Michael S.; Boffi, Nicholas M.; Vanden-Eijnden, Eric)
Abstract: A class of generative models that unifies flow-based and diffusion-based methods is introduced. These models extend the framework proposed in Albergo & Vanden-Eijnden (2023), enabling the use of a broad class of continuous-time stochastic processes called `stochastic interpolants' to bridge any two arbitrary probability density functions exactly in finite time. These interpolants are built by combining data from the two prescribed densities with an additional latent variable that shapes the bridge in a flexible way. The time-dependent probability density function of the stochastic interpolant is shown to satisfy a first-order transport equation as well as a family of forward and backward Fokker-Planck equations with tunable diffusion coefficient. Upon consideration of the time evolution of an individual sample, this viewpoint immediately leads to both deterministic and stochastic generative models based on probability flow equations or stochastic differential equations with an adjustable level of noise. The drift coefficients entering these models are time-dependent velocity fields characterized as the unique minimizers of simple quadratic objective functions, one of which is a new objective for the score of the interpolant density. We show that minimization of these quadratic objectives leads to control of the likelihood for generative models built upon stochastic dynamics, while likelihood control for deterministic dynamics is more stringent. We also discuss connections with other methods such as score-based diffusion models, stochastic localization processes, probabilistic denoising techniques, and rectifying flows. In addition, we demonstrate that stochastic interpolants recover the Schr\"odinger bridge between the two target densities when explicitly optimizing over the interpolant. Finally, algorithmic aspects are discussed and the approach is illustrated on numerical examples.
Excerpt: Stochastic Interpolants: A Unifying Framework for Flows and Diffusions
Michael S. Albergo∗1, Nicholas M. Boffi∗2, and Eric Vanden-Eijnden2
1Center for Cosmology and Particle Physics, New York University 2Courant Institute of Mathematical Sciences, New York University
November 7, 2023
Abstract
A class of generative models that unifies flow-based and diffusion-based methods is introduced. These models extend the framework proposed in [2], enabling the use of a broad class of continuoustime stochastic processes called ‘stochastic interpolants’ to bridge any two arbitrary probability density functions exactly in finite time. These interpolants are built by combining data from the two prescribed densities with an additional latent variable that shapes the bridge in a flexible way. The time-dependent probability density function of the stochastic interpolant is shown to satisfy a first-order transport equation as well as a family of forward and backward Fokker-Planck equations with tunable diffusion coefficient. Upon consideration of the time evolution of an individual sample, this viewpoint immediately leads to both deterministic and stochastic generative models based on probability flow equations or stochastic differential equations with an adjustable level of noise. The drift coefficients entering these models are time-dependent velocity fields characterized as the unique minimizers of simple quadratic objective functions, one of which is a new objective for the score of the interpolant density. We show that minimization of these quadratic objectives leads to control of the likelihood for generative models built upon stochastic dynamics, while likelihood control for deterministic dynamics is more stringent. We also construct estimators for the likelihood and the cross-entropy of interpolant-based generative models, and we discuss connections with other methods such as score-based diffusion models, stochastic localization processes, probabilistic denoising techniques, and rectifying flows. In addition, we demonstrate that stochastic interpolants recover the Schro ̈dinger bridge between the two target densities when explicitly optimizing over the interpolant. Finally, algorithmic aspects are discussed and the approach is illustrated on numerical examples.
∗Author ordering alphabetical; authors contributed equally.
1
arXiv:2303.08797v3 [cs.LG] 6 Nov 2023


Contents
1 Introduction 3 1.1 Background and motivation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3 1.2 Main contributions and organization . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4 1.3 Related work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5 1.4 Notation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
2 Stochastic interpolant framework 7 2.1 Definitions and assumptions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7 2.2 Transport equations, score, and quadratic objectives . . . . . . . . . . . . . . . . . . . 10 2.3 Generative models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14 2.4 Likelihood control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15 2.5 Density estimation and cross-entropy calculation . . . . . . . . . . . . . . . . . . . . . 17
3 Instantiations and extensions 20 3.1 Diffusive interpolants . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20 3.2 One-sided interpolants for Gaussian ρ0 . . . . . . . . . . . . . . . . . . . . . . . . . . 22 3.3 Mirror interpolants . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24 3.4 Stochastic interpolants and Schr  ̈odinger bridges . . . . . . . . . . . . . . . . . . . . . 25
4 Spatially linear interpolants 26 4.1 Factorization of the velocity field . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26 4.2 Some specific design choices . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27 4.3 Impact of the latent variable γ(t)z and the diffusion coefficient ε(t) . . . . . . . . . . 29 4.4 Spatially linear one-sided interpolants . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
5 Connections with other methods 32 5.1 Score-based diffusion models and stochastic localization . . . . . . . . . . . . . . . . 32 5.2 Denoising methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33 5.3 Rectified flows . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
6 Algorithmic aspects 36 6.1 Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36 6.2 Sampling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
7 Numerical results 40 7.1 Deterministic versus stochastic models: 2D . . . . . . . . . . . . . . . . . . . . . . . . 40 7.2 Deterministic versus stochastic models: 128D Gaussian mixtures . . . . . . . . . . . . 41 7.3 Image generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
8 Conclusion 45
A Bridging two Gaussian mixture densities 47
B Proofs 49 B.1 Proof of Theorems 2.6, 2.7, and 2.8, and Corollary 2.10. . . . . . . . . . . . . . . . . . 49 B.2 Proof of Lemma 2.19 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 54 B.3 Proofs of Lemmas 2.21 and 2.22, and Theorem 2.23. . . . . . . . . . . . . . . . . . . . 56 B.4 Proofs of Lemma 2.25 and Theorem 2.26 . . . . . . . . . . . . . . . . . . . . . . . . . . 58 B.5 Proof of Theorem 3.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 61 B.6 Proof of Lemma 3.12 and Theorem 3.13. . . . . . . . . . . . . . . . . . . . . . . . . . . 62 B.7 Proof of Theorem 5.3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 63 B.8 Proof of Theorem 5.5 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
C Expe

[A4] Understanding Diffusion Models: A Unified Perspective (Luo, Calvin)
Abstract: Diffusion models have shown incredible capabilities as generative models; indeed, they power the current state-of-the-art models on text-conditioned image generation such as Imagen and DALL-E 2. In this work we review, demystify, and unify the understanding of diffusion models across both variational and score-based perspectives. We first derive Variational Diffusion Models (VDM) as a special case of a Markovian Hierarchical Variational Autoencoder, where three key assumptions enable tractable computation and scalable optimization of the ELBO. We then prove that optimizing a VDM boils down to learning a neural network to predict one of three potential objectives: the original source input from any arbitrary noisification of it, the original source noise from any arbitrarily noisified input, or the score function of a noisified input at any arbitrary noise level. We then dive deeper into what it means to learn the score function, and connect the variational perspective of a diffusion model explicitly with the Score-based Generative Modeling perspective through Tweedie's Formula. Lastly, we cover how to learn a conditional distribution using diffusion models via guidance.
Excerpt: arXiv:2208.11970v1 [cs.LG] 25 Aug 2022

Understanding Diﬀusion Models: A Uniﬁed Perspective
Calvin Luo Google Research, Brain Team
calvinluo@google.com
August 26, 2022
Contents
Introduction: Generative Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1 Background: ELBO, VAE, and Hierarchical VAE . . . . . . . . . . . . . . . . . . . . . . . . 2
Evidence Lower Bound . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2 Variational Autoencoders . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4 Hierarchical Variational Autoencoders . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5 Variational Diﬀusion Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6 Learning Diﬀusion Noise Parameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14 Three Equivalent Interpretations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15 Score-based Generative Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17 Guidance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20 Classiﬁer Guidance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21 Classiﬁer-Free Guidance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21 Closing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
Introduction: Generative Models
Given observed samples x from a distribution of interest, the goal of a generative model is to learn to model its true data distribution p(x). Once learned, we can generate new samples from our approximate model at will. Furthermore, under some formulations, we are able to use the learned model to evaluate the likelihood of observed or sampled data as well. There are several well-known directions in current literature, that we will only introduce brieﬂy at a high level. Generative Adversarial Networks (GANs) model the sampling procedure of a complex distribution, which is learned in an adversarial manner. Another class of generative models, termed "likelihood-based", seeks to learn a model that assigns a high likelihood to the observed data samples. This includes autoregressive models, normalizing ﬂows, and Variational Autoencoders (VAEs). Another similar approach is energy-based modeling, in which a distribution is learned as an arbitrarily ﬂexible energy function that is then normalized.
1

Score-based generative models are highly related; instead of learning to model the energy function itself, they learn the score of the energy-based model as a neural network. In this work we explore and review diﬀusion models, which as we will demonstrate, have both likelihood-based and score-based interpretations. We showcase the math behind such models in excruciating detail, with the aim that anyone can follow along and understand what diﬀusion models are and how they work.

Background: ELBO, VAE, and Hierarchical VAE
For many modalities, we can think of the data we observe as represented or generated by an associated unseen latent variable, which we can denote by random variable z. The best intuition for expressing this idea is through Plato’s Allegory of the Cave. In the allegory, a group of people are chained inside a cave their entire life and can only see the two-dimensional shadows projected onto a wall in front of them, which are generated by unseen three-dimensional objects passed before a ﬁre. To such people, everything they observe is actually determined by higher-dimensional abstract concepts that they can never behold.
Analogously, the objects that we encounter in the actual world may also be generated as a function of some higher-level representations; for example, such representations may encapsulate abstract properties such as color, size, shape, and more. Then, what we observe can be interpreted as a three-dimensional projection or instantiation of such abstract concepts, just as what the cave people observe is actually a two-dimensional projection of three-dimensional objects. Whereas the cave people can never see (or even fully comprehend) the hidden objects, they can still reason and draw inferences about them; in a similar way, we can approximate latent representations that describe the data we observe.
Whereas Plato’s Allegory illustrates the idea behind latent variables as potentially unobservable representations that determine observations, a caveat of this analogy is that in generative modeling, we generally seek to learn lower-dimensional latent representations rather than higher-dimensional ones. This is because trying to learn a representation of higher dimension than the observation is a fruitless endeavor without strong priors. On the other hand, learning lower-dimensional latents can also be seen as a form of compression, and can potentially uncover semantically meaningful structure describing observations.

Evidence Lower Bound

Mathematically, we can imagine the latent variables and the data we observe as modeled by a joint distribution p(x, z). Recall one approach of generative modeling, termed "likelihood-based", is to learn a model to maximize the likelihood p(x) of all observed x. There are two ways we can manipulate this joint distribution to recover the likelihood of purely our observed data p(x); we can explicitly marginalize out the latent variable z:

p(x) = p(x, z)dz

(1)

or, we could also appeal to the chain rule of probability:

p(x, z) p(x) =

(2)

p(z|x)

Directly computing and maximizing the likelihood p(x) is diﬃcult because it either involves integrating out all latent variables z in Equation 1, which is intractable for complex models, or it involves having access to a ground truth latent encoder p(z|x) in Equation 2. However, using these two equations, we can derive a term called the Evidence Lower Bound (ELBO), which as its name sugges

[A5] Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (Lou, Aaron; Ermon, Stefano; Meng, Chenlin)
Abstract: Despite their groundbreaking performance for many generative modeling tasks, diffusion models have fallen short on discrete data domains such as natural language. Crucially, standard diffusion models rely on the well-established theory of score matching, but efforts to generalize this to discrete structures have not yielded the same empirical gains. In this work, we bridge this gap by proposing score entropy, a novel loss that naturally extends score matching to discrete spaces, integrates seamlessly to build discrete diffusion models, and significantly boosts performance. Experimentally, we test our Score Entropy Discrete Diffusion models (SEDD) on standard language modeling tasks. For comparable model sizes, SEDD beats existing language diffusion paradigms (reducing perplexity by 25-75%) and is competitive with autoregressive models, in particular outperforming GPT-2. Furthermore, compared to autoregressive mdoels, SEDD generates faithful text without requiring distribution annealing techniques like temperature scaling (around 68× better generative perplexity than un-annealed GPT-2), can trade compute and quality (similar quality with 32× fewer network evaluations), and enables controllable infilling (matching nucleus sampling quality while enabling other strategies besides left to right prompting).
Excerpt: Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution

arXiv:2310.16834v3 [stat.ML] 6 Jun 2024

Aaron Lou 1 Chenlin Meng 1 2 Stefano Ermon 1

Abstract
Despite their groundbreaking performance for many generative modeling tasks, diffusion models have fallen short on discrete data domains such as natural language. Crucially, standard diffusion models rely on the well-established theory of score matching, but efforts to generalize this to discrete structures have not yielded the same empirical gains. In this work, we bridge this gap by proposing score entropy, a novel loss that naturally extends score matching to discrete spaces, integrates seamlessly to build discrete diffusion models, and significantly boosts performance. Experimentally, we test our Score Entropy Discrete Diffusion models (SEDD) on standard language modeling tasks. For comparable model sizes, SEDD beats existing language diffusion paradigms (reducing perplexity by 25-75%) and is competitive with autoregressive models, in particular outperforming GPT-2. Furthermore, compared to autoregressive mdoels, SEDD generates faithful text without requiring distribution annealing techniques like temperature scaling (around 68× better generative perplexity than un-annealed GPT-2), can trade compute and quality (similar quality with 32× fewer network evaluations), and enables controllable infilling (matching nucleus sampling quality while enabling other strategies besides left to right prompting).

The crucial part for any deep generative model is the probabilistic modeling technique. For discrete data such as natural language, autoregressive modeling (Yule, 1971)– arguably the simplest modeling type since it derives from the probabilistic chain rule–has remained the only competitive method for decades. Although modern autoregressive transformers have produced stunning results (Vaswani et al., 2017; Radford et al., 2019), there are limits. For example, the sequential sampling of tokens is slow, hard to control, and often degrades without distribution annealing techniques like nucleus sampling (Holtzman et al., 2019).
To alleviate these issues, researchers have sought alternative approaches to generating text data. In particular, inspired by their success in the image domain, many works have extended diffusion models (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2021c) to language domains (Li et al., 2022; Austin et al., 2021). Yet, despite considerable effort, no such approach yet rivals autoregressive modeling, as they are not competitive on likelihoods, are slower to sample from, and do not generate comparable samples without resorting to heavy annealing and empirical alterations.
In our work, we challenge the longstanding dominance of autoregressive models by introducing Score Entropy Discrete Diffusion models (SEDD). SEDD parameterizes a reverse discrete diffusion process using the ratios of the data distribution. These are learned using score entropy, a novel loss that is analogous to score matching for standard diffusion models (Hyva¨rinen, 2005; Song & Ermon, 2019) and results in several empirical benefits:

1. Introduction
Many recent advances in deep learning have centered around generative modeling. Here, a model learns how to generate novel samples from unstructured data. With the powerful capabilities of modern neural networks, these “generative AI” systems have developed unparalleled capabilities, such as creating images given only text (Ramesh et al., 2022) and answering complex questions (Brown et al., 2020).
1Stanford University 2Pika Labs. Correspondence to: Aaron Lou <aaronlou@stanford.edu>.
Proceedings of the 41 st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s).

1. On core language modeling tasks, SEDD outperforms all existing language diffusion models (Li et al., 2022; Austin et al., 2021; Gulrajani & Hashimoto, 2023; He et al., 2022) by large margins and is competitive with autoregressive models of the same size (beating GPT-2 on its zero-shot perplexity tasks (Radford et al., 2019)).
2. SEDD generates high quality unconditional samples and enables one to naturally trade off compute for quality. When measuring the generative perplexity (given by large models) of unconditional and un-annealed samples from similarly sized models, SEDD beats
We open source our code at github.com/louaaron/ScoreEntropy-Discrete-Diffusion

1

Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution

GPT-2 by 6-8× and can match performance using 32× fewer function evaluations.
3. By directly parameterizing probability ratios, SEDD is highly controllable. In particular, one can prompt SEDD from arbitrary positions without specialized training. For both standard (left to right) and infilling, SEDD outperforms language diffusion models and is comparable with autoregressive models with nucleus sampling (as measured by MAUVE score (Pillutla et al., 2021)).

2. Preliminaries

2.1. Discrete Diffusion Processes

We will be modeling probability distributions over a finite support X = {1, . . . , N }. As the support is discrete, note
that our probability distributions can be represented by probability mass vectors p ∈ RN that are positive and sum to 1. To define a discrete diffusion process, we evolve a family of distributions pt ∈ RN according to the a continuous time Markov process given by a linear ordinary differential
equation (Campbell et al., 2022; Anderson, 2012):

dpt dt

=

Qtpt

p0 ≈ pdata

(1)

Here, Qt are the diffusion matrices RN×N and have non-

negative non-diagonal entries and columns which sum to

zero

(so

that

the

rate

dpt dt

sums

to

0,

meaning

pt

does not

gain or lose total mass). Generally, Qt are simple (e.g.

a simple scalar factor Qt = σ(t)Q) so pt approaches a

limiting distribution pbase as t → ∞.

One can simulate this process by taking small ∆t Euler steps and randomly sampling the resu

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
We address this by introducing a novel flow-based model for discrete data named Discrete Flow Models (DFMs) and thereby unlock a complete framework for flow-based multimodal generative modeling. Our key insight comes from seeing that a discrete flow-based model can be realized using Continuous Time Markov Chains (CTMCs). DFMs are a new discrete generative modeling paradigm: less restrictive than diffusion, allows for sampling flexibility without re-training and enables simple combination with continuous state space flows to form multimodal flow models.
Fig. 1A provides an overview of DFMs. We first define a
