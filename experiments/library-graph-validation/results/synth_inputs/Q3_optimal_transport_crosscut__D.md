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
=== Diffusion Schrödinger Bridge Matching (Campbell, Andrew; Shi, Yuyang; Bortoli, Valentin De; Doucet, Arnaud) ===
Diffusion Schrödinger Bridge Matching
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
novel algorithm approximating numerically the SB solution derived from IMF. DSBM requires at
each iteration solving a simple regression problem in the spirit of Bridge and Flow Matching, and does
not suffer from the time-discretization and “forgetting” issues of previous DSB techniques (De Bortoli
et al., 2021; Vargas et al., 2021; Chen et al., 2022). Finally, we demonstrate the performance of
DSBM on a variety of transport tasks.2
Notations. We denote by P(C), the space of path measures, i.e. P(C) = P(C([0, T ], Rd)) where
T > 0. The subset of Markov path measures associated with an SDE of the form dXt = vt(Xt)dt +
σtdBt, with σ, v locally Lipschitz, is denoted M. For any Q ∈ M, the reciprocal class of Q is
denoted R(Q), see Definition 3. We also denote Qt its marginal distribution at time t, Qs,t the
joint distribution at times s and t, Qs|t the conditional distribution at time s given state at time
t, and Q|0,T ∈ P(C) its diffusion bridge. Unless specified otherwise, all gradient operators ∇
are w.r.t. the variable xt with time index t. Let (X, X ) and (Y, Y) be probability spaces. Given
a Markov kernel K : X × Y → [0, 1] and a probability measure μ defined on X , we write μK
the probability measure on Y such that for any A ∈ Y we have μK(A) = R
X K(x, A)dμ(x). In
particular, for any joint distribution Π0,T over Rd × Rd, we denote the mixture of bridges measure as
Π = Π0,T Q|0,T ∈ P(C), which is short for Π(·) = R
Rd×Rd Q|0,T (·|x0, xT )Π0,T (dx0, dxT ).
2 Dynamic Mass Transport Techniques
2.1 Denoising Diffusion and Bridge Matching Models
Denoising Diffusion Models (Song et al., 2021b; Ho et al., 2020) are a popular class of generative
models. They define a forward noising process Q ∈ M using the SDE dXt = − 1
2 Xtdt + dBt on
the time-interval [0, T ], where X0 ∈ Rd is drawn from the data distribution π0 and (Bt)t∈[0,T ] is a d
2Code can be found at https://github.com/yuyang-shi/dsbm-pytorch.
2


dimensional Brownian motion. This diffusion3 converges towards the standard Gaussian distribution
N(0, Id) as T → ∞. A generative model is given by its time-reversal (Yt)t∈[0,T ] = (XT −t)t∈[0,T ], where Y0 ∼ QT and dYt = { 1
2 Yt + ∇ log QT −t(Yt)}dt + dBt (Anderson, 1982; Haussmann
and Pardoux, 1986). In practice, (Yt)t∈[0,T ] is initialized with Y0 ∼ πT = N(0, Id), and the Stein
score ∇ log Qt(xt) = EQ0|t [∇ log Qt|0(Xt|X0) | Xt = xt] is approximated using a neural network
sθ(t, xt) minimizing the denoising score matching loss EQ0,t [∥∇ log Qt|0(Xt|X0) − sθ(t, Xt)∥2].
An alternative to considering the time-reversal of a forward noising process is to “build bridges”
between the two distributions and learn a mimicking diffusion process. This approach generalizes
DDMs and allows for more flexible choices of sampling processes. We call this framework Bridge
Matching and adopt a presentation similar to Peluchetti (2021); Liu et al. (2022b), where πT is the
data distribution.4 We denote Q ∈ M the path measure associated with the following process
dXt = ft(Xt)dt + σtdBt, X0 ∼ Q0. (1)
Consider now the distribution of this process pinned down at an initial and terminal point x0, xT ,
denoted Q|0,T (·|x0, xT ). Under mild assumptions, the pinned process Q|0,T (·|x0, xT ) is a diffusion
bridge and is given by
dX0,T
t = {ft(X0,T
t ) + σt2∇ log QT |t(xT |X0,T
t )}dt + σtdBt, X0,T
0 = x0, (2)
which satisfies X0,T
T = xT using Doob h-transform theory (Rogers and Williams, 2000). Next, we
define an independent coupling Π0,T = π0 ⊗ πT , and let Π = Π0,T Q|0,T . This path measure Π is a
mixture of bridges. We aim to find a Markov diffusion dYt = {ft(Yt) + vt(Yt)}dt + σtdBt on
[0, T ] which admits the same marginals as Π; i.e. for any t ∈ [0, T ], Yt ∼ Πt, so YT ∼ πT . For
such vt, a generative model for sampling data distribution πT is obtained by simulating (Yt)t∈[0,T ].
It can be verified that indeed Yt ∼ Πt for vt⋆(xt) = σt2EΠT|t [∇ log QT |t(XT |Xt) | Xt = xt]. We
present the theory behind this idea more formally using Markovian projections in Section 3.1. In
practice, we do not have access to vt⋆ and it is learned using neural networks with regression loss
EΠt,T [∥σt2∇ log QT |t(XT |Xt) − vθ(t, Xt)∥2]. (3)
For ft = 0 and σt = σ, Q|0,T is a Brownian Bridge and we have
X0,T
t =t
T xT + (1 − t
T )x0 + σt(Bt − t
T BT ), dX0,T
t = {(xT − X0,T
t )/(T − t)}dt + σtdBt, (4)
with (Bt − t
T BT ) ∼ N(0, t(1 − t
T ) Id). The regression loss (3) associated with (4) is given by
EΠt,T [∥(XT − Xt)/(T − t) − vθ(t, Xt)∥2]. (5)
Letting σ → 0, we recover Flow Matching models (see Appendix A.1 for further details).
2.2 Schrödinger Bridges and Optimal Transport
The Schrödinger Bridge (SB) problem (Schrödinger, 1932) consists in finding a path measure
PSB ∈ P(C) such that
PSB = argminP{KL(P|Q) : P0 = π0, PT = πT }, (6)
where Q ∈ P(C) is a reference path measure. In what follows, we consider Q defined by the diffusion
process (1) which is Markov, and without loss of generality, we assume Q0 = π0. Hence PSB is the
path measure closest to Q in terms of Kullback–Leibler divergence which satisfies the initial and
terminal constraints PS0B = π0 and PSTB = πT .
Another crucial property of PSB is that it can also be defined as a mixture of bridges PSB = ΠS0,BT Q|0,T ,
where ΠS0,BT = argminΠ0,T {KL(Π0,T |Q0,T ) : Π0 = π0, ΠT = πT } is the solution of the static SB
problem (Léonard, 2014b). In particular, for Q associated with (σBt)t∈[0,T ] we have
ΠS0,BT = argminΠ0,T {EΠ0,T [||X0 − XT ||2 − 2σ2T H(Π0,T ) : Π0 = π0, ΠT = πT },
3This is known as the Ornstein–Uhlenbeck (OU) process or VPSDE (Song et al., 2021b).
4To keep notations consistent with existing works, π0 is the data distribution in the context of DDM and SB,
whereas πT is the data distribution in Bridge Matching. However, both SB and Bridge Matching methods allow
transfer between arbitrary distributions π0, πT , so this distinction is not important.
3


where H(μ) denotes the entropy, i.e. ΠS0,BT is the solution of the entropy-regularized OT problem. In
this case, the SB can also be obtained theoretically by solving the following problem (Dai Pra, 1991)
vSB = argminv{R T
0 EPt [||v(t, Xt)||2]dt : dXt = v(t, Xt)dt + σdBt, P0 = π0, PT = πT }.
Then PSB is given by the SDE with drift vSB initialized with X0 ∼ π0. For σ = 0, we recover the
classical OT problem and the Benamou–Brenier formula (Benamou and Brenier, 2000).
A common approach to solve (6) is the Iterative Proportional Fitting (IPF) method (Fortet, 1940; Kullback, 1968; Rüschendorf, 1995) defining a sequence of path measures (P ̃n)n∈N where
 ̃P2n+1 = argmin ̃P{KL( ̃P| ̃P2n) :  ̃PT = πT },  ̃P2n+2 = argmin ̃P{KL( ̃P| ̃P2n+1) : P ̃0 = π0},(7)
with initialization  ̃P0 = Q. This procedure alternates between projections on the set of path measures
with given initial distribution π0 and terminal distribution πT . It can be shown (De Bortoli et al., 2021)
that (P ̃n)n∈N are associated with diffusions and that for any n ∈ N,  ̃P2n+1 is the time-reversal of P ̃2n with initialization πT , and P ̃2n+2 is the time-reversal of P ̃2n+1 with initialization π0. Leveraging this
property, De Bortoli et al. (2021) proposed Diffusion Schrödinger Bridge (DSB), an algorithm which
learns the time-reversals iteratively. In particular, DDMs can be seen as the first iteration of DSB.
3 Iterative Markovian Fitting
3.1 Markovian Projection and Reciprocal Projection
Markovian Projection. Projecting on Markov measures is a key ingredient in our methodology
and in the Bridge Matching framework. This concept was introduced multiple times in the literature
(Gyöngy, 1986; Peluchetti, 2021; Liu et al., 2022b). In particular, we focus on Markovian projection
of path measures given by a mixture of bridges Π = Π0,T Q|0,T ∈ P(C).
Definition 1. Assume that Q is given by (1) and that for any (x0, xT ) ∈ Rd, Q|0,T (·|x0, xT ) is
associated with (X0,T
t )t∈[0,T ] given by dX0,T
t = {ft(X0,T
t ) + σt2∇ log QT |t(xT |X0,T
t )}dt + σtdBt,
with σ : [0, T ] → (0, +∞). Then, when it is well-defined, we introduce the Markovian projection of
Π, M⋆ = projM(Π) ∈ M, which is associated with the SDE
dXt⋆ = {ft(Xt⋆) + vt⋆(Xt⋆)}dt + σtdBt, vt⋆(xt) = σt2EΠT|t [∇ log QT |t(XT |Xt) | Xt = xt].
Note that in our definition σt > 0 so ∇ log QT |t(xT |xt) is well-defined, but Flow Matching can be
recovered as the deterministic case in the limit σt = σ → 0. In the following proposition, we show
that the Markovian projection is indeed a projection for the reverse Kullback–Leibler divergence, and
that it preserves marginals of Πt.
Proposition 2. Assume that σt > 0. Let M⋆ = projM(Π). Then, under mild assumptions, we have
M⋆ = argminM{KL(Π|M) : M ∈ M},
KL(Π|M⋆) = 1
2
RT
0 EΠ0,t [∥σt2EΠT |0,t [∇ log QT |t(XT |Xt) | X0, Xt] − vt⋆(Xt)∥2]/σt2dt.
In addition, we have that for any t ∈ [0, T ], Mt⋆ = Πt. In particular, M⋆T = ΠT .
Reciprocal Projection. While the Markovian projection ensures that the obtained measure is
Markov, the associated bridge measure is not preserved in general, i.e. projM(Π)|0,T ̸= Π|0,T =
Q|0,T . Measures with same bridge as Q are said to be in its reciprocal class (Léonard et al., 2014).
Definition 3. Π ∈ P(C) is in the reciprocal class R(Q) of Q ∈ M if Π = Π0,T Q|0,T . We define the
reciprocal projection of P ∈ P(C) as Π⋆ = projR(Q)(P) = P0,T Q|0,T .
Similarly to Proposition 2, we have the following result, which justifies the term reciprocal projection.
Proposition 4. Let P ∈ P(C), Π⋆ = projR(Q)(P). Then, Π⋆ = argminΠ{KL(P|Π) : Π ∈ R(Q)}.
The reciprocal projection Π⋆ of a Markov path measure M does not preserve the Markov property in
general. In fact, the Schrödinger Bridge is the unique path measure which satisfies the initial and
terminal conditions, is Markov and is in the reciprocal class of Q, see (Léonard, 2014b).
Proposition 5. Let P be a Markov measure in the reciprocal class of Q such that P0 = π0, PT = πT .
Then, under assumptions on Q, π0 and πT , P is unique and is equal to the Schrödinger Bridge PSB.
4


3.2 Iterative Markovian Fitting
Based on Proposition 5, we propose a novel methodology called Iterative Markovian Fitting (IMF) to
solve Schrödinger Bridges. We consider a sequence (Pn)n∈N such that
P2n+1 = projM(P2n), P2n+2 = projR(Q)(P2n+1), (8)
with P0 such that P00 = π0, P0T = πT and P0 ∈ R(Q). These updates correspond to alternatively
performing Markovian projections and reciprocal projections.
Combining Proposition 2 and Definition 3, we get that for any n ∈ N, P0n = π0 and PTn = πT . This
property is in contrast to the IPF algorithm (7) for which the marginals at the initial and final times
are not preserved. We highlight this duality between IPF (7) and IMF (8) in Table 1.
We conclude this section with a theoretical analysis of IMF. First, we start by showing a Pythagorean
theorem for both the Markovian projection and the reciprocal projection.
Lemma 6. Under mild assumptions, if M ∈ M, Π ∈ R(Q) and KL(Π|M) < +∞, we have
KL(Π|M) = KL(Π|projM(Π)) + KL(projM(Π)|M).
If KL(M|Π) < +∞, we have
KL(M|Π) = KL(M|projR(Q)(M)) + KL(projR(Q)(M)|Π).
Using Lemma 6, we have the following proposition.
Proposition 7. Under mild assumptions, we have KL(Pn+1|PSB) ≤ KL(Pn|PSB) < ∞, and
limn→+∞ KL(Pn|Pn+1) = 0.
Hence, for the IMF sequence (Pn)n∈N, the Markov path measures (P2n+1)n∈N are getting closer to
the reciprocal class, while the reciprocal path measures (P2n+2)n∈N are getting closer to the set of
Markov measures. Proposition 7 should be compared with (Rüschendorf, 1995, Proposition 2.1, Equation (2.16)) which shows that, for the IPF sequence ( ̃Pn)n∈N, we have limn→+∞ KL(P ̃n+1|P ̃n) = 0.
This result is similar to Proposition 7 but for the forward Kullback–Leibler divergence.
Using Proposition 7, we finally prove the convergence of the IMF sequence (Pn)n∈N to the
Schrödinger Bridge. This result was first shown in the concurrent work (Peluchetti, 2023, The
orem 2). We present a simpler proof in Appendix C.6.
Theorem 8. Under mild assumptions, the IMF sequence (Pn)n∈N admits a unique fixed point
P⋆ = PSB, and limn→+∞ KL(Pn|P⋆) = 0.
4 Diffusion Schrödinger Bridge Matching
In

=== Optimal transport mapping via input convex neural networks (Makkuva, Ashok Vardhan; Taghvaei, Amirhossein; Oh, Sewoong; Lee, Jason D.) ===
Optimal transport mapping via input convex neural networks
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
Further, when used to train deep generative models, our approach can be viewed as a novel framework to train a generator that is modeled as a gradient of a convex function. We provide a principled training rule based on the optimal transport theory. This ensures that (i) the generator converges to the optimal transport, independent of how we initialize the neural network; and (ii) represent sharp boundaries when the target has multiple disconnected supports. Gradient of a neural network naturally represents discontinuous functions, which is critical in mapping from a single connected support to disconnected supports.
To model convex functions, we leverage Input Convex Neural Networks (ICNNs), a class of scalar-valued neural networks f (x; θ) such that the function x 7→ f (x; θ) ∈ R is convex. These neural networks were introduced by Amos et al. (2016) to provide efficient inference and optimization procedures for structured prediction, data imputation and reinforcement learning tasks. In this paper, we show that ICNNs can be efficiently trained to learn the optimal transport map between two distributions P and Q. To the best of our knowledge, this is the first such instance where ICNNs are leveraged for the well-known task of learning optimal transport maps in a scalable fashion. This framework opens up a new realm for understanding problems in optimal transport theory using parametric convex neural networks, both in theory and practice. Figure 1 provides an example where the optimal transport map has been learned via our proposed Algorithm 1 from the orange distribution to the green distribution.
Notation. P(X ) denotes the set of probability measures on a Polish space X , and B(X ) denotes the Borel sub
sets of X . For P ∈ P(X ) and Q ∈ P(Y), P ⊗ Q denotes the product measure on X × Y. For measurable map T : X → Y, T#P denotes the push-forward of P under T , i.e. (T#P )(A) = P (T −1(A)), ∀A ∈ B(Y). L1(P ) , {f is measurable & ∫ f dP < ∞} denotes the set of integrable functions with respect to P . CVX(P ) denotes the set of all convex functions in L1(P ). Id : x 7→ x denotes the identity function. 〈·, ·〉 and ‖ · ‖ denote the inner-product and `2-Euclidean norm.
2. Background on optimal transport
Let P and Q be two probability distributions on Rd with finite second order moments. The Monge’s optimal transportation problem is to transport the probability mass under Q to P with the least amount of cost1, i.e.
minimize
T :T#Q=P
1
2 EX∼Q‖X − T (X)‖2. (1)
Any transport map T achieving the minimum in (1) is called optimal transport map. Optimal transport map may not exist. In fact, the feasible set in the above optimization problem may itself be empty, for example when Q is a Dirac distribution and P is any non-Dirac distribution.
To resolve the existence issue of the Monge problem (1), Kantorovich introduced a relaxation of the problem,
W2
2 (P, Q) , inf
π∈Π(P,Q)
1
2 E(X,Y )∼π‖X − Y ‖2, (2)
where Π(P, Q) denotes the set of all joint probability distributions (or equivalently, couplings) whose first and second marginals are P and Q, respectively. The optimal value in (2) is the 2-Wasserstein distance W2(·, ·) squared. Any coupling π achieving the infimum is called the optimal coupling. Optimization problem (2) is also referred to as the primal formulation for 2-Wasserstein distance.
1In general, Monge’s problem is defined in terms of cost function c(x, y). This paper is concerned with quadratic cost function c(x, y) = 1
2 ‖x − y‖2 because of its nice geometrical properties
and connection to convex analysis (Villani, 2003, Ch. 2).


Optimal transport mapping via input convex neural networks
Kantorovich also provided a dual formulation for (2), known as the Kantorovich duality (Villani, 2003, Theorem 1.3),
W2
2 (P, Q) = sup
(f,g)∈Φc
EP [f (X)] + EQ[g(Y )], (3)
where Φc denotes the constrained space of functions, defined as Φc ,
{(f, g) ∈ L1(P ) × L1(Q) : f (x) + g(y) ≤
1
2 ‖x − y‖22, ∀(x, y) dP ⊗ dQ a.e.}.
The dual problem (3) can be recast as an stochastic optimization problem by approximating the expectations using independent samples from P and Q. However, there is no easy way to ensure the feasibility of the constraint (f, g) ∈ Φc along the gradient updates. Common approach is to translate the optimization into a tractable form, while sacrificing the original goal of finding the optimal transport map. Concretely, an entropic or a quadratic regularizer is added to the primal problem (2) (Cuturi, 2013; Essid & Solomon, 2018; Peyré et al., 2019; Blondel et al., 2017). Then, the dual to the regularized primal problem is an unconstrained version of (3) with additional penalty term. The unconstrained problem can be numerically solved using Sinkhorn algorithm in discrete setting (Cuturi, 2013) or stochastic gradient methods with suitable function representation in continuous setting (Genevay et al., 2016; Seguy et al., 2017). The optimal transport can then be obtained from f and g, using the first-order optimality conditions of the FenchelRockafellar’s duality theorem (Seguy et al., 2017), or by training a generator through an adversarial computational procedure (Leygonie et al., 2019).
In this paper, we take a different approach: solve the dual problem without introducing a regularization. This builds upon (Taghvaei & Jalali, 2019), where ICNN for the task of approximating the Wasserstein distance and optimal transport map is originally proposed. We bring the idea proposed (Taghvaei & Jalali, 2019) into practice by introducing a novel minimax optimization formulation. We describe our proposed method in Section 3 and provide a detailed comparison in Remark 3.5. Discussion about other related works (Lei et al., 2017; Guo et al., 2019; Xie et al., 2019; Muzellec & Cuturi, 2019; Rabin et al., 2011; Korotin et al., 2019) appears in Appendix D.
3. A novel minimax formulation to learn optimal transport
Our goal is to learn the optimal transport map T ∗ from Q to P , from samples drawn from P and Q, respectively. We use the fundamental connection between optimal transport and Kantorovich dual in Theorem 3.1, to formulate learning T ∗ as a problem of estimating W22(P, Q). However, W22(P, Q) is notoriously hard to estimate. The standard Kantorovich dual formulation in Eq. (3) involves a supremum over a set Φc with infinite constraints, which is challenging to even approximately project onto. To this end, we derive an
alternative optimization formulation in Eq. (5), inspired by the convexification trick (Villani, 2003, Section 2.1.2). This allows us to eliminate the distance constraint of Φc, and instead constrain our search over all convex functions. This constrained optimization can now be seamlessly integrated with recent advances in designing deep neural architectures with convexity guarantees. This leads to a novel minimax optimization to learn the optimal transport.
We exploit the fundamental properties of W22(P, Q) and the corresponding optimal transport to reparametrize the optimization formulation. Note that for any (f, g) ∈ Φc,
f (x) + g(y) ≤ 1
2 ‖x − y‖2
2 ⇐⇒
[1
2 ‖x‖2
2 − f (x)
]
+
[1
2 ‖y‖2
2 − g(y)
]
≥ 〈x, y〉.
Hence reparametrizing 1
2 ‖ · ‖22 − f (·) and 1
2 ‖ · ‖22 − g(·) by f and g respectively, and substituting them in (3) yields
W2
2 (P, Q) = CP,Q − inf
(f ,g)∈  ̃Φc
{
EP [f (X)] + EQ[g(Y )]
}
,
where CP,Q = (1/2)E[‖X‖22 + ‖Y ‖22] is a constant independent of (f, g) and Φ ̃c , {(f, g) ∈ L1(P ) × L1(Q) : f (x) + g(y) ≥ 〈x, y〉, ∀(x, y) dP ⊗ dQ a.e.}. While the above constrained optimization problem involves a pair of functions (f, g), it can be transformed into the following form involving only a single convex function f , thanks to Villani (2003, Theorem 2.9):
W2
2 (P, Q) = CP,Q − inf
f∈CVX(P ) EP [f (X)]+EQ[f ∗(Y )], (4)
where f ∗(y) = supx〈x, y〉 − f (x) is the convex conjugate of f (·).
The crucial tools behind our formulation are the following celebrated results due to Knott-Smith and Brenier (Villani, 2003), which relate the optimal solutions for the dual form in (4) and the primal form in (2).
Theorem 3.1 ((Villani, 2003, Theorem 2.12)). Let P, Q be two probability distributions on Rd with finite second order moments. Then,
1. (Knott-Smith optimality criterion) A coupling π ∈ Π(P, Q) is optimal for the primal (2) if and only if there exists a convex function f ∈ CVX(Rd) such that Supp(π) ⊂ Graph(∂f ). Or equivalently, for all dπalmost (x, y), y ∈ ∂f (x). Moreover, the pair (f, f ∗) achieves the minimum in the dual form (4).
2. (Brenier’s theorem) If Q admits a density with respect to the Lebesgue measure on Rd, then there is a unique optimal coupling π for the primal problem. In particular, the optimal coupling satisfies that
dπ(x, y) = dQ(y)δx=∇f∗(y),


Optimal transport mapping via input convex neural networks
where the convex pair (f, f ∗) achieves the minimum in the dual problem (4). Equivalently, π = (∇f ∗ × Id)#Q.
3. Under the above assumptions of Brenier’s theorem, ∇f ∗ in the unique solution to Monge transportation problem from Q to P , i.e.
EQ‖∇f ∗(Y ) − Y ‖2 = inf
T :T#Q=P EQ‖T (Y ) − Y ‖2.
Remark 3.2. Whenever Q admits a density, we refer to ∇f ∗ as the optimal transport map.
Henceforth, throughout the paper we assume that the distribution Q admits a density in Rd. Note that in view of Theorem 3.1, any optimal pair (f, f ∗) from the dual formulation in (4) provides us an optimal transport map ∇f ∗ pushing forward Q onto P . However, the objective (4) is not amenable to standard stochastic optimization schemes due to the conjugate function f ∗. To this end, we propose a novel minimax formulation in the following theorem where we replace the conjugate with a new convex function.
Theorem 3.3. Whenever Q admits a density in Rd, we have
W2
2 (P, Q) = sup
f ∈CVX(P ), f ∗∈L1(Q)
inf
g∈CVX(Q)
VP,Q(f, g) + CP,Q, (5)
where VP,Q(f, g) is a functional of f, g defined as
VP,Q(f, g) = −EP [f (X)]−EQ[〈Y, ∇g(Y )〉−f (∇g(Y ))].
In addition, there exists an optimal pair (f0, g0) achieving the infimum and supremum respectively, where ∇g0 is the optimal transport map from Q to P .
Proof sketch. The proof follows from the inequality 〈y, ∇g(y)〉 − f (∇g(y)) ≤ f ∗(y) for all functions g, and then taking the expectation over Q, and observing that the equality is achieved with g = f ∗. The technical details appear in Appendix A.
Remark 3.4. For any convex function f , the function g ∈ L1(Q) that achieves the infimum in (5) is convex and equals f ∗. Therefore, the constraint g ∈ CVX(Q) can be relaxed to g ∈ L1(Q) without changing the optimal value and optimizing functions. We numerically observe that the optimization algorithm performs better under this relaxation.
Formulation (5) now provides a principled approach to learn the optimal transport mapping ∇g(·) as a solution of a minimax optimization. Since the optimization involves the search over the space of convex functions, we utilize the recent advances in input convex neural networks (ICNNs) to parametrize them as discussed in the following section.
W1 W2 WL-1
...
...
Figure 2. The input convex neural network (ICNN) architecture.
3.1. Minimax optimization over ICNNs
We propose using parametric models based on deep neural networks to approximate the set of convex functions. This is known as input convex neural networks (Amos et al., 2016), denoted by ICNN(Rd). We propose estimating the following approximate Wasserstein-2 distance, from samples:
W ̃2
2 (P, Q) = sup
f ∈ICNN(Rd)
inf
g ∈ICNN(Rd )
VP,Q(f, g)+CP,Q. (6)
ICNNs are a class of scalar-valued neural networks f (x; θ) such that the function x 7→ f (x; θ) ∈ R is convex.
The neural network architecture for an ICNN is as follows. Given an input x ∈ Rd, the mapping x 7→ f (x; θ) is given by a L-layer feed-forward NN using the following equations for l = 0, 1, . . . , L − 1:
zl+1 = σl(Wlzl + Alx + bl), f (x; θ) = zL,
where {Wl}, {Al} are weight matrices (with the convention that W0 = 0), and {bl} are the bias terms. σl denotes the entry-wise activation function at the layer l. This is illustrated in Figure 2. We denote the total set of parameters by θ = ({Wl}, {Al}, {bl}). It follows from Amos et al. (2016, Proposition 1) that f (x; θ) is convex in x provided
(i) all entries of the weights Wl are non-negative;
(ii) activation function σ0 is convex;
(iii) σl is convex and non-decreasing, for l = 1, . . . , L − 1.
While ICNNs are a specific parametric cla

=== SCOT: Single-Cell Multi-Omics Alignment with Optimal Transport (Demetci, Pinar; Santorella, Rebecca; Sandstede, Björn; Noble, William Stafford; ) ===
Research Articles
SCOT: Single-Cell Multi-Omics Alignment with Optimal
Transport
PINAR DEMETCI,1,2,*,i REBECCA SANTORELLA,3 BJO ̈ RN SANDSTEDE,3,* WILLIAM STAFFORD NOBLE,4,5 and RITAMBHARA SINGH1,2,ii
ABSTRACT
Recent advances in sequencing technologies have allowed us to capture various aspects of the genome at single-cell resolution. However, with the exception of a few of co-assaying technologies, it is not possible to simultaneously apply different sequencing assays on the same single cell. In this scenario, computational integration of multi-omic measurements is crucial to enable joint analyses. This integration task is particularly challenging due to the lack of sample-wise or feature-wise correspondences. We present single-cell alignment with optimal transport (SCOT), an unsupervised algorithm that uses the Gromov–Wasserstein optimal transport to align single-cell multi-omics data sets. SCOT performs on par with the current state-of-the-art unsupervised alignment methods, is faster, and requires tuning of fewer hyperparameters. More importantly, SCOT uses a self-tuning heuristic to guide hyperparameter selection based on the Gromov–Wasserstein distance. Thus, in the fully unsupervised setting, SCOT aligns single-cell data sets better than the existing methods without requiring any orthogonal correspondence information.
Keywords: data integration, manifold alignment, multi-omics, optimal transport, single-cell genomics.
1. INTRODUCTION
T
he growing variety of single-cell assays allows us to measure the heterogeneous landscape of cell state in a sample, revealing distinct subpopulations and their developmental and regulatory trajectories across time. Different technologies can interrogate different molecular aspects of the cell, such as gene expression, protein synthesis, chromatin accessibility, DNA methylation, histone modifications, and chromatin three-dimensional (3D) confirmation. Combining data generated by these single-cell assays can
1Center for Computational Molecular Biology, Brown University, Providence, Rhode Island, USA. 2Department of Computer Science, Brown University, Providence, Rhode Island, USA. 3Division of Applied Mathematics, Brown University, Providence, Rhode Island, USA. 4Department of Genome Sciences, University of Washington, Seattle, Washington, USA. 5Paul G. Allen School of Computer Science and Engineering, University of Washington, Seattle, Washington, USA. *These authors contributed equally to this work. iORCID ID (https://orcid.org/0000-0002-5644-0326). iiORCID ID (https://orcid.org/0000-0002-7523-160X).
JOURNAL OF COMPUTATIONAL BIOLOGY Volume 29, Number 1, 2022 # Mary Ann Liebert, Inc. Pp. 3–18 DOI: 10.1089/cmb.2021.0446
3
Downloaded by COLUMBIA UNIVERSITY LIBRARY from www.liebertpub.com at 02/16/25. For personal use only.


provide novel insights into the interactions between these molecular views and their joint regulatory mechanisms. Hence, learning this combined information is critical to our understanding of complex biological processes and heterogeneous diseases. Despite its importance, combining single-cell multi-omics data is a challenging task. Aside from a few recent co-assay procedures that simultaneously isolate separate molecular material for each measurement, applying multiple assays on the same single cell is impossible. Sometimes, sequencing assays need access to the same molecular material, such as with chromatin accessibility and 3D chromatin conformation capture assays. In such cases, the measurements are taken by dividing a cell population into subpopulations and assaying them separately, losing the potential for 1–1 correspondence of cells that is required for easy data integration. Moreover, in cases where we can take measurements in the same cell and preserve the 1–1 correspondences, the choice of the experimental method for processing cells and isolating molecular materials of interest can introduce additional challenges and noise in the co-assayed data (Hu et al., 2018). For example, for simultaneous isolation of DNA and RNA, there are two general approaches: physical separation of DNA and RNA followed by separate amplification, or simultaneous preamplification followed by physical separation of the two materials. For the first approach, separation techniques such as centrifugation and micropipetting are not high-throughput; however, high-throughput approaches (Macaulay et al., 2015; Angermueller et al., 2016) have been found to introduce variability in coverage and sequencing depth of various genomic regions in the isolated DNA (Hu et al., 2018). In recent years, computational methods have been developed to solve the single-cell data integration problem. Many of these methods combine different experiments from a single modality such as RNA sequencing for correcting batch effects (Welch et al., 2017, 2019; Amodio and Krishnaswamy, 2018; Barkas et al., 2019; Stuart et al., 2019). However, integrating data from multiple modalities such as gene expression and DNA methylation presents unique challenges. For example, when we measure different properties of a cell, we cannot a priori identify correspondences between features in the two domains. Accordingly, integrating two or more single-cell data modalities requires methods that rely on neither common cells nor features across the data types. This aspect prevents the application of some existing single-cell alignment methods to unsupervised settings because they require some correspondence information to perform alignment (Welch et al., 2017, 2019; Amodio and Krishnaswamy, 2018; Barkas et al., 2019; Stuart et al., 2019). Earlier versions of the popular batch integration method Seurat required correspondence information in the form of cells from a similar biological state that are shared across the two data sets (known as ‘‘anchor points’’). While a more recent version automatically selects these anchor points, it still requires features from one domain to be mapped to the other domain to perform the single-cell alignment (Stuart et al., 2019). This mapping might be possible for experiments such as gene expression and chromatin accessibility, where one can map the chromatin region read counts to the corresponding gene regions. However, it can be difficult to perform for other sequencing assay combinations. Furthermore, Cao et al. (2020) have shown that such methods do not yield quality alignments in unsupervised settings. Multiple approaches have tried to align data sets in an entirely unsupervised manner. One of the earliest attempts, the joint Laplacian manifold alignment algorithm, constructs eigenvector projections based on k-nearest neighbor (k-NN) graph Laplacians of the data (Wang and Mahadevan, 2009). The generalized unsupervised manifold alignment (GUMA) (Cui et al., 2014) algorithm seeks a 1–1 correspondence between two data sets based on optimization of a local geometry matching term. Liu et al. (2019) showed that these methods do not perform well on the single-cell alignment task and proposed a manifold alignment (MA) algorithm based on the maximum mean discrepancy (MMD) measure, called MMD-MA. Another method, UnionCom (Cao et al., 2020), extends GUMA to perform unsupervised topological alignment and makes it more suitable for single-cell multi-omics integration. While MMD-MA aims to match the global distributions of the data sets in a shared latent space, UnionCom emphasizes learning both local and global alignments between the two distributions. Neither method requires any correspondence information, either among samples or features, to perform an alignment. The respective articles demonstrate state-of-the-art performance on simulated and real data sets. Although these results are encouraging, MMD-MA and UnionCom require that the user specify three and four hyperparameters, respectively. Hyperparameter selection can significantly affect the quality of alignments. Therefore, in an unsupervised real-world setting with no validation data on correspondences, hyperparameter tuning can be difficult to perform and can lead to subpar alignments.
4 DEMETCI ET AL.
Downloaded by COLUMBIA UNIVERSITY LIBRARY from www.liebertpub.com at 02/16/25. For personal use only.


In this article, we propose an unsupervised alignment method based on optimal transport theory. Optimal transport finds the most cost-effective way to move data points from one domain to another. One way to think about it is as the problem of moving a pile of sand to fill in a hole through the least amount of work. Traditionally, optimal transport problems have been difficult to compute, especially for large-scale data sets. However, subsequent relaxations (Kantorovich, 1942; Peyre ́ et al., 2019) modify the original optimal transport problem, making it more applicable and easier to compute. Recently, several regularization procedures (Peyre ́ et al., 2016) have further improved the computational scalability of optimal transport. In biology, an emerging number of applications are using optimal transport to learn a mapping between data distributions (Alvarez-Melis and Jaakkola, 2018; Yang et al., 2018; Schiebinger et al., 2019; Yang and Uhler, 2019; Cang and Nie, 2020). Schiebinger et al. (2019) used it to study temporal changes in gene expression by using regularized unbalanced optimal transport to compute expression differences between time points. SpaOTsc (Cang and Nie, 2020) maps cells with high ligand expression onto cells with high receptor expression to recover cell signaling relationships in spatially resolved single-cell RNA-seq data sets. ImageAEOT (Yang et al., 2018) maps single-cell images to a common latent space through an autoencoder and then uses optimal transport to track cell trajectories. In related work, the same authors used autoencoders and optimal transport to learn transport maps among multiple domains (Yang and Uhler, 2019). However, the application of their method to single-cell data sets requires some form of supervision, such as class labels, to be used during transport. The classic optimal transport problem requires data sets from the same metric space. Me ́moli (2011) generalized optimal transport to the Gromov–Wasserstein distance, which compares metric spaces directly instead of comparing samples across spaces, making optimal transport suitable for multimodal alignment. In natural language processing, Alvarez-Melis and Jaakkola (2018) used this approach to measure similarities between pairs of words across languages to compute the similarity between languages. As far as we are aware, the only biological application of the Gromov–Wasserstein optimal transport comes from the study by Nitzan et al. (2019), which uses it to reconstruct the spatial organization of cells from transcriptional profiles. We present single-cell alignment with optimal transport (SCOT), an unsupervised algorithm that uses the Gromov–Wasserstein-based optimal transport to align single-cell multi-omics data sets (presented schematically in Fig. 1). Like UnionCom, SCOT aims to preserve local geometry when aligning single-cell data. SCOT achieves this by constructing a k-NN graph for each data set (or domain) and then computing graph distance matrices for each k-NN graph to capture the intra-domain distances. SCOT then finds a probabilistic coupling matrix that minimizes the discrepancy between the intra-domain distance matrices. Finally, it uses the coupling matrix to project one single-cell data set onto another through barycentric projection, thus aligning them. Unlike MMD-MA and UnionCom, SCOT requires tuning only two hyperparameters and is robust to the choice of one. We compare the alignment performance of SCOT with MMD-MA and UnionCom on four simulated and two real-world data sets. SCOT aligns data sets as well as the state-of-the-art methods and scales well with increasing numbers of samples. Moreover, we demonstrate that the Gromov–Wasserstein distance can guide SCOTs hyperparameter tuning in a fully unsupervised setting when no orthogonal alignment information is available. Thus, unlike other methods, SCOT provides a heuristic for hyperparameter selection without validation data. The source code for SCOT is publicly available at http:// rsinghlab.github.io/SCOT.
FIG. 1. Schematic of SCOT alignment of single-cell multi-omics data. A population of cells is aliquoted for different single-cell sequencing assays. SCOT constructs k-NN graphs based on sample-wise correlations and finds a probabilistic coupling between the samples of each domain that minimizes the distance between the two intra-domain graph distance matrices. Barycentric projection projects one domain onto another based on this coupling matrix. SCOT, single-cell alignment with optimal transport.
SINGLE-CELL ALIGNMENT WITH OPTIMAL TRANSPORT 5
Downloaded by COLUMBIA UNIVERSITY LIBRARY from www.liebertpub.com at 02/16/25. For personal use only.


2. METHODS
SCOT relies on the Gromov–Wasserstein optimal transport to move data points from one domain to another while preserving the original local geometry. The goal of the transport problem at the core of SCOT is to find an ideal ‘‘coupling’’ (also called ‘‘correspondence’’) matrix that describes the probability of alignment between each point across domains. In this section, we first introduce optimal transport theory, followed by its extension to the Gromov–Wasserstein distance. Then, we present the details of our algorithm. We have two data sets representing two domains, X = (x1‚ x2‚ . . . ‚ xnx ) from X and Y = (y1‚ y2‚ . . . ‚ yny ) from Y. The data sets have nx and ny points, respectively. We do not require any correspondence information or assume that there is any ground truth for 1–1 correspondence between samples or features, but we do assume that there is some underlying shared biology (e.g., cells across the data sets sharing a lineage or belonging to shared cell types), so that the data sets can be meaningfully aligned.
2.1. Optimal transport
The Kantorovich optimal transport problem seeks to find a minimal cost mapping between two probability distributions or discrete measures (Peyre ́ et al., 2019). Referring back to the problem of moving a sand pile to fill in a hole, the Kantorovich optimal transport allows us to split the mass of a grain of sand instead of moving the whole grain; therefore, the mappings need not be 1–1. Consider discrete measures l and  as such
l=
n Xx
i=1
pidxi and  =
n Xy
j=1
qjdyj ‚ (1)
where Pnx
i = 1 pi = 1 = Pny
j = 1 qj‚ pi  0‚ qj  0 and dxi is the Dirac measure. This optimal transport problem finds a minimal coupling p that attains
min
p2P(‚ l)
n Xx
i=1
n Xy
j=1
c(i‚ j)p(i‚ j) (2)
subject to : p(i‚ j)  0‚
n Xx
i=1
p(i‚ j) = qj‚
n Xy
j=1
p(i‚ j) = pi
where c(i‚ j) is a cost function defined over the samples from the two data sets and P(l‚ ) is the set of couplings of l and  given by
P(l‚ ) = fp 2 Rnx · ny
+ : p1ny = l‚ pT 1nx = g: (3)
Intuitively, the cost function says how many resources it will take to move point xi in the first data set to point yj in the second data set, and the coupling p relates the two discrete measures l and  by correspondence probabilities. Each row pi tells us how to split the mass of data point xi onto the points yj for j = 1‚ . . . ‚ ny, and the condition p1ny = p requires that the sum of each row pi is equal to pi, the probability of sample xi. The discrete optimal transport problem finds a coupling matrix, G, that minimizes the cost of moving samples through the linear program:
min
G2P(l‚ ) ÆG‚ Cæ: (4)
Although this problem can be solved with minimum cost flow solvers, it is usually regularized with entropy for more efficient optimization and empirically better results (Cuturi, 2013). Entropy diffuses the optimal coupling, meaning that more masses will be split. Thus, the numerical optimal transport problem is
min
G2P(l‚ ) ÆG‚ Cæ -  H(G)‚ (5)
where  > 0 and H(G) is the Shannon entropy (Pnx
i=1
Pny
j = 1 Gij log Gij).
6 DEMETCI ET AL.
Downloaded by COLUMBIA UNIVERSITY LIBRARY from www.liebertpub.com at 02/16/25. For personal use only.


Equation (5) is a strictly convex optimization problem, and for some unknown vectors u 2 Rnx and v 2 Rny , the solution has the form G = diag(u)K diag(v)‚ with K = exp - C
e
 , element-wise. This solution can be obtained efficiently via Sinkhorn’s algorithm, which iteratively computes
u)l%Kv and v)%KT u‚ (6)
where % denotes element-wise division. This derivation immediately follows from solving the corresponding dual problem for Equation (5) (Peyre ́ et al., 2019).
2.2. The Gromov–Wasserstein optimal transport
While the classic optimal transport formulation requires us to define a cost function across domains [Eq. (2)], this is difficult to do when working with data from different metric spaces. This is because we cannot directly compare data points with different modalities, such as in the case of multi-omic alignment. The Gromov–Wasserstein distance extends optimal transport by comparing distances between data points rather than directly comparing the data points themselves (Alvarez-Melis and Jaakkola, 2018) and allows us to work with data from different modalities. Consider the same discrete measures l and  as above, the cost function in the formulation of the optimal transport problem will now be defined over sample-wise pairwise distances dx(i‚ k) and dy(j‚ l) in the X and Y data sets, respectively:
GW(l‚ ) : = min
p2P(l‚ )
n Xx
i‚ k
n Xy
j‚ l
L(dx(i‚ k)‚ dy(j‚ l)) p (i‚ j) p(k‚ l): (7)
where L indicates the cost function. The main change from basic optimal transport [Eq. (2)] to the GromovWasserstein optimal transport [Eq. (7)] is that we consider the effect of transporting pairs of samples rather than single samples. Intuitively, L(dx(i‚ k)‚ dy(j‚ l)) captures how transporting xi to yj and xk to yl would distort the original distances between i and k and between xj and xl. This change ensures that the optimal transport plan p will preser

=== Learning single-cell perturbation responses using neural optimal transport (Bunne, Charlotte; Stark, Stefan G.; Gut, Gabriele; del Castillo, Jacobo Sarabia;) ===
Nature Methods | Volume 20 | November 2023 | 1759–1768 1759
nature methods
Article https://doi.org/10.1038/s41592-023-01969-x
Learning single-cell perturbation responses using neural optimal transport
Charlotte Bunne 1,2,9, Stefan G. Stark1,2,3,4,9, Gabriele Gut 5,9, Jacobo Sarabia del Castillo5, Mitch Levesque6, Kjong-Van Lehmann 1,7 , Lucas Pelkmans 5 , Andreas Krause 1,2 & Gunnar Rätsch 1,2,3,4,8
Understanding and predicting molecular responses in single cells upon chemical, genetic or mechanical perturbations is a core question in biology. Obtaining single-cell measurements typically requires the cells to be destroyed. This makes learning heterogeneous perturbation responses challenging as we only observe unpaired distributions of perturbed or non-perturbed cells. Here we leverage the theory of optimal transport and the recent advent of input convex neural architectures to present CellOT, a framework for learning the response of individual cells to a given perturbation by mapping these unpaired distributions. CellOT outperforms current methods at predicting single-cell drug responses, as profiled by scRNA-seq and a multiplexed protein-imaging technology. Further, we illustrate that CellOT generalizes well on unseen settings by (1) predicting the scRNA-seq responses of holdout patients with lupus exposed to interferon-β and patients with glioblastoma to panobinostat; (2) inferring lipopolysaccharide responses across different species; and (3) modeling the hematopoietic developmental trajectories of different subpopulations.
Characterizing and modeling perturbation responses at the single-cell level from non-time-resolved data remains one of biology’s grand challenges. It finds applications in predicting cellular reactions to environmental stress or a patient’s response to drug treatments. Accurate inference of perturbation responses at the single-cell level allows us to understand how and why individual tumor cells evade cancer therapies1. More generally, it deepens the mechanistic understanding of the molecular machinery that determines the respective responses to perturbations. Single-cell responses to genetic or chemical perturbations are highly heterogeneous2 due to multiple factors, including pre-existing variability in the abundance and subcellular organization of messenger RNA and proteins3–6, cellular states7 and the cellular microenvironment8. To effectively predict the drug response of each cell in a population, whether derived from tissue culture or as primary
cells from a patient biopsy, it is thus crucial to incorporate this heterogeneous multivariate subpopulation structure into the analysis. A fundamental difficulty in learning perturbation responses is that cells are usually fixed and stained or chemically destroyed to obtain these measurements. Hence, it is only possible to measure the same cells before or after a perturbation is applied. Therefore, while we do not have access to a set of paired control/perturbed single-cell observations, we do have access to separate sets of single-cell observations from control and perturbed cells, respectively. To subsequently match single cells between conditions and, at the same time, account for cellular heterogeneity is a highly complex pairing problem. Here, we seek to learn a perturbation model that robustly describes the cellular dynamics upon intervention while still accounting for underlying variability across samples. Learning the responses on an
Received: 28 June 2022
Accepted: 23 June 2023
Published online: 28 September 2023
Check for updates
1Department of Computer Science, ETH Zurich, Zürich, Switzerland. 2AI Center, ETH Zurich, Zürich, Switzerland. 3Medical Informatics Unit, University of Zurich Hospital, Zürich, Switzerland. 4Swiss Institute of Bioinformatics, Zurich, Switzerland. 5Department of Molecular Life Sciences, University of Zurich, Zürich, Switzerland. 6Department of Dermatology, University of Zurich Hospital, University of Zurich, Zürich, Switzerland. 7Cancer Research Center Cologne–Essen, Site: Center Integrated Oncology Aachen, Aachen, Germany. 8Department of Biology, ETH Zurich, Zürich, Switzerland. 9These authors contributed equally: Charlotte Bunne, Stefan G. Stark, Gabriele Gut. e-mail: kjlehmann@ukaachen.de; lucas.pelkmans@mls.uzh.ch; krausea@ethz.ch; gunnar.raetsch@inf.ethz.ch


Nature Methods | Volume 20 | November 2023 | 1759–1768 1760
Article https://doi.org/10.1038/s41592-023-01969-x
of each cell from the unperturbed cell population ρc into their perturbed state ρk upon treatment k. Despite originating from different observations, map Tk determines for each cell xi the most likely corresponding cell Tk(xi) in the perturbed population (Fig. 1c). Finding this map then not only allows us to model single-cell trajectories upon perturbation but also to predict the perturbed state of previously unseen control cells. As a result, we can forecast the outcome of a perturbation k by applying the learned map Tk to a new unperturbed population ρ′c (Fig. 1d).
The optimal map Tk aligning the control and perturbed population, which we seek to find, should best describe the incremental changes in the multivariate profile of each cell after applying a perturbation k. Using OT23,24 to recover these maps and unveil single-cell reprogramming trajectories has been proposed as a strong modeling hypothesis in the domain of single-cell biology16,17,25–28. OT problems return the alignment between distributions ρc and ρk corresponding to the minimal overall cost between aligned molecular profiles, thus determining the most likely state of each cell upon perturbation (Fig. 1c). Tk is learned such that its image corresponds to ρk and mass is moved from ρc into ρk according to a principle of minimal effort. As directly parameterizing the OT map Tk
20,21,29 is unstable18, we parameterize the convex potentials of the dual optimal transport problem f and g by input convex neural networks22 and recover the optimal map Tk using the gradient of a convex function gk (∇gk)18. Supplementary Section A.3 provides a more detailed review of optimal transport methods proposed for single-cell biology problems and how our approach deviates from previous methods. To put CellOT’s performance in perspective, we benchmark it against current state-of-the-art methods based on autoencoders12,13, which attempt to add perturbation effects through the manipulation of a learned latent representation (reviewed in Supplementary Section A.1). To further test the hypothesis of the OT modeling prior, we compare the learned OT map ∇gk for each perturbation k with naive non-OT-based alignments.
CellOT outperforms state-of-the-art methods
We apply CellOT to predict the responses of cell populations to cancer treatments using a proteomic dataset consisting of two melanoma cell lines (M130219 and M130429)30, profiled by 4i5 and a single-cell RNA-sequencing (scRNA-seq) dataset31, which contain 34 and 9 different treatments, respectively. For more details on the datasets see Online Methods. We benchmarked CellOT against two autoencoder-based tools, scGEN13 and cAE12, as well as PopAlign32, a method based on aligning subpopulations of the control and treated space approximated through a mixture of Gaussian densities. Due to the high-dimensional nature of scRNA-seq data, we apply CellOT on latent representations learned by an autoencoder. The marginal distributions for observed and predicted cell populations for two 4i treatments and two scRNA-seq treatments are shown in Fig. 2a,d. Two features are selected for each perturbation and the complete set of marginals is shown in Supplementary Figs. 1–4. While the autoencoder baselines tend to capture the mean of the treated cell population, they are less successful in matching all heterogeneous states of the perturbed population (higher moments of the perturbed population). Thus, these models tend to learn over-simplified perturbation effects and are insufficient when aiming to understand heterogeneous rather than average cellular behaviors. CellOT, on the other hand, is able to capture these higher moments, yielding accurate and nuanced predictions. This can be further quantified through distributional metrics such as the maximum mean discrepancy (MMD)33. Low values of MMD imply that all moments of two distributions are matched and thus the entire distribution of perturbed cells is captured in fine detail, beyond the population average (Online Methods provides details). The MMDs between the predicted and observed populations for the
existing patient cohort enables inference of treatment responses for new (previously unseen) patients, assuming that we captured the heterogeneous drug reactions of patients during training. It is crucial, however, to not simply model average perturbation responses of a patient cohort, but to capture the specificities of a single patient through personalized treatment effect predictions. Previous methods to approximate single-cell perturbation responses fall short of solving this highly complex pairing problem while, at the same time, accounting for cellular heterogeneity and the strong subpopulation structure of cell samples9–11. Current state-of-the-art methods12–14 predict perturbation responses via linear shifts in a learned latent space. While this can capture nonlinear cell-type-specific responses, the use of linear interpolations reduces the alignment problem to the possibly more challenging task of learning representations that are invariant to the corresponding perturbation. In this work, we introduce CellOT, a new approach that predicts perturbation responses of single cells by directly learning and uncovering maps between control and perturbed cell states, thus explicitly accounting for heterogeneous subpopulation structures in multiplexed molecular readouts. Assuming perturbations incrementally alter molecular profiles of cells, such as gene expression or signaling activities, we learn these changes and alignments using optimal transportation theory (OT)15. Optimal transport provides natural geometric and mathematical tools to manipulate probability distributions. It has found recent successes modeling cellular development processes16,17, albeit in a non-parameterized setting. Thus, current OT-based approaches are unable to make predictions on unseen cells, such as those from unseen samples, for example from new patients. Based on recent developments in neural optimal transport18, CellOT learns an optimal transport map for each perturbation in a fully parameterized and highly scalable manner. Instead of directly learning a transport map19–21, CellOT parameterizes a pair of dual potentials with input convex neural networks22. This choice induces an important theory-motivated inductive bias essential to model stability18. We demonstrate CellOT’s effectiveness by (1) learning single-cell marker responses to different cancer drugs in melanoma cell lines; (2) predicting single-cell transcriptome responses in biopsies of patients with systemic lupus erythematosus as well as panobinostat treatment outcomes of glioblastoma patients; (3) inferring lipopolysaccharide (LPS) responses across different animal species; and (4) modeling the transcriptome evolution of cell fates in hematopoiesis. Moreover, we benchmark CellOT against current state-of-the-art methods on multiple tasks12,13.
Results
Predicting perturbation responses via optimal transport maps Small molecule drugs can have profound effects on the cellular phenotype by, for instance, altering signaling cascades. Most of these effects depend on the context in which the perturbation occurs. Given the heterogeneity among single cells in cell populations and tissues, predicting cellular responses requires understanding the rules by which context shapes genome activity and its response to drugs. High-dimensional single-cell data measured via single-cell genomics or multiplexed imaging technologies can provide this contextual information but only return unpaired or unaligned observations of cell populations. Here, CellOT allows us to utilize such unpaired data and enables learning cell-state transitions upon perturbation. In formal terms, we denote the unperturbed control population by ρc consisting of n cells xi for i = 1, ..., n. Upon perturbation k, the multivariate state of each cell xi of the unperturbed population changes, which we observe as the perturbed population ρk (Fig. 1a). To understand the mode of action and effect of perturbations, we seek to learn the transition and alignment between populations ρc and ρk via parameterizing a map Tk (see Fig. 1a,b), which explains the transition


Nature Methods | Volume 20 | November 2023 | 1759–1768 1761
Article https://doi.org/10.1038/s41592-023-01969-x
selected perturbations are shown in Fig. 2b,e. For scRNA-seq data, MMD evaluations are computed using the top 50 marker genes. An analysis on the influence of the number of chosen marker genes can be found in Supplementary Fig. 7. In addition to the autoencoder baselines, we include the trivial identity baseline that predicts treatment effects simply by returning the untreated states, as well as a theoretical lower bound, observed, consisting of a different set of observed perturbed cells, thus only varying from the true predictions up to experimental noise. We find that CellOT can approach the lower bound (observed setting), whereas the baseline methods often do not improve much over the identity setting. Different evaluation metrics across all 35 4i therapies and 6 scRNA-seq therapies are summarized in Supplementary Figs. 5 and 6. Besides MMD, we additionally include the l2 mean that measures the distance between the observed and predicted mean drug effect over all features. Lastly, we compare the overall mean correlation coefficient r2 between the predicted and observed data on all features (Online Methods). CellOT outperforms the baselines in both metrics across all treatments, typically by one order of magnitude. We attribute the strong performance of CellOT to its ability to learn a transport function that considers explicitly the data geometries of cell populations through the theory of optimal transport. This hypothesis is supported by the observation that the inter-feature correlation structure remains largely conserved between treated and untreated populations, thus depicting a setting where OT approaches excel. For more information, see Extended Data Fig. 1. Extended Data Fig. 2 visualizes the learned maps, further demonstrating CellOT’s ability to model finegrained responses. Finally, we computed Uniform Manifold Approximation and Projection (UMAP) projections34 on a joint set of predicted and observed perturbed cells utilizing the full feature space (Fig. 2c,f). We observe that the perturbed cell states inferred by CellOT are well integrated with the observed perturbed cells. Again, both baselines do not recover the perturbed distribution in its entirety and thus the perturbed state of different subpopulations is not captured consistently.
CellOT captures cell-to-cell variability in drug responses
Capturing distinct perturbation responses of different cell types within the same sample remains a challenging computational task. To reduce the task’s complexity, prediction algorithms can be guided by predefined cell-type labels both in the perturbed and unperturbed states32 or set to approximate the mean drug response13. These simplifications come at a cost: the reliance on a priori knowledge about present and relevant cell types, the assumption that cell types are characterized by the same features before and after a perturbation and that the drug response is uniform within a cell type. In the worst case, these limitations risk masking true and important drug response heterogeneity and thus hamper the discovery of new cell-type- or cell-state-specific perturbation responses (further comparisons are provided in Supplementary Fig. 13). CellOT is free of these limitations and enables scientists to query the predicted single-cell responses at the granularity best suited to answer their biological questions. As a proof of concept, we co-cultured the aforementioned patient-derived melanoma cell lines (Online Methods) at equal ratios and performed a boutique drug screen, during which we exposed cells for 8 h to a panel of 34 drugs and measured the single-cell drug responses with the 4i technology. Using CellOT, we predict the perturbed cell states of a shared set of control (dimethylsulfoxide (DMSO)-treated) cells (Fig. 3a) for each drug. Previous work7 shows that phosphorylation levels of signaling kinases upon drug treatments are tightly linked to the cellular state. To assess whether this relationship was retained in predicted compared to observed perturbed cells, we analyzed the phosphorylation levels of extracellular signal-regulated kinases (pERK) using the transport maps learned by CellOT on each drug. Using 750 predicted and 750 observed perturbed cells, we computed UMAP projections joint-wise from all features except pERK. Figure 3b shows the predicted and observed population individually annotated with the respective pERK levels of each cell. We found that the spatial organization of the two projections looked almost identical and that pERK levels had a highly comparable distribution across the cells of either class and all drug treatments (further analysis in Extended Data Fig. 3a,b and Online Methods).
Find such that overall cost to transport ρc to ρk is minimal.
ab c
Control
n-dimensional space
Perturbationk
Perturbationl
Perturbationm
d
=
∆
∆
∆
∆
=
∆
Tk
Tk
Tk (xj)
Tk (xi)
gk
xi
xj
xi − Tk (xi)
Control Observed
perturbationk
=
=
=
ρk
ρk
ρk
ρl
ρl
ρm
ρm
+
+
+
Training data
Trained for each perturbation
Apply learned OT maps
Predicted perturbations
Training phase Testing phase
Optimized
OT maps New sample
i
2 2
Arg min Tk#ρc = ρk
ρc
ρc
ρc
ρc’
Tk
Tl
Tm
Tk*
Tl*
Tm*
Tk*
Tl*
Tm*
ρ^k
ρ^l

=== Wasserstein Flow Matching: Generative modeling over families of distributions (Haviv, Doron; Pe'er, Dana; Pooladian, Aram-Alexandre; Amos, Brandon) ===
Wasserstein Flow Matching: Generative modeling over families of distributions
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
Contributions. We introduce Wasserstein Flow Matching (WFM), a principled extension of the FM framework lifted to the space of probability distributions. As illustrated in Figure 1, a single point in our source and target datasets is itself a distribution (e.g., a single discrete measure or a single Gaussian), and our aim is to learn vector fields acting on the space of probability distributions and match the optimal transport map, which is the geodesic in Wasserstein space. WFM is an instantiation of Riemannian FM (Chen and Lipman, 2023), where we train a neural model to learn a continuous normalizing flow (CNF) between distributions over distributions. We demonstrate the effectiveness of our approach for generative modeling between distributions over Gaussian distributions and distributions over point-clouds. The former task is motivated by recent directions in single-cell and spatial transcriptomics (Haviv et al., 2024b; Persad et al., 2023), where we consider matching problems over the Bures–Wasserstein space (BW), the Gaussian submanifold of the Wasserstein space. In this case, we show that WFM can be further modified, resulting in the Bures–Wasserstein FM (BW-FM) algorithm. We validate BW-FM on a variety of
2


Gaussian-based datasets, where we observe that samples generated by our algorithm are significantly more robust than naı ̈ve approaches which do not fully exploit the underlying geometry of the data. In turn, we present a generative model for cell states and niches from single-cell genomics data. Point-cloud generation is made possible by two distinct, yet crucial, algorithmic primitives: (1) incorporating transformers in our neural network architecture (Vaswani, 2017; Lee et al., 2019), and (2) recent algorithmic advances in entropic optimal transport (Pooladian and Niles-Weed, 2021). Indeed, our WFM algorithm performs generative modeling in the Wasserstein space, where geodesics are given by pushforwards of optimal transport (OT) maps; see Section 2.3 for more information. Both the transformer architecture and entropic optimal transport are crucial to approximating the OT map between independent point-clouds. Indeed, the permutation equivariance of attention makes the transformer a natural basis for our model, inherently modeling the equivariance feature of the Wasserstein geometry while maintaining scalability in high-dimensions. For datasets of 3D point-clouds with uniform sizes, the performance of WFM is comparable to other current generative models. However, due to their particular training paradigms (namely the voxelization of 3D spaces), contemporary approaches cannot scale to high-dimensional point-clouds and fail on datasets with variable sized examples. Conversely, WFM succeeds in the high-dimensional and inhomogeneous settings, unlocking generative modeling to new, previously uncharted domains such as synthesizing niches from spatial genomics data. The ability to model tissue biology in this generative manner could enhance our understanding of how environment is associated with cell state. In the context of many diseases, most notably cancer and its tumor-immune microenvironment, these insights are critical for developing novel therapeutics (Binnewies et al., 2018).
2 Background and related work
We let P2(Rd) denote the set of probability distributions over Rd with finite second moment, and write P2,ac(Rd) to be those with densities. For a probability measure μ and (vector-valued) function f , we interchangeably write R ∥f (x)∥2 dμ(x) and ∥f ∥2
L2(μ). Let M be a Riemannian manifold, with
P(M) defining the space of probability measures over said manifold. For x ∈ M, we write TxM to mean the tangent space of the manifold at x, and write the metric on the tangent space (at x) as g(x). For x0 ∈ M with initial velocity v ∈ Tx0M, the terminal location of the resulting geodesic is expressed as the output of the exponential map v 7→ expx0(v) ∈ M. Similarly, for an initial point x0 and terminal location x1, the logarithmic map defines the tangent vector, denoted x1 7→ logx0(x1), such that expx0(logx0(x1)) = x1. The set of symmetric matrices (resp. positive
definite matrices) over Rd are denoted by Sd (resp. Sd
++).
2.1 Riemannian flow matching
We first briefly discuss the Riemannian flow matching (RFM) framework of Chen and Lipman (2023). Let p0 be the source distribution and p1 be the target distribution over a Riemannian manifold M, and let (γt)t∈[0,1] be a curve of probability measures satisfying γ0 = p0 and γ1 = p1. Letting (wt)t∈[0,1] denote a family of vector fields, we say that the pair (γt, wt)t∈[0,1] satisfy the continuity equation with respect to the metric g, abbreviated to (γt, wt) ∈ Cg if
∂tγt + ∇g ·(γtwt) = 0 , (1)
where ∇g· is the Riemannian divergence operator.
3


The goal of RFM is to regress a parameterized vector field (e.g., a neural network), written fθ(x, t) ∈ TxM for t ∈ [0, 1], onto the family wt by minimizing
mθin
Z1
0
Z
∥fθ(zt, t) − wt(zt)∥2
g(zt) dγt(zt) dt ,
assuming access to a pair (γt, wt)t∈[0,1] that satisfies equation 1, which is not possible in many scenarios. Borrowing insights from recent work (e.g., Albergo and Vanden-Eijnden (2022); Lipman et al. (2022); Liu et al. (2022)), the authors construct a simple vector field that satisfies the continuity equation, resulting in the tractable objective
mθin
Z1
0
ZZ
∥fθ(xt, t) − x ̇ t∥2
g(xt) dp0(x) dp1(y) dt , (2)
where, for example, xt = expx((1 − t) logx(y)) ∈ M, and x ̇ t ∈ TxtM. For complete discussions and proofs, see (Chen and Lipman, 2023). Once fθ is appropriately fit using equation 2, we can generate
new samples from p1: start by sampling X0 ∼ p0, then follow X ̇ t = fθ(Xt, t) numerically by discretizing the dynamics given by the exponential map, resulting in X1 ∼ p1. We emphasize that the dynamics are only simulated at inference time and not when training fθ, which is commonly known as a simulation-free training paradigm.
2.2 Related work
Generative models for point-clouds. Paralleling the progress in generative models for natural images, the field of point-cloud generation is rapidly expanding. Many different models have been used from this task, namely generative-adversarial-nets (Achlioptas et al., 2018), variational autoencoders (Gadelha et al., 2018), normalizing flows (Yang et al., 2019; Kim et al., 2020; Klokov et al., 2020), diffusion (Zhou et al., 2021; Cai et al., 2020) and even euclidean FM (Wu et al., 2023). Thus far, these approaches are limited to uniformly sized point-clouds in 2D & 3D, and fail on high-dimensional spaces which cannot be voxelized.
Generative models over families of distributions. Our work is not the first to instantiate Riemannian FM with a manifold of probability measures. Two notable works are Fisher FM (Davis et al., 2024) and Categorical FM (Cheng et al., 2024), which consider the FM algorithm with respect to the Fisher–Rao geometry Amari (2016); Nielsen (2020) over the d-dimensional simplex ∆d. The work of Stark et al. (2024) is similar in spirit, where they focus on the Dirichlet distribution for generation of discrete data. Another related work is that of Atanackovic et al. (2024), called Meta FM. Their approach requires pairs of distributions which are already coupled, with the goal of solving FM between a distribution over pairs. In contrast, we emphasize that our proposed Wasserstein FM applies between two separate uncoupled distributions over distributions.
2.3 Wasserstein geometry
The (squared) 2-Wasserstein distance between two probability measures μ, ν ∈ P2,ac(Rd) is given by the non-convex optimization problem over vector-valued maps T : Rd → Rd
W2
2 (μ, ν) := min
T :T♯μ=ν ∥id − T ∥2
L2(μ) , (3)
4


where the pushforward constraint, written T♯μ = ν, means that, for X ∼ μ, the image follows T (X) ∼ ν. The minimizer to equation 3 is called the optimal transport (OT) map, denoted T μ→ν
⋆
(we abbreviate this to T⋆ when it is clear from context). The existence and uniqueness of the optimal transport map under the stated regularity conditions is due to Brenier (1991). The Wasserstein space is the space of probability densities with finite second moment endowed with the Wasserstein distance; this space is known to be a metric space (Villani, 2009). Following the celebrated work of Otto (2001), the Wasserstein space can be formally (meaning, non-rigorously) viewed as a Riemannian manifold, whose properties we now describe in brief; see e.g., Ambrosio et al. (2008) for a rigorous treatment. Following the definition in Theorem 8.5.1 from Ambrosio et al. (2008), the tangent space at a point μ ∈ P2,ac(Rd) consists of all possible tangent vectors that emanate from μ, written formally as
TμP2,ac(Rd) := {λ(T μ→ν
⋆ − id) : λ > 0, ν ∈ P2(Rd)}L2(μ) ,
where the overline denotes the closure of the set in L2(μ), and the norm on the tangent space is also L2(μ). The exponential and logarithmic maps read
v 7→ expμ(v) := (id + v)♯μ , ν 7→ logμ(ν) := T μ→ν
⋆ − id ,
where id is the identity map. Consequently, the (constant-speed) geodesic, or McCann interpolation, between two measures μ and ν is given by the curve (μt)t∈[0,1] where
μt := (T μ→ν
t )♯μ := ((1 − t)id + tT μ→ν
⋆ )♯μ ≡ expμ((1 − t) logμ(ν)) , (4)
where the last expression writes the pushforward in terms of the exponential and logarithmic maps. Equivalently, at the level of the random variables, one can write Xt = (1 − t)X0 + tT μ→ν
⋆ (X0), where X0 ∼ μ and Xt ∼ μt for any t ∈ [0, 1]. Combined with (vt)t∈[0,1] a suitable family of vector fields, the McCann interpolation satisfies the continuity equation equation 1 over Rd, re-written as
∂tμt + ∇ · (μtvt) = 0 , s.t. μ0 = μ , μ1 = ν , (5)
where the divergence operator is the usual Euclidean one over Rd, thus we write (μt, vt) ∈ C. The link between the constant speed geodesics and the 2-Wasserstein distance can be viewed from the celebrated Benamou–Brenier formulation of optimal transport (Benamou and Brenier, 2000):
W2
2 (μ, ν) = inf
(μt,vt)∈C
Z1
0
∥vt∥2
L2(μt) dt . (6)
The optimal curve of measures is given by the constant-speed geodesics described above, and the optimal velocity field is given by
vt = (T μ→ν
⋆ − id) ◦ (T μ→ν
t )−1 . (7)
The vector field equation 7 should be interpreted as the time-derivative of the McCann interpolation:
X ̇ t = (T μ→ν
⋆ − id)(X0) = (T μ→ν
⋆ − id) ◦ (T μ→ν
t )−1(Xt) , X0 ∼ μ .
5


2.3.1 Bures–Wasserstein (BW) space
A known special case of the Wasserstein space is the Bures–Wasserstein space, which consists of the submanifold of non-degenerate Gaussians parameterized by means and covariances {(m, Σ) : m ∈ Rd, Σ ∈ Sd
++}, endowed with the Wasserstein metric. We provide a brief exposition on the geometry of the Bures–Wasserstein space and refer the interested reader to Lambert et al. (2022) for detailed calculations and explanations, as we follow their notation conventions. The OT map between μ = N (mμ, Σμ) and ν = N (mν, Σν) has a closed-form (Gelbrich, 1990):
T⋆(x) := mν + Cμ→ν(x − mμ) := b + Σ− 1
μ 2 (Σ
1
μ2 Σν Σ
1
μ2 )
1
2 Σ− 1
μ 2 (x − mμ) .
As this map is affine, it is clear that the McCann interpolation between two Gaussians is always Gaussian (indeed, Gaussians undergoing affine transformations remain Gaussian). More generally, we have the succinct representation of the tangent space at a point in the Bures–Wasserstein space
TμBW(Rd) := {a + S(id − mμ) : a ∈ Rd, S ∈ Sd} ,
and the exponential and logarithmic maps between two non-degenerate Gaussians are
(a, S) 7→ expμ((a, S)) := N (mμ + a, (S + I)Σμ(S + I)) ,
ν 7→ logμ(ν) := (mν − mμ, Σ− 1
μ 2 (Σ
1
μ2 Σν Σ
1
μ2 )
1
2 Σ− 1
μ 2 − I),
where the exponential map requires S ≻ −I. We also mention that the norm on the tangent space at μ in the Bures–Wasserstein space can be written as
∥(a, S)∥2
BW(μ) := ∥a − mμ∥2 + Tr(S2Σμ)
With the above, it is easy to compute the closed-form solutions for the mean and covariance of the McCann interpolation μt = (Tt)♯μ = N (mt, Σt), given by
mt := (1 − t)a + tb , Σt := TtATt := ((1 − t)I + tCA→B)A((1 − t)I + tCA→B) . (8)
We can relate the Euclidean and Riemannian time-derivatives of Σt through the following manipulation (the latter of which respects the exponential and logarithmic maps above):
 ̇ΣE
t = T ̇tATt + TtAT ̇t = T ̇t(Tt)−1TtATt + TtATt(Tt)−1T ̇t =  ̇ΣBW
t Σt + Σt  ̇ΣBW
t.
To this end, we can draw parallels to equation 7 by writing
m ̇ t = b − a , Σ ̇ BW
t = (CA→B − I)((1 − t)I + tCA→B)−1 . (9)
3 Flow matching over the Wasserstein space
3.1 Training
Let p0 and p1 denote probability measures over the Wasserstein space.∗ Our goal is to learn a vector field that transports the family of measures p0 to the family p1. To accomplish this, we
6


Figure 2: When the number of training examples is too few, all methods collapse on the tra

=== Stochastic Interpolants: A Unifying Framework for Flows and Diffusions (Albergo, Michael S.; Boffi, Nicholas M.; Vanden-Eijnden, Eric) ===
Stochastic Interpolants: A Unifying Framework for Flows and Diffusions
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
C Experimental Specifications 64 C.1 Image Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
2


1 Introduction
1.1 Background and motivation
Dynamical approaches for deterministic and stochastic transport have become a central theme in contemporary generative modeling research. At the heart of progress is the idea to use ordinary or stochastic differential equations (ODEs/SDEs) to continuously transform samples from a base probability density function (PDF) ρ0 into samples from a target density ρ1 (or vice-versa), and the realization that inference over the velocity field in these equations can be formulated as an empirical risk minimization problem over a parametric class of functions [24, 58, 25, 60, 5, 2, 41, 39].
A major milestone was the introduction of score-based diffusion methods (SBDM) [60], which map an arbitrary density into a standard Gaussian by passing samples through an Ornstein-Uhlenbeck (OU) process. The key insight of SBDM is that this process can be reversed by introducing a backwards SDE whose drift coefficient depends on the score of the time-dependent density of the process. By learning this score – which can be done by minimization of a quadratic objective function known as the denoising loss [68] – the backwards SDE can be used as a generative model that maps Gaussian noise into data from the target. Though theoretically exact, the mapping takes infinite time in both directions, and hence must be truncated in practice.
While diffusion-based methods have become state-of-the-art for tasks such as image generation, there remains considerable interest in developing methods that bridge two arbitrary densities (rather than requiring one to be Gaussian), that accomplish the transport exactly, and that do so on a finite time interval. Moreover, while the highest quality results from score-based diffusion were originally obtained using SDEs [60], this has been challenged by recent works that find equivalent or better performance with ODE-based methods if the score is learned sufficiently well [32]. If made to match the performance of their stochastic counterparts, ODE-based methods exhibit a number of desirable characteristics, such as an exact, computationally tractable formula for the likelihood and the easy application of well-developed adaptive integration schemes for sampling. It is an open question of significant practical importance to understand if there exists a separation in sample quality between generative models based on deterministic dynamics and those based on stochastic dynamics.
In order to satisfy the desirable characteristics outlined in the previous paragraph, we develop a framework for generative modeling based on the method proposed in [2], which is built on the notion of a stochastic interpolant xt used to bridge two arbitrary densities ρ0 and ρ1. We will consider more general designs below, but as one example the reader can keep in mind:
xt = (1 − t)x0 + tx1 + p2t(1 − t)z, t ∈ [0, 1], (1.1)
where x0, x1, and z are random variables drawn independently from ρ0, ρ1, and the standard Gaussian density N(0, Id), respectively. The stochastic interpolant xt defined in (1.1) is a continuoustime stochastic process that, by construction, satisfies xt=0 = x0 ∼ ρ0 and xt=1 = x1 ∼ ρ1. Its paths therefore exactly bridge between samples from ρ0 at t = 0 and from ρ1 at t = 1. A key observation is that:
The law of the interpolant xt at any time t ∈ [0, 1] can be realized by many different processes, including an ODE and forward and backward SDEs whose drifts can be learned from data.
To see why this is the case, one must consider the probability distribution of the interpolant xt. As
shown below, for a large class of densities ρ0 and ρ1 supported on Rd, this distribution is absolutely continuous with respect to the Lebesgue measure. Moreover, its time-dependent density ρ(t) satisfies a first-order transport equation and a family of forward and backward Fokker-Planck equations in which the diffusion coefficient can be varied at will. Out of these equations, we can readily derive generative models that satisfy ODEs and SDEs, respectively, and whose densities at time t are given by ρ(t).
3


March 13, 2023 1
Ω(t)
v s
t i m e
Ω0
Ω1
ODE
Ω(t)
v s
t i m e
Ω0
Ω1
SDE
Figure 1: The stochastic interpolant paradigm. Example generative models based on the proposed framework, which connects two densities ρ0 and ρ1 using samples from both. The design of the time-dependent probability density ρ(t) that bridges between ρ0 and ρ1 is separated from the choice of how to sample it, which can be accomplished with deterministic or stochastic generative models. Left panel: Sampling with a deterministic (ODE) generative model known as the probability flow equation. Right panel: Sampling with a stochastic generative model given by an SDE with a tunable diffusion coefficient. The probability flow equation and the SDE have different paths, but their time-dependent density ρ(t) is the same. Moreover, the two equations rely on the same estimates for the velocity and the score.
Interestingly, the drift coefficients entering these ODEs/SDEs are the unique minimizers of quadratic objective functions that can be estimated empirically using data from ρ0, ρ1, and N(0, Id). The resulting least-squares regression problem allows us to estimate the drift coefficients of the ODE/SDEs, which can then be used to push samples from ρ0 onto new samples from ρ1 and vice-versa.
1.2 Main contributions and organization
The approach introduced here is a versatile way to build generative models that unifies and extends many existing algorithms. In Sec. 2, we develop the framework in full generality, where we emphasize the following key contributions:
• We prove that the stochastic interpolant defined in Section 2.1 has a distribution that is absolutely continuous with respect to the Lebesgue measure on Rd, and that its density ρ(t) satisfies a first-order transport equation (TE) as well as a family of forward and backward Fokker-Planck equations (FPEs) with tunable diffusion coefficients.
• We show how the stochastic interpolant can be used to learn the drift coefficients that enter the TE and the FPEs. We characterize these coefficients as the minimizers of simple quadratic objective functions given in Section 2.2. We introduce a new objective for the score ∇ log ρ(t) of the interpolant density, as well as an objective function for learning a denoiser ηz, which we relate to the score.
• In Section 2.3, we derive ordinary and stochastic differential equations associated with the TE and FPEs that lead to deterministic and stochastic generative models. In Section 2.4, we show that regressing the drift for SDE-based models controls the likelihood, but that regressing the drift alone is not sufficient for ODE-based models, which must also minimize a Fisher divergence. We show how to optimally tune the diffusion coefficient to maximize the likelihood for SDEs.
• In Section 2.5, we develop a general formula to evaluate the likelihood of SDE-based generative models that serves as a natural counterpart to the continuous change-of-variables formula commonly used to compute the likelihood of ODE-based models. In addition, we give formulas to
4


estimate the cross-entropy.
In Section 3, we discuss instantiations of the stochastic interpolant method. In Section 3.4 we first show that interpolants are equivalent to a class of stochastic bridges, but that they avoid the need for Doob’s h-transform, which is generically unknown; we show that this simplifies the construction of a broad class of generative models. In Section 3.2, we define the one-sided interpolant, which corresponds to the conventional setting in which the base ρ0 is taken to be a Gaussian. With a Gaussian base, several aspects of the interpolant simplify, and we detail the corresponding objective functions. In Section 3.3, we introduce a mirror interpolant in which the base ρ0 and the target ρ1 are identical. Finally, in Section 3.4, we show how the interpolant framework leads to a natural formulation of the Schr  ̈odinger bridge problem between two densities.
In Section 4, we discuss a special case in which the interpolant is spatially linear in x0 and x1. In this case, the velocity field can be factorized, which we show in Section 4.1 leads to a simpler learning problem. We detail specific choices of linear interpolants in Section 4.2, and in Section 4.3 we illustrate how these choices influence the performance of the resulting generative model, with a particular focus on the role of the latent variable and the diffusion coefficient. For exposition, we focus on Gaussian mixture densities, for which the drift coefficients can be computed analytically. We provide the resulting formula in Appendix A. Finally, in Section 4.4, we discuss the case of spatially linear one-sided interpolants.
In Section 5, we formalize the connection between stochastic interpolants and related classes of generative models. In Section 5.1, we show that score-based diffusion models can be re-written as one-sided interpolants after a reparameterization of time; we highlight how this approach eliminates singularities that appear when naively compressing score-based diffusion onto a finite-time interval. In Section 5.2, we show how interpolants can be used to derive the Bayes-optimal estimator for a denoiser, and we show how this approach can be iterated to create a generative model. In Section 5.3, we consider the possibility of rectifying the flow map of a learned generative model. We show that the rectification procedure does not change the underlying generative model, though it may change the time-dependent density of the interpolant.
In Section 6, we provide the details of practical algorithms associated with the mathematical results presented above. In Section 6.1, we describe how to numerically estimate the objectives given empirical datasets from the base and the target. In Section 6.2, we complement this discussion on learning with algorithms for sampling with the ODE or an SDE.
We provide numerical demonstrations in line with these recommendations in Section 7, and we conclude with some remarks in Section. 8.
1.3 Related work
Deterministic Transport and Normalizing Flows. Transport-based sampling and density estimation has its contemporary roots in Gaussianizing data via maximum entropy methods [23, 12, 64, 63]. The change of measure under such transformation is the backbone of normalizing flow models. The first neural network realizations of these methods arose through imposing clever structure on the transformation to make the change of measure tractable in discrete, sequential steps [52, 16, 50, 28, 19]. A continuous time version of this procedure was made possible by viewing the map T = Xt(x) as the solution of an ODE [11, 24], whose parametric drift defining the transport is learned via maximum likelihood estimation. Training this way is intractable at scale, as it requires simulating the ODE. Various methods have introduced regularization on the path taken between the two densities to make the ODE solves more efficient [22, 48, 65], but the fundamental difficulty remains. We also work in continuous time; however, our approach allows us to learn the drift without simulation of the dynamics, and can be formulated at sample generation time through either deterministic or
5


stochastic transport.
Stochastic Transport and Score-Based Diffusions (SBDMs). Complementary to approaches based on deterministic maps, recent works have realized that connecting a data distribution to a Gaussian density can be viewed as the evolution of an Ornstein-Ulhenbeck (OU) process which gradually degrades samples from the distribution of interest to Gaussian noise [54, 25, 58, 60]. The OU process specifies a path in the space of probability densities; this path is simple to traverse in the forward direction by addition of noise, and can be reve

=== Gene trajectory inference for single-cell data by optimal transport metrics (Qu, Rihao; Cheng, Xiuyuan; Sefik, Esen; Stanley III, Jay S.; Landa, Boris; Strin) ===
Nature Biotechnology | Volume 43 | February 2025 | 258–268 258
nature biotechnology
Article https://doi.org/10.1038/s41587-024-02186-3
Gene trajectory inference for single-cell data by optimal transport metrics
Rihao Qu1,2,3,11, Xiuyuan Cheng4,11, Esen Sefik 3, Jay S. Stanley III5, Boris Landa5, Francesco Strino 6, Sarah Platt2,7, James Garritano5, Ian D. Odell3,7, Ronald Coifman5,8,9, Richard A. Flavell 3,10,12, Peggy Myung 2,7,12 & Yuval Kluger 1,2,5,12
Single-cell RNA sequencing has been widely used to investigate cell state transitions and gene dynamics of biological processes. Current strategies to infer the sequential dynamics of genes in a process typically rely on constructing cell pseudotime through cell trajectory inference. However, the presence of concurrent gene processes in the same group of cells and technical noise can obscure the true progression of the processes studied. To address this challenge, we present GeneTrajectory, an approach that identifies trajectories of genes rather than trajectories of cells. Specifically, optimal transport distances are calculated between gene distributions across the cell–cell graph to extract gene programs and define their gene pseudotemporal order. Here we demonstrate that GeneTrajectory accurately extracts progressive gene dynamics in myeloid lineage maturation. Moreover, we show that GeneTrajectory deconvolves key gene programs underlying mouse skin hair follicle dermal condensate differentiation that could not be resolved by cell trajectory approaches. GeneTrajectory facilitates the discovery of gene programs that control the changes and activities of biological processes.
Dynamic gene expression changes often specify mechanisms through which cells determine state and function. Indeed, tightly regulated gene cascades underlie a myriad of fundamental processes, such as cell cycle (CC)/mitosis1–4 and tissue/organ differentiation5–8. With the emergence of single-cell RNA-sequencing (scRNA-seq) platforms, cell trajectory inference techniques9–19 are widely applied to study the cellular dynamics of biological processes. These techniques use single-cell whole-transcriptome data to organize cells into lineages and infer a unidimensional latent variable (that is, pseudotime20) that describes a cell’s position along a lineage process. After pseudotime
construction, gene dynamics underlying a biological process can be inferred by tracking the changing patterns of their expression levels along the cell pseudotime12,15,21. However, when cells undergo multiple processes in parallel (for example, CC coupled with cell differentiation22 or circadian clock23) and each process is governed by a different set of genes, cell pseudotime learned by organizing cells using the collective genes becomes less informative, as it mixes the effects of multiple processes. Mathematically, when multiple processes that are not strongly correlated with each other co-occur in the same group of cells, cell geometry
Received: 19 December 2022
Accepted: 26 February 2024
Published online: 5 April 2024
Check for updates
1Computational Biology & Bioinformatics Program, Yale University, New Haven, CT, USA. 2Department of Pathology, Yale University School of Medicine, New Haven, CT, USA. 3Department of Immunobiology, Yale University School of Medicine, New Haven, CT, USA. 4Department of Mathematics, Duke University, Durham, NC, USA. 5Program in Applied Mathematics, Yale University, New Haven, CT, USA. 6PCMGF Limited, Watford, UK. 7Department of Dermatology, Yale University School of Medicine, New Haven, CT, USA. 8Department of Mathematics, Yale University, New Haven, CT, USA. 9Department of Electrical Engineering, Yale University, New Haven, CT, USA. 10Howard Hughes Medical Institute, Yale University School of Medicine, New Haven, CT, USA. 11These authors contributed equally: Rihao Qu, Xiuyuan Cheng. 12These authors jointly supervised this work: Richard A. Flavell, Peggy Myung, Yuval Kluger. e-mail: yuval.kluger@yale.edu


Nature Biotechnology | Volume 43 | February 2025 | 258–268 259
Article https://doi.org/10.1038/s41587-024-02186-3
populations. In our work, we distinctively define the graph-based Wasserstein distance between pairs of genes to study their underlying pseudotemporal dynamics. Specifically, we normalize the expression of a gene into a probabilistic distribution over cells and then compute the Wasserstein distances between gene distributions in the cell graph (Fig. 1d). Here the cell graph is constructed in a way that provides a representation of cells, which preserves the cell manifold structure in the high-dimensional space (Fig. 1c). In this construction, the graph-based Wasserstein distance between pairwise gene distributions has the following characteristics: (1) it takes into account the geometry of cells; that is, it assigns a higher cost to transport a point mass from one cell to a distant cell as compared to its adjacent neighbors. (2) It prevents the transport across the ambient cell space, which is often a problematic issue when using spatial distance measures (for example, the Euclidean distance in the cell space). In our approach, the computation of gene–gene Wasserstein distances is based on the following two steps (Table 1):
• Construct a cell graph. As an initial step, we learn a reduceddimensional cell embedding that can capture and represent the cell manifold structure in the original high-dimensional space. Next, we construct a k-nearest neighbor (kNN) graph of cells based on their relative distances in the cell embedding (Fig. 1c). This establishes a cell–cell connectivity map that serves as the ‘roadmap’ for transporting gene distributions in the next step. Here, for a given pair of cells u and v, we search for the shortest path connecting them in the kNN cell graph and denote its length as the graph distance dG(u, v) between cells u and v. This graph distance will be used to define the cost of transporting a point mass between cells u and v in the next step. • Compute gene–gene Wasserstein distances over the cell graph. We model the expression level of genes as discrete distributions on the cell graph. Specifically, we divide the original expression level of a given gene in each cell by the sum of its expression level in all cells. We then define the distance between two gene distributions by the graph-based Wasserstein-p distance (Wp distance, 1 ≤ p < ∞; Fig. 1c,d). Accordingly, the transport cost between cells
u and v is defined as Cu,v = dG(u, v)p. Here p is a user-defined parameter, and p = 1 refers to the well-known Earth Mover’s distance. Algorithmic details are described in ‘Step 2. Compute graph-based Wasserstein distances between genes’.
In practice, computing the Wasserstein distance between all pairwise gene distributions can be computationally expensive. When the cell graph is large, the time cost for finding the OT solution increases exponentially. In our framework, we have designed two strategies to accelerate the computation based on (1) cell graph coarse-graining, and (2) gene graph sparsification (details in ‘Step 2. Compute graph-based Wasserstein distances between genes’).
Gene trajectory construction
The gene–gene Wasserstein distance captures the pseudotemporal relations of genes in the sense that if two genes are activated consecutively along a biological process, their distributions are expected to have a substantial overlap in the cell graph and thus have a small Wasserstein distance between each other (Fig. 1e). To visualize the geometry of all genes, we convert pairwise gene–gene Wasserstein distances into gene–gene affinities and use diffusion map to get a low-dimensional representation of genes. If dynamical cascades of gene activation and deactivation exist in the data, viewing the gene embedding by a combination of leading diffusion map eigenvectors delineates trajectories of genes (Fig. 1f). Each trajectory is linked with a specific gene program that dictates the underlying biological process. In our approach, the extraction of gene trajectories is performed in a sequential manner (Fig. 1g). To identify the first trajectory, we search
(determined by these processes) cannot be effectively parametrized by a common single latent variable. Hence, organizing cells into unidimensional lineages is no longer appropriate. To address this challenge, we propose GeneTrajectory, an approach to studying dynamic processes that does not rely on unidimensional parameterization of the cell manifold. GeneTrajectory allows us to deconvolve multiple, independent processes with sequential gene dynamics. In contrast to cell trajectory approaches, GeneTrajectory constructs trajectories of genes rather than trajectories of cells. Our algorithm dissects out gene programs from the whole transcriptome, eliminating the need for initial cell trajectory construction or the specification of the initial and terminal cell states for each process. Using this method, genes that sequentially contribute to a given biological process can be extracted and organized into a respective gene trajectory that reveals the successive order of gene activity. In this work, we begin by showing GeneTrajectory’s efficacy for unraveling gene dynamics through simulation experiments and application to a human myeloid lineage dataset. Subsequently, we use our approach on a mouse embryonic skin dataset to demonstrate that GeneTrajectory can resolve critical cell state transitions during the early-stage development of hair follicles5,24. Our results indicate that GeneTrajectory extracts gene geometry without the need for constructing cell pseudotime, revealing independent trajectories of concurrent processes that are otherwise obscured by cell pseudotime approaches.
Results
Computing optimal transport between genes over the cell graph
A progressive dynamic biological process is usually governed by a finely regulated gene cascade25–27, in which genes are activated and deactivated in a temporal order along the process, dictating the transcriptomic changes of underlying cell states. Moreover, cells can participate in multiple processes simultaneously, either in a dependent or independent manner. For instance, we illustrate two contrasting scenarios by considering the concurrence of a linear process (for example, differentiation) and a cyclic process (for example, CC; Fig. 1a). When these two processes are strictly dependent on each other, they can be parameterized by a common latent variable and result in a one-dimensional cell curve. In this scenario, it is straightforward to assign a meaningful pseudotime for the cells by ordering them along the curve. However, deconvolving genes into two processes and retrieving their pseudotemporal order in each process is not immediately apparent, which requires additional postprocessing (for example, clustering gene dynamics along the cell pseudotime12). In contrast, when these two processes are independent, cells fall into a manifold (as a Cartesian product of these two processes) with an intrinsic dimension >1. These processes do not share a common latent variable, thus gene dynamics inference based on unidimensional interpolation along the cell–cell manifold is no longer appropriate. In practice, the weak and stochastic nature of the dependency between concurrent biological processes can complicate the extraction of the cell path and the construction of cell pseudotime. Here we present GeneTrajectory, an approach to inferring gene processes through learning the gene–gene geometry without one-dimensional parameterization of the cell manifold (Fig. 1b). Specifically, GeneTrajectory quantifies the distance of genes based on their expression distributions over a cell graph using optimal transport (OT) metrics (Fig. 1d). Previously, OT metrics (for example, Wasserstein distance) have been applied in a wide range of scenarios in single-cell analysis, including (1) defining a distance measure between cells28,29 or cell populations30, (2) constructing cell trajectories31,32, (3) spatial reconstruction of single-cell transcriptome profiles33,34 and (4) multi-omics data integration35. In these works, the dissimilarity was quantified either between a pair of cells or between a pair of cell


Nature Biotechnology | Volume 43 | February 2025 | 258–268 260
Article https://doi.org/10.1038/s41587-024-02186-3
for the gene that has the largest distance from the origin of diffusion map embedding, which serves as the terminus of the first gene trajectory. To retrieve the other genes along the first trajectory, we take that terminus gene as the starting point of a diffusion process. Specifically, we assign a unit point mass to that gene and then diffuse the mass to the other genes. As the probability mass propagates along the gene trajectory from its terminus, the trajectory can be retrieved by a heuristic thresholding procedure (‘Step 3. Construct gene trajectories’). After retrieving genes for the first trajectory, we identify the terminus of the subsequent gene trajectory among the remaining genes and iterate the same procedure, until all detectable gene trajectories are extracted (Fig. 1g,h). To order the genes along a given trajectory, we retain only these genes to recompute a diffusion map embedding based on their pairwise gene–gene Wasserstein distances. The obtained first nontrivial
eigenvector of the diffusion map embedding provides an intrinsic ordering of the genes along that trajectory36,37. To examine how the gene order along a given gene trajectory is reflected over the cell graph, we can track how these genes are expressed across different regions in the cell embedding. Specifically, we first group genes along each gene trajectory into successive bins and generate a cell embedding ‘snapshot’ for each bin. In each snapshot, we color the cells according to the fraction of genes (from that bin) that they express. By plotting the expression level of each gene bin on the cell embedding, we can visualize how the underlying biological process progresses across cell populations.
Assessing GeneTrajectory’s performance using simulation
Assuming that a progressive biological process is temporally dictated by a sequence of genes, we simulated several artificial scRNA-seq
(Dependent)
(Intrinsic dimension = 1) (Intrinsic dimension = 2)
Cell geometry
Cell geometry
Cell pseudotime Cell pseudotime (linear process)
Cell pseudotime (cyclic process)
Scenario A: Scenario B:
Process diagram
Linear
Cyclic
(Independent)
Process diagram
Linear
Cyclic
Trajectory identification and gene ordering
Gene
Cell
Cell
Cell
Cell graph Gene
Gene
Gene graph Gene trajectories
Count Distance OT
distance
g1
Gene dynamics
g2 g3
g4
Gene–gene OT distance matrix (submatrix example)
Gene–gene OT distances (over the cell graph)
Gene expression profiles
Gene affinity graph
b
g1
g2
g3
g4
g1 g2 g3 g4 g1
g2
g3
g4
d
e fg
Visualization
h
Trajectory 1 Trajectory 2 Trajectory 3
Terminus 1
Terminus 2 Terminus 3
Gene pseudo-order (normalized) 0 1
Gene 1 Gene 2 g1
g2
Step 1: cell graph construction Step 2: gene–gene dist. computation Step 3: gene trajectory inference Step 4: gene ordering
a
0T 0 0
Gene 1 (g1)
Gene 2 (g2)
Gene 3 (g3)
Gene 4 (g4)
g1
g2
g4
g3
TL TC
Cell kNN graph
c
Transcriptome space (high-dimensional)
Cell cloud
10 8 6 4 2 0
0.9 0.6 0.3 0
1.5 1.0 0.5 0
3 2 1 0
1.5 1.0 0.5 0
Fig. 1 | Overview of GeneTrajectory. a, Illustration of two scenarios when a
linear process and a cyclic process are dependent or independent of each other, resulting in cell manifolds with different intrinsic dimensions and requiring distinct pseudotime parametrizations. b, Schematic representation of the major workflow of GeneTrajectory. c, Construction of cell kNN graph. d, Computation of graph-based OT (Wasserstein) distances between paired gene distributions (four representative genes are shown) over the cell graph. Gene distributions are defined by their normalized expression levels over cells. e, Heatmap of OT (Wasserstein) distances for genes g1–g4 in d. f, Construction of gene graph based
on gene–gene affinities (transformed from gene–gene Wasserstein distances). g, Sequential identification of gene trajectories using a diffusion-based strategy. The initial node (terminus 1) is defined by the gene with the largest distance from the origin in the diffusion map embedding. A random-walk procedure is then used on the gene graph to select the other genes that belong to the trajectory terminated at terminus 1. After retrieving genes for the first trajectory, we identify the terminus of the subsequent gene trajectory among the remaining genes and repeat the steps above. This is done iteratively until all detectable trajectories are extracted. h, Diffusion map visualization of gene trajectories.


Nature Biotechnology | Volume 43 | February 2025 | 258–268 261
Article https://doi.org/10.1038/s41587-024-02186-3
datasets with a variety of gene dynamics by modeling the change of gene expression over time (Extended Data Fig. 1a,b; ‘Workflow of gene dynamics simulation’). Specifically, for a gene involved in a given biological process, we simulate its expected expression level λ(t) as a function of time t. For clarity, we note that t represents the pseudotime of a biological process, linked with the cell state (for example, differentiation status) rather than the actual time (for example, specific day of a developmental process). Here we use multiple parameters to account for the heterogeneity of gene expression profiles in single-cell data, including the variation of duration time and expression intensities (details in ‘Workflow of gene dynamics simulation’). For each cell state at t along a biological process, we apply a Poisson sampling to generate the observed expression level of each gene by taking λ(t) as the mean of Poisson distribution. In these simulation experiments, we know the ground truth of both the pseudotime of each cell in the corresponding biological pro

=== A unified computational framework for single-cell data integration with optimal transport (Cao, Kai; Gong, Qiyu; Hong, Yiguang; Wan, Lin) ===
Article https://doi.org/10.1038/s41467-022-35094-8
A unified computational framework for single-cell data integration with optimal transport
Kai Cao 1,2,5, Qiyu Gong3,5, Yiguang Hong 4 & Lin Wan 1,2
Single-cell data integration can provide a comprehensive molecular view of cells. However, how to integrate heterogeneous single-cell multi-omics as well as spatially resolved transcriptomic data remains a major challenge. Here we introduce uniPort, a unified single-cell data integration framework that combines a coupled variational autoencoder (coupled-VAE) and minibatch unbalanced optimal transport (Minibatch-UOT). It leverages both highly variable common and dataset-specific genes for integration to handle the heterogeneity across datasets, and it is scalable to large-scale datasets. uniPort jointly embeds heterogeneous single-cell multi-omics datasets into a shared latent space. It can further construct a reference atlas for gene imputation across datasets. Meanwhile, uniPort provides a flexible label transfer framework to deconvolute heterogeneous spatial transcriptomic data using an optimal transport plan, instead of embedding latent space. We demonstrate the capability of uniPort by applying it to integrate a variety of datasets, including single-cell transcriptomics, chromatin accessibility, and spatially resolved transcriptomic data.
The latest developments in high-throughput single-cell multi-omics sequencing technologies, e.g., single-cell RNA-sequencing (scRNA) and single-cell Assay for Transposase-Accessible Chromatin using sequencing (scATAC), enable comprehensive studies of heterogeneous cell populations that make up tissues, the dynamics of developmental processes, and the underlying regulatory mechanisms that control cellular functions. The computational integration of single-cell datasets is drawing heavy attention toward making advancements in machine learning and data science1–3. Among existing single-cell integration methods, tremendous efforts4–7 have been devoted to integrating multiple datasets simultaneously profiled from the same cells (e.g., paired-cell datasets generated by the cellular indexing of transcriptomes and epitopes by sequencing (CITE-seq)8). However, these paired datasets are technically challenging and costly to obtain. Therefore, a vast number of
integrative methods have been developed for data profiled from different cells taken from the same, or similar, populations. For example, the celebrated platform Seurat9 projected feature space into a common subspace using canonical correlation analysis (CCA), which maximizes inter-dataset correlation. LIGER10 and DC311 employed nonnegative matrix factorization to find the shared low-dimension factors of the common features to match single-cell omics datasets. Harmony12 iterated between maximum diversity clustering and a mixture model-based linear batch correction, providing a latent space in which batch effects are removed. However, these methods rely on linear operation, thus lacking the ability to handle nonlinear deformations across cellular modalities. In addition, they only leverage filtered common genes, while ignoring the importance of datasetspecific genes for the identification of cell populations, which usually capture cell-type heterogeneity not present in common genes13.
Received: 15 June 2022
Accepted: 18 November 2022
Check for updates
1LSC, NCMIS, Academy of Mathematics and Systems Science, Chinese Academy of Sciences, Beijing, China. 2School of Mathematical Sciences, University of Chinese Academy of Sciences, Beijing, China. 3Shanghai Institute of Immunology, Faculty of Basic Medicine, Shanghai Jiao Tong University School of Medicine, Shanghai, China. 4Department of Control Science and Engineering, Tongji University, Shanghai, China. 5These authors contributed equally: Kai Cao, Qiyu Gong. e-mail: yghong@iss.ac.cn; lwan@amss.ac.cn
Nature Communications | (2022)13:7419 1
1234567890():,;
1234567890():,;


To address these shortcomings, manifold alignment methods are emerging and have achieved promising results in integrating singlecell multi-omics datasets14–17. However, manifold alignment methods are limited by relatively high computational complexity, and they are not scalable to large-scale datasets. With the development of deep learning, many autoencoder-based approaches have been proposed and demonstrated their power in data integration across modalities. However, most of them require paired datasets profiled from the same cells, such as DCCA18 and Cobolt19, to utilize cell-paring information. When cell-paring information is unavailable, the alternative is simultaneous training of different autoencoders and aligning cells across different modalities in a latent space. However, this option still makes computation a challenging exercise. Recently, an emerging number of methods have been developed to account for unpaired data. For example, methods like scDART20 and cross-modal autoencoders21 attempted to learn a latent space by autoencoders and align latent representations via kernelbased or discriminator-based discrepancy. However, these methods require global alignment which is often too restrictive for integrate heterogeneous cellular populations. In addition, the transfer learningbased methods were also developed to establish a source atlas via one modality for knowledge (e.g., cell labels) transfer to another modality by learning a modality-invariant latent space22,23. Although having achieved encouraging results, these methods are restricted to using source modality with annotated cell labels. Recently published methods for single-cell genomics integration such as scMC24 and SCALEX25 showed state-of-the-art performance on batch effect correction of one modality, but they have not been
benchmarked on single-cell multi-omics data integration. GLUE26, another state-of-the-art method for single-cell multi-omics (e.g., scATAC, scRNA) integration and integrative regulatory inference, based its development on advanced graph autoencoders. Meanwhile, many other methods are proposed for integrative analysis of spatial transcriptomics (ST) and scRNA data. Among these methods, gimVI27 and Tangram28 achieved the most advanced performance29. However, to the best of our knowledge, no method has been developed for a unified integration of single-cell multi-omics as well as spatially resolved transcriptomic data. To address this gap, we herein advance the field by developing uniPort, an accurate, robust, and efficient computational platform for integrating heterogeneous single-cell datasets with optimal transport (OT). To overcome the limitation thwarting conventional VAE for single-cell heterogeneous and/or unpaired data integration, we propose a unified computational framework by combining a coupled variational autoencoder (coupled-VAE) and Minibatch Unbalanced OT (Minibatch-UOT)30 (Fig. 1). This framework allows leveraging both highly variable common and dataset-specific genes for integration in order to handle the heterogeneity across datasets. Experimental results show that uniPort can accurately and robustly integrates scATAC and scRNA datasets profiled from peripheral blood mononuclear cells (PBMC) and mouse spleen. It can also accurately impute unmeasured spatially resolved multiplexed error robust fluorescence in situ hybridization (MERFISH)31 genes through scRNA data. Moreover, with an output OT plan, we demonstrate that uniPort can accurately decipher canonical structures of the mouse brain and assist in locating tertiary lymphoid
Fig. 1 | Overview of uniPort algorithm. uniPort integrates single-cell data by combining a coupled-VAE and Minibatch-UOT. uniPort takes as input a highly variable common gene set of single-cell datasets across different modalities or technologies. a uniPort projects input datasets into a cell-embedding latent space through a shared probabilistic encoder. Then uniPort minimizes a Minibatch-UOT loss between cell embeddings across different datasets. Finally, uniPort
reconstructs two terms. The first consists of input datasets by a decoder with different DSBN layers. The second consists of highly variable gene sets corresponding to each dataset by dataset-specific decoders. b uniPort outputs a shared latent space and an optimal transport plan that can be used for downstream analysis, such as visualization, gene imputation and spots deconvolution.
Article https://doi.org/10.1038/s41467-022-35094-8
Nature Communications | (2022)13:7419 2


structures (TLS) in the breast cancer region, as well as reveal cancer heterogeneity in microarray-based spatial data.
Results
uniPort embeds and integrates datasets by coupled-VAE and Minibatch-UOT
As input, uniPort takes diverse and heterogeneous single-cell datasets across different modalities or technologies. uniPort is based on a coupled variational auto-encoder (coupled-VAE) and employs a dataset-free encoder to project highly variable common gene sets of different datasets into a generalized cell-embedding latent space. Then uniPort reconstructs two terms. One is input by a dataset-free decoder with dataset-specific batch normalization (DSBN)25,32 layers. The other is a highly variable gene set through a dataset-specific decoder corresponding to each dataset (Fig. 1 and Supplementary Fig. 1). Some overlapping genes are often found between the two terms as some common genes are also highly variable in each dataset. However, with slight abuse of ‘specific’, we still name the second term a datasetspecific gene set in the following context. During integration, uniPort minimizes a Minibatch-UOT loss between cell embeddings in the latent space from different datasets. It is necessary to introduce the loss as it feeds back a gradient to the encoder to achieve a better alignment result, especially when dataset-specific decoders are considered that increase the heterogeneity across datasets in the latent space. Meanwhile, the minibatch strategy substantially improves the computational efficiency of OT, making it scalable to large datasets, and the unbalanced OT makes it more suitable for heterogeneous data integration. We employed a series of scores to assess the performance of single-cell data integration. To quantify dataset mixing and cell-type separation, we computed two scores used by SCALEX: the Batch Entropy score33 to evaluate the extent of mixing cells across datasets and the Silhouette coefficient34 to evaluate the separation of biological distinctions. To benchmark annotation clustering accuracy, we adopted the adjusted rand index (ARI), the normalized mutual information (NMI), and the F1 scores using cell-type annotations (Methods). Then, for paired datasets, we employed the average fraction of samples closer than the true match (FOSCTTM)14 to measure the preservation of cell–cell correspondence across datasets.
uniPort integrates scATAC and scRNA data
We benchmarked uniPort against current state-of-the-art single-cell genomics integration methods9,10,12,13,15,17,24–26,35 on one dataset of paired scATAC and scRNA (the paired PBMC dataset36) and two datasets of unpaired scATAC and scRNA (the microfluidic-based PBMC dataset37 and the mouse spleen dataset13). We employed Uniform Manifold Approximation and Projection (UMAP)38 to visualize the integration results. We first applied uniPort to integrate the paired PBMC dataset (Fig. 2). The pairing information was only used for performance evaluation. We found that uniPort and GLUE achieved the best performance with comparable results (Fig. 2c–e). Specifically, uniPort achieved the highest Silhouette coefficient of 0.64, while GLUE had the second highest Silhouette coefficient of 0.621; uniPort had the second best average FOSCTTM of 0.0694, a total score of ARI, NMI and F1 of 2.321, and the third highest Batch Entropy score of 0.64, slightly below GLUE (average FOSCTTM of 0.0441, total score of 2.514 and Batch Entropy score of 0.677). Among all compared methods, uniPort, Seurat, Harmony, SCOT, and GLUE accurately integrated most cell types in two modalities (Fig. 2b and Supplementary Fig. 2). In addition to integrating the paired PBMC dataset, we further evaluated uniPort on an unpaired microfluidic-based PBMC dataset (Supplementary Fig. 3). As a result, uniPort accurately integrated the scATAC and scRNA data with competitive performance comparable to
that of GLUE, MultiMAP and Harmony. For example, uniPort aligned most cell types well in two modalities (Supplementary Fig. 3a, b), demonstrating Silhouette coefficient and Batch Entropy score of 0.68 and 0.623 (Supplementary Fig. 3c, d), which were similar to GLUE (0.682 and 0.638, respectively), MultiMAP (0.648 and 0.623, respectively), and Harmony (0.636 and 0.626, respectively), but surpassing other compared methods. We also tested uniPort on another unpaired scATAC and scRNA profiled from the mouse spleen dataset (Fig. 3). uniPort, scMC, Harmony, and Seurat achieved the highest performance. Specifically, uniPort achieved the highest Silhouette coefficient of 0.709, slightly higher than scMC (0.704), Harmony (0.704), and Seurat (0.699); Harmony ranked first in Batch Entropy score of 0.676, higher than uniPort (0.632) and Seurat (0.671); scMC had the highest total score of ARI, NMI and F1 of 2.466, while uniPort (2.436), Harmony (2.416), and Seurat (2.437) followed close behind with a total score for each higher than 2.4, slightly below that of scMC. In summary, among all methods, uniPort performed favorably when compared with recently published state-of-the-art methods, showing accurate and robust results across both paired and unpaired datasets.
uniPort performs unbalanced matching tasks of heterogeneous datasets
uniPort minimizes a Minibatch-UOT loss, which is suitable for unbalanced matching and provides a strong guarantee for heterogeneous data integration. To evaluate the performance of uniPort on heterogeneous data integration, we conducted two unbalanced matching tasks by removing some cell types from scATAC or scRNA of mouse spleen, separately. First, we removed “DC”, “Granulocyte”, “Macrophage” and “NK” types from scATAC data, while keeping scRNA data unchanged, and denoted the integration task as ATAC unbalanced matching (“UBM-ATAC”). Second, we removed the same cell types from scRNA data, while keeping scATAC data unchanged, and denoted the integration task as RNA unbalanced matching (“UBM-RNA”). For comparison, we also defined the integration of complete mouse spleen data as balanced matching (“BM”). uniPort accurately identified and separated the cells of “DC”, “Granulocyte”, “Macrophage” and “NK” from other cell types in the two unbalanced matching cases, while still aligning modality-shared cell types well (Fig. 4a, b). We compared uniPort with GLUE, Harmony, Seurat, MultiMAP, and scMC, all of which achieved high accurate performance on the “BM” task. Among all methods, only uniPort and Seurat achieved stable performance in all three cases (Fig. 4c, d). uniPort had the highest total score of 2.225 and Silhouette coefficient of 0.676, in the case of “UBM-ATAC”, and the second highest total score of 2.191 and the third highest Silhouette coefficient of 0.688, in the case of “UBM-RNA”. Therefore, compared with the case of “BM”, uniPort is more robust than the other methods when heterogeneity is presented in the datasets.
uniPort integrates MERFISH and scRNA data
We further considered the integration of ST and scRNA data. Two main types of ST sequencing technologies are high-plex RNA imaging-based and barcoding-based. High-plex RNA imaging-based spatial sequencing has the advantage of single-cell precision with greater depth, but it is restricted to partial measurement with lower coverage. To test the performance of uniPort over high-plex RNA imaging-based data, we applied uniPort to integrate MERFISH and scRNA data31. Among 155 genes in the MERFISH data, we used 153 common genes in both scRNA and MERFISH for integration. We applied UMAP to visualize the integration results of cell embeddings by uniPort, Harmony, Seurat, SCALEX, scVI, gimVI27 and MultiMAP (Fig. 5a, b and Supplementary Fig. 4). As shown in the figures, uniPort and scVI
Article https://doi.org/10.1038/s41467-022-35094-8
Nature Communications | (2022)13:7419 3


Fig. 3 | uniPort integrates unpaired scATAC and scRNA of the mouse spleen data. a UMAP visualization of mouse spleen data before integration colored by omics and cell annotations. b UMAP visualization of mouse spleen data after
uniPort integration. c Comparison of Batch Entropy scores and Silhouette coefficients of different methods. d Comparison of total scores of ARI, NMI and F1 of different methods.
Fig. 2 | uniPort integrates paired scATAC and scRNA of the PBMC data from 10× Genomics. a UMAP visualization of PBMC data before integration colored by omics and cell annotations. b UMAP visualization of PBMC data after uniPort integration.
c Comparison of total scores of ARI, NMI and F1 of different methods. d Comparison of Batch Entropy scores and Silhouette coefficients of different methods. e Comparison of average FOSCTTM of different methods.
Article https://doi.org/10.1038/s41467-022-35094-8
Nature Communications | (2022)13:7419 4


outperformed other methods in identifying and separating OD Immature cells from other cell types. Besides, uniPort accurately identified ependymal cells as a MERFISH-specific cell type and separated them from other cell types in the scRNA embeddings. We again benchmarked uniPort’s integration performance against other methods9,12,13,25,27,35 by the Silhouette coefficient and the total score (Fig. 5c, d). We found that uniPort outperformed other methods with the highest Silhouette coefficient of 0.706 and the highest total score of 2.404 for ARI, NMI and F1, while scVI achieved the second highest Silhouette coefficient of 0.688, and MultiMAP ranked second in the total score of 2.37.
uniPort imputes genes for MERFISH data
uniPort 

=== NovoSpaRc: flexible spatial reconstruction of single-cell gene expression with optimal transport (Friedman, Nir; Moriel, Noa; Senel, Enes; Rajewsky, Nikolaus; Karaiskos, Nikos; N) ===
NovoSpaRc: flexible spatial reconstruction
of single-cell gene expression with optimal
transport
Noa Moriel1,6, Enes Senel2,6, Nir Friedman 1,3, Nikolaus Rajewsky 2, Nikos Karaiskos 2✉
and Mor Nitzan 1,4,5✉
Single-cell RNA-sequencing (scRNA-seq) technologies have revolutionized modern biomedical sciences. A fundamental challenge is to incorporate spatial information to study tissue organization and spatial gene expression patterns. Here, we describe a detailed protocol for using novoSpaRc, a computational framework that probabilistically assigns cells to tissue locations. At the core of this framework lies a structural correspondence hypothesis, that cells in physical proximity share similar gene expression profiles. Given scRNA-seq data, novoSpaRc spatially reconstructs tissues based on this hypothesis, and optionally, by including a reference atlas of marker genes to improve reconstruction. We describe the novoSpaRc algorithm, and its implementation in an open-source Python package (https://pypi.org/ project/novosparc). NovoSpaRc maps a scRNA-seq dataset of 10,000 cells onto 1,000 locations in <5 min. We describe results obtained using novoSpaRc to reconstruct the mouse organ of Corti de novo based on the structural correspondence assumption and human osteosarcoma cultured cells based on marker gene information, and provide a step-by-step guide to Drosophila embryo reconstruction in the Procedure to demonstrate how these two strategies can be combined.
Introduction
The emergence of single-cell RNA sequencing (scRNA-seq) technologies during the past decade has transformed the biomedical sciences1,2. High-throughput methods have enabled the simultaneous profiling of tens of thousands of cellular transcriptomes stemming from the same tissue3,4, and have been successfully employed throughout multiple discoveries, such as to dissect tissue heterogeneity5,6, to identify rare cell populations5,7,8, and to investigate cell states5,9 and cell differentiation processes10,11, among others. Most scRNA-seq methods, however, require dissociation of the tissue, which results in the loss of spatial information. The physical context of the cells is vital for the understanding of biological functions at the global collective scale, such as spatial gene expression patterns12–15, the organization of cell types in space8,16,17, and heterogeneous responses to perturbations or drug responses throughout diseased tissues18. At the local level, spatial information is critical to thoroughly study cell–cell interactions and individual cellular states19. A growing number of experimental techniques that preserve spatial information have been developed over the past few years to bridge this gap20. While these techniques are generally still at least partially limited in throughput14,16,17,21,22 spatial resolution23 and commercially available solutions are often costly and do not offer single-cell resolution (10x Genomics (https://www. 10xgenomics.com/products/spatial-gene-expression), Spatial Transcriptomics (https://spatialtra nscriptomics.com) and GeoMx Digital Spatial Profiling (https://www.nanostring.com/products/ geomx-digital-spatial-profiler/geomx-dsp-overview), experimental techniques are constantly diversifying, advancing and improving24. However, there is an urgent need to decipher spatial information from the vast single-cell data that already exist. Furthermore, there is a need to leverage the expanding
1School of Computer Science and Engineering, The Hebrew University of Jerusalem, Jerusalem, Israel. 2Systems Biology of Gene Regulatory Elements, Berlin Institute for Medical Systems Biology, Max Delbrück Center for Molecular Medicine in the Helmholtz Association, Berlin, Germany. 3Institute of Life Sciences, The Hebrew University of Jerusalem, Jerusalem, Israel. 4Racah Institute of Physics, The Hebrew University of Jerusalem, Jerusalem, Israel. 5Faculty of Medicine, The Hebrew University of Jerusalem, Jerusalem, Israel. 6These authors contributed equally: Noa Moriel, Enes Senel.
✉e-mail: nikolaos.karaiskos@mdc-berlin.de; mor.nitzan@mail.huji.ac.il
NATURE PROTOCOLS | VOL 16 | SEPTEMBER 2021 | 4177–4200 | www.nature.com/nprot 4177
PROTOCOL
https://doi.org/10.1038/s41596-021-00573-7
1234567890():,;
1234567890():,;


set of high-quality spatial transcriptomic experiments as complementary information for scRNA-seq data and learn how to efficiently integrate these two sources of information. The challenge of reconstructing spatial gene expression from single-cell data is tackled by multiple computational techniques that require the existence of a spatial atlas of marker genes to be used as a reference guide. Such a reference atlas is generally only feasible for stereotypical tissues with robust, relatively simple spatial expression patterns (which can repeat across multiple subunits within the tissue), such as liver lobules, the intestinal epithelium and some embryos at early developmental stages. In addition, such a reference atlas may not be straightforward to construct25–29. Recently, we presented novoSpaRc30, a new computational approach that can spatially reconstruct gene expression without the need of a reference atlas, while being able to incorporate it and enhance performance if such an atlas exists. NovoSpaRc is based on the hypothesis that physically neighboring cells share similar transcriptional profiles, so that gene expression, on average, does not change abruptly but in a continuous manner for a substantial subset of genes. We formulated this hypothesis within the framework of optimal transport(OT)31,32, which allows us to probabilistically assign single cells to tissue locations by interpolating between the continuity assumption and other types of prior experimental data, such as the spatial expression of a subset of marker genes (or a reference atlas), the local density of cells in the tissue and the technical quality of read measurements extracted from single cells. In this paper, we provide detailed guidelines for using novoSpaRc to recover the spatial organization of cells and genes in their tissue-of-origin based on single-cell data.
Overview of the algorithm and workflow
The main objective of novoSpaRc is to probabilistically map single cells onto the tissue’s physical structure, and infer gene expression patterns across the tissue. To do that, novoSpaRc requires a gene expression matrix and a target space (coordinates of the physical space). Atlas expression, that is, spatial expression of a subset of genes over the tissue, is an additional optional input. Using these inputs, novoSpaRc computes three cost matrices, which together allow us to interpolate between minimizing the deviation of a certain mapping from a structural correspondence assumption between distances of cells in gene expression and physical space, and from a potentially available reference atlas. NovoSpaRc outputs a transport matrix, a probabilistic mapping of cells onto the target space locations, using the OT framework, and computes the inferred spatial gene expression over the target space. The workflow is schematically represented in Fig. 1 along a detailed description below of each of these steps, the inputs and outputs of novoSpaRc, and optional validations and follow-up analyses.
Input cells and locations descriptions to construct Tissue object (Steps 1–6)
Cell expression. The main input to the novoSpaRc algorithm is a gene expression matrix that captures single-cell gene expression levels within a population of cells. Cell-by-gene matrices where each entry is the count of RNA molecules retrieved from scRNA-seq are a typical input. However, outputs of other experimental procedures that quantify gene expression levels can be integrated as well, such as using RNA quantization through amplification rounds33 (see organ of Corti example below), fluorescent imaging14,16,17 (see osteosarcoma example below) or other sequencing techniques (refs. 23 and34 and from 10x Genomics (https://www.10xgenomics.com/products/spatial-gene-expression)) (demonstrated for Slide-seq data in ref. 30). Preprocessing of the gene expression matrix can be minimal, such as the standard librarynormalization scheme of cell-count normalization and log transformation for scRNA-seq data35. Since scRNA-seq protocols suffer from low capture probabilities and expression representation is redundant and extensive in dimensions (e.g., ~20,000 genes), using a meaningful low-dimensional representation of expression such as a highly variable set of genes or a latent representation of expression (e.g., using principal component analysis (PCA)) can drastically enhance the quality and runtime of reconstruction.
Target space. A target space is a set of coordinates corresponding to the physical locations across the tissue onto which novoSpaRc maps the single cells. The set of locations can span any 1D, 2D or 3D structure corresponding either to the explicit tissue structure or a representation that captures the structure’s inherent spatial symmetries. For optimal reconstruction results, the shape of the target space should resemble the shape (or underlying symmetries) of the tissue-of-origin, as the inherent coordinate relationships will be used for the spatial reconstruction. Note that while a faithful
PROTOCOL NATURE PROTOCOLS
4178 NATURE PROTOCOLS | VOL 16 | SEPTEMBER 2021 | 4177–4200 | www.nature.com/nprot


Input cells and locations descriptions to construct Tissue object Steps 1–6
Step 4
a
b
c
d
Location
X
Steps 1–3
Construct Tissue object
Step 6
Steps 7–8
Location–location physical distance
Cell–cell expression distance
(Optional) atlas: cell–location expression distance
KNN graph KNN graph
Cells Locations
Cell
Cell
Cell
Cell
Compute
transport
Predict
expression
Cell
Location
Location
Location
=
D phys Dexp D exp,phys
Compute optimal transport of cells to locations and predict expression over target space tissue.reconstruct(α) Step 9
Fetch mapping and predicted expression over target space
Cell-to-location mapping tissue.gw
Predicted expression over target space tissue.sdge
Gene 2
Cells ij locations kl
Cells i locations k
Tik – εH (T)
D exp.phys
ik
Gene k
Gene 1
= argmincoupling T
Σ
Σ
Gene 1 Gene 2
Gene k
Cell1
Cell2
Cell3
Celln
Location
Location
Compute cost matrices
Cell1
Celln
Cell2
Cell3
Gene 1 Gene 2
Gene k
y
Target space e.g., locations = novosparc.geometry.construct_sphere(...)
dataset = sc.read(...)
Cell expression (Optional) atlas expression e.g., atlas_matrix = sc.read(...).X Step 5
Gene 2
Gene 4
Gene 5
tissue = novosparc.cm.Tissue(dataset, locations, atlas_matrix)
tissue.sdge
tissue.gw
tissue.gw
tissue.setup_reconstruction(...)
(1 – α)
+α
L(D phys,D exp)TikTjl
kl ij
•••
•
••
••
••
•
•
•••
NATURE PROTOCOLS PROTOCOL
NATURE PROTOCOLS | VOL 16 | SEPTEMBER 2021 | 4177–4200 | www.nature.com/nprot 4179


representation of the tissue-of-origin shape or symmetries is ideal, simpler target spaces or ones that only capture local structures and symmetries of the tissue are many times sufficient. There are two ways to create the target space if no prior reference is available. The most straightforward way is to use novoSpaRc’s internal functions and create a basic shape target space, e.g., rectangle, circle, sphere, prism, etc. If we are interested in reconstructing spatial variability along a single axis, for example, such as that corresponding to a 1D gradient of oxygen or morphogenes in the biological system, we should use a linear target space. For example, in the organ of Corti, a 2D spiral organ essential for hearing, gene expression within cell subpopulations mainly varies along a 1D apexto-base axis. By constructing a corresponding 1D target space, we illustrate novoSpaRc expression reconstruction along this axis (see ‘De novo spatial reconstruction of the organ of Corti’ below). Additionally, a target space can be created from experimental measurements. Representative images can be processed to determine cell locations (along with their corresponding information for gene expression). This is illustrated in the expression reconstruction of human osteosarcoma cultured cells where cell locations are deduced from microscope imaging obtained using multiplexed error-robust fluorescence in situ hybridization (MERFISH) (see osteosarcoma example below). A 3D analog of this case is illustrated in the reconstruction of the Drosophila embryo described in the Procedure.
(Optional) Atlas expression. A reference atlas is an optional input, carrying information about the expression levels of a subset of genes across the target space. Such a reference atlas can be incorporated into novoSpaRc and increase the reconstruction quality by essentially restricting the space of possible reconstruction solutions to those that are consistent with the atlas, or by spatially regulating the mapping process. The reference atlas can guide the selection of the target space. For example, if marker gene expression is measured using in situ imaging at single-cell resolution, then we can set the target locations at the cells’ centroid locations. To account for atlas information at lower resolution, such as retrieved experimentally from bulk sequencing of sectioned tissue36–38, or from computational local aggregation of nearby cells due to low signal23, spatial expression of genes is binned and integrated (e.g., averaged) to provide expression over the set of target locations. In general, there are no special requirements or restrictions regarding the data format, number of genes and experimental method used to construct the reference atlas. However, reconstruction is likely to benefit from a reference atlas quantifying the expression of spatially informative genes. Given the target space locations, cellular gene expression and, optionally, the reference atlas of spatial expression, we construct a Tissue object, the main object of the novoSpaRc package.
Compute cost matrices (Steps 7 and 8)
Having the normalized gene expression matrix and the target space at our disposal, and potentially a reference atlas, we continue with computing the cost matrices that are needed for performing the spatial reconstruction.
Computing the cell–cell and location–location cost matrices. The cell–cell cost matrix summarizes the distances between cells in gene expression space, and the location–location cost matrix summarizes the physical distances between locations in the target space. The assumption at the heart of novoSpaRc is that there is a correspondence between the structure of locations in physical space and the structure of cells in gene expression space potentially along a low-dimensional nonlinear manifold. More concretely, it implies that there is a correspondence between pairwise distances of locations in physical space and cells in gene expression space. To capture distances along potentially nonlinear low-dimensional structures, we construct k-nearest-neighbors (kNN) graphs (based on Euclidean distances) in physical space and in gene expression space. The corresponding cost matrices consist of pairwise distances between cells and locations, computed as the shortest paths along the corresponding kNN graphs. These cost matrices would be used to capture the essence of the
Fig. 1 | Schematic representation of the novoSpaRc algorithm. a, Preparation of inputs for novoSpaRc’s Tissue object—constructing a target space and reading gene expression datasets. If a reference atlas is used, then the target space corresponds to its locations. b, Computation of cost matrices including physical distances between locations and expression distances between cells, both computed as the shortest path in kNN graphs. If a reference atlas is used, then an additional cost matrix of atlas correspondence captures the expression discrepancy between locations and cells according to the reference atlas. c, Computation of the OT of cells to locations (tissue.gw) given a parameter α interpolating between the structural correspondence and atlas correspondence objectives. The predicted expression of genes over locations (tissue.sdge) is then computed by matrix multiplication of cellular gene expression and their probabilistic mapping to locations (tissue.gw). The output probabilistic embedding and the predicted spatial gene expression can be fetched from the Tissue object.
PROTOCOL NATURE PROTOCOLS
4180 NATURE PROTOCOLS | VOL 16 | SEPTEMBER 2021 | 4177–4200 | www.nature.com/nprot


structural correspondence assumption, that is, the averaged transcriptional similarity among physically proximal cells.
Computing the reference atlas cost matrix. If a reference atlas is available for a subset of genes, the corresponding cost matrix captures the discrepancy between the expression of these genes in each cell and in each location of the target space. Specifically, we compute the Euclidean distance across the subset of genes composing the reference atlas between the cells and locations.
Compute OT of cells to locations and predict expression over target space (Step 9)
Setting marginal distributions. Here we set the marginal distributions for both the cells and locations. By default, novoSpaRc initializes the marginal distributions to be uniform. This means that the total spatial mapping probability associated with each cell is the same, and the total mapping probability associated with each location is the same. In cases where nonuniform mapping is desired, where prior biological knowledge exists for the physical density of cells, or where there is varying technical quality of cells, this can be readily incorporated at this step.
Setting the alpha parameter. The alpha parameter is used to interpolate between two modes of reconstruction: (1) a de novo spatial reconstruction (α = 0), based only on the underlying structural correspondence assumption, and (2) a reconstruction based only on the information provided by the reference atlas for the spatial expression of a set of marker genes (α = 1). Inte

