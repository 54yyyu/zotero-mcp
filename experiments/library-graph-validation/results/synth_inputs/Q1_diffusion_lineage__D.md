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

=== Tutorial on Diffusion Models for Imaging and Vision (Chan, Stanley H.) ===
arXiv:2403.18103v1 [cs.LG] 26 Mar 2024

Tutorial on Diffusion Models for Imaging and Vision
Stanley Chan1
March 28, 2024
Abstract. The astonishing growth of generative tools in recent years has empowered many exciting applications in text-to-image generation and text-to-video generation. The underlying principle behind these generative tools is the concept of diffusion, a particular sampling mechanism that has overcome some shortcomings that were deemed difficult in the previous approaches. The goal of this tutorial is to discuss the essential ideas underlying the diffusion models. The target audience of this tutorial includes undergraduate and graduate students who are interested in doing research on diffusion models or applying these models to solve other problems.

Contents

1 The Basics: Variational Auto-Encoder (VAE)

2

1.1 VAE Setting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2

1.2 Evidence Lower Bound . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

1.3 Training VAE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

1.4 Loss Function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

1.5 Inference with VAE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

2 Denoising Diffusion Probabilistic Model (DDPM)

10

2.1 Building Blocks . . .√. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10 2.2 The magical scalars αt and 1 − αt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13 2.3 Distribution qϕ(xt|x0) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

2.4 Evidence Lower Bound . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

2.5 Rewrite the Consistency Term . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

2.6 Derivation of qϕ(xt−1|xt, x0) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

2.7 Training and Inference . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

2.8 Derivation based on Noise Vector . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

2.9 Inversion by Direct Denoising (InDI) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

3 Score-Matching Langevin Dynamics (SMLD)

30

3.1 Langevin Dynamics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30

3.2 (Stein’s) Score Function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33

3.3 Score Matching Techniques . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35

4 Stochastic Differential Equation (SDE)

39

4.1 Motivating Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39

4.2 Forward and Backward Iterations in SDE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41

4.3 Stochastic Differential Equation for DDPM . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43

4.4 Stochastic Differential Equation for SMLD . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45

4.5 Solving SDE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46

5 Conclusion

49

1School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN 47907. Email: stanchan@purdue.edu.

© 2024 Stanley Chan. All Rights Reserved.

1

1 The Basics: Variational Auto-Encoder (VAE)
1.1 VAE Setting
A long time ago, in a galaxy far far away, we want to build a generator that generates images from a latent code. The simplest (and perhaps one of the most classical) approach is to consider an encoder-decoder pair shown below. This is called a variational autoencoder (VAE) [1, 2, 3].

The autoencoder has an input variable x and a latent variable z. For the sake of understanding the subject, we treat x as a beautiful image and z as some kind of vector living in some high dimensional space.
Example. Getting a latent representation of an image is not an alien thing. Back in the time of JPEG compression (which is arguably a dinosaur), we use discrete cosine transform (DCT) basis φn to encode the underlying image / patches of an image. The coefficient vector z = [z1, . . . , zN ]T is obtained by projecting the patch x onto the space spanned by the basis: zn = ⟨φn, x⟩. So, if you give us an image x, we will return you a coefficient vector z. From z we can do inverse transform to recover (ie decode) the image. Therefore, the coefficient vector z is the latent code. The encoder is the DCT transform, and the decoder is the inverse DCT transform.

The name “variational” comes from the factor that we use probability distributions to describe x and z. Instead of resorting to a deterministic procedure of converting x to z, we are more interested in ensuring that the distribution p(x) can be mapped to a desired distribution p(z), and go backwards to p(x). Because of the distributional setting, we need to consider a few distributions.

• p(x): The distribution of x. It is never known. If we knew it, we would have become a billionaire. The whole galaxy of diffusion models is to find ways to draw samples from p(x).
• p(z): The distribution of the latent variable. Because we are all lazy, let’s just make it a zero-mean unit-variance Gaussian p(z) = N (0, I).
• p(z|x): The conditional distribution associated with the encoder, which tells us the likelihood of z when given x. We have no access to it. p(z|x) itself is not the encoder, but the encoder has to do
something so that it will behave consistently with p(z|x). • p(x|z): The conditional distribution associated with the decoder, which tells us the posterior proba-
bility of getting x given z. Again, we have no access to it.

The four distributions above are not too mysterious. Here is a somewhat trivial but educational example that can illustrate the idea.

Example. Consider a random variable X distributed according to a Gaussian mixture model with

a latent variable z ∈ {1, . . . , K} denoting the cluster identity such that pZ(k) = P[Z = k] = πk for

k = 1, . . . , K. We assume

K k=1

πk

=

1.

Then,

if

we

are

told

that

we

need

to

look

at

the

k-th

cluster

only, the conditional distribution of X given Z is

pX|Z (x|k) = N (x | µk, σk2I).

© 2024 Stanley Chan. All Rights Reserved.

2

The marginal distribution of x can be found using the law of total probability, giving us

K

K

pX(x) = pX|Z (x|k)pZ (k) = πkN (x | µk, σk2I).

(1)

k=1

k=1

Therefore, if we start with pX(x), the design question for the encoder to build a magical encoder such that for every sample x ∼ pX(x), the latent code will be z ∈ {1, . . . , K} with a distribution z ∼ pZ(k).
To illustrate how the encoder and decoder work, let’s assume that the mean and variance are known and are fixed. Otherwise we will need to estimate the mean and variance through an EM algorithm. It is doable, but the tedious equations will defeat the purpose of this illustration.
Encoder: How do we obtain z from x? This is easy because at the encoder, we know pX(x) and pZ(k). Imagine that you only have two class z ∈ {1, 2}. Effectively you are just making a binary decision of where the sample x should belong to. There are many ways you can do the binary decision. If you like maximum-a-posteriori, you can check

pZ|X(1|x)

≷ccllaassss

1 2

pZ|X(2|x),

and this will return you a simple decision rule. You give us x, we tell you z ∈ {1, 2}. Decoder: On the decoder side, if we are given a latent code z ∈ {1, . . . , K}, the magical decoder
just needs to return us a sample x which is drawn from pX|Z (x|k) = N (x | µk, σk2I). A different z will give us one of the K mixture components. If we have enough samples, the overall distribution will follow the Gaussian mixture.
Smart readers like you will certainly complain: “Your example is so trivially unreal.” No worries. We understand. Life is of course a lot harder than a Gaussian mixture model with known means and known variance. But one thing we realize is that if we want to find the magical encoder and decoder, we must have a way to find the two conditional distributions. However, they are both high-dimensional creatures. So, in order for us to say something more meaningful, we need to impose additional structures so that we can generalize the concept to harder problems.
In the literature of VAE, people come up with an idea to consider the following two proxy distributions:
• qϕ(z|x): The proxy for p(z|x). We will make it a Gaussian. Why Gaussian? No particular good reason. Perhaps we are just ordinary (aka lazy) human beings.
• pθ(x|z): The proxy for p(x|z). Believe it or not, we will make it a Gaussian too. But the role of this Gaussian is slightly different from the Gaussian qϕ(z|x). While we will need to estimate the mean and variance for the Gaussian qϕ(z|x), we do not need to estimate anything for the Gaussian pθ(x|z). Instead, we will need a decoder neural network to turn z into x. The Gaussian pθ(x|z) will be used to inform us how good our generated image x is.
The relationship between the input x and the latent z, as well as the conditional distributions, are summarized in Figure 1. There are two nodes x and z. The “forward” relationship is specified by p(z|x) (and approximated by qϕ(z|x)), whereas the “reverse” relationship is specified by p(x|z) (and approximated by pθ(x|z)).

Figure 1: In a variational autoencoder, the variables x and z are connected by the conditional distributions p(x|z) and p(z|x). To make things work, we introduce two proxy distributions pθ(x|z) and qϕ(z|x), respectively.

© 2024 Stanley Chan. All Rights Reserved.

3

Example. It’s time to consider another trivial example. Suppose that we have a random variable x and a latent variable z such that
x ∼ N (x | µ, σ2), z ∼ N (z | 0, 1).
Our goal is to construct a VAE. (What?! This problem has a trivial solution where z = (x − µ)/σ and x = µ + σz. You are absolutely correct. But please follow our derivation to see if the VAE framework makes sense.)

By constructing a VAE, we mean that we want to build two mappings “encode” and “decode”. For simplicity, let’s assume that both mappings are affine transformations:

z = encode(x) = ax + b, x = decode(z) = cz + d,

so that ϕ = [a, b], so that θ = [c, d].

We are too lazy to find out the joint distribution p(x, z), nor the conditional distributions p(x|z) and p(z|x). But we can construct the proxy distributions qϕ(z|x) and pθ(x|z). Since we have the freedom to choose what qϕ and pθ should look like, how about we consider the following two Gaussians

qϕ(z|x) = N (z | ax + b, 1), pθ(x|z) = N (x | cz + d, c).

The choice of these two Gaussians is not mysterious. For qϕ(z|x): if we are given x, of course we want the encoder to encode the distribution according to the structure we have chosen. Since the encoder structure is ax + b, the natural choice for qϕ(z|x) is to have the mean ax + b. The variance is chosen as 1 because we know that the encoded sample z should be unit-variance. Similarly, for pθ(x|z): if we are given z, the decoder must take the form of cz + d because this is how we setup the decoder. The variance is c which is a parameter we need to figure out.
We will pause for a moment before continuing this example. We want to introduce a mathematical tool.

1.2 Evidence Lower Bound
How do we use these two proxy distributions to achieve our goal of determining the encoder and the decoder? If we treat ϕ and θ as optimization variables, then we need an objective function (or the loss function) so that we can optimize ϕ and θ through training samples. To this end, we need to set up a loss function in terms of ϕ and θ. The loss function we use here is called the Evidence Lower BOund (ELBO) [1]:

ELBO(x) d=ef Eqϕ(z|x)

p(x, z) log
qϕ(z|x)

.

(2)

You are certainly puzzled how on the Earth people can come up with this loss function!? Let’s see what ELBO means and how it is derived.

© 2024 Stanley Chan. All Rights Reserved.

4

In a nutshell, ELBO is a lower bound for the prior distribution log p(x) because we can show that

p(x, z)

log p(x) = some magical steps = Eqϕ(z|x)

log qϕ(z|x)

+ DKL(qϕ(z|x)∥p(z|x))

(3)

p(x, z)

≥ Eqϕ(z|x)

log qϕ(z|x)

d=ef ELBO(x),

where the inequality follows from the fact that the KL divergence is always non-negative. Therefore, ELBO is a valid lower bound for log p(x). Since we never have access to log p(x), if we somehow have access to ELBO and if ELBO is a good lower bound, then we can effectively maximize ELBO to achieve the goal of maximizing log p(x) which is the gold standard. Now, the question is how good the lower bound is. As you can see from the equation and also Figure 2, the inequality will become an equality when our proxy qϕ(z|x) can match the true distribution p(z|x) exactly. So, part of the game is to ensure qϕ(z|x) is close to p(z|x).

Figure 2: Visualization of log p(x) and ELBO. The gap between the two is determined by the KL divergence DKL(qϕ(z|x)∥p(z|x)).

Proof of Eqn (3). The whole trick here is to use our magical proxy qϕ(z|x) to poke around p(x) and derive the bound.

log p(x) = log p(x) × qϕ(z|x)dz

multiply 1

=1

=

log p(x) × qϕ(z|x) dz

some constant wrt z
= Eqϕ(z|x)[log p(x)],

distribution in z

move log p(x) into integral (4)

where the last equality is an interesting fact that a × pZ(z)dz = E[a] for any random variable Z and a scalar a. Of course, E[a] = a.
See, we have already got Eqϕ(z|x)[·]. Just a few more steps. Let’s use Bayes theorem which states that p(x, z) = p(z|x)p(x):

p(x, z)

Eqϕ(z|x)[log p(x)] = Eqϕ(z|x)

log p(z|x)

= Eqϕ(z|x)

log p(x, z) × qϕ(z|x) p(z|x) qϕ(z|x)

= Eqϕ(z|x)

p(x, z) log
qϕ(z|x)

+ Eqϕ(z|x)

log qϕ(z|x) , p(z|x)

ELBO

DKL (qϕ (z|x)∥p(z|x))

Bayes Theorem Multiply and divide qϕ(z|x)
(5)

where we recognize that the first term is exactly ELBO, whereas the second term is exactly the KL divergence. Comparing Eqn (5) with Eqn (3), we know that life is good.

© 2024 Stanley Chan. All Rights Reserved.

5

We now have ELBO. But this ELBO is still not too useful because it involves p(x, z), something we have no access to. So, we need to do a little more things. Let’s take a closer look at ELBO

ELBO(x) d=ef Eqϕ(z|x)

p(x, z) log
qϕ(z|x)

p(x|z)p(z) = Eqϕ(z|x) log qϕ(z|x)

p(z)

= Eqϕ(z|x) [log p(x|z)] + Eqϕ(z|x)

log qϕ(z|x)

= Eqϕ(z|x) [log pθ(x|z)] − DKL(qϕ(z|x)∥p(z)),

definition p(x, z) = p(x|z)p(z)
split expectation definition of KL

where we secretly replaced the inaccessible p(x|z) by its proxy pθ(x|z). This is a beautiful result. We just showed something very easy to understand.

a Gaussian

a Gaussian a Gaussian

ELBO(x) = Eqϕ(z|x)[log pθ(x|z) ]

−

DKL qϕ(z|x) ∥ p(z) .

(6)

how good your decoder is

how good your encoder is

There are two terms in Eqn (6):
• Reconstruction. The first term is about the decoder. We want the decoder to produce a good image x if we feed a latent z into the decoder (of course!!). So, we want to maximize log pθ(x|z). It is similar to maximum likelihood where we want to find the model parameter to maximize the likelihood of observing the image. The expectation here is taken with respect to the samples z (conditioned on x). This shouldn’t be a surprise because the samples z are used to assess the quality of the decoder. It cannot be an arbitrary noise vector but a meaningful latent vector. So, z needs to be sampled from qϕ(z|x).
• Prior Matching. The second term is the KL divergence for the encoder. We want the encoder to turn x into a latent vector z such that the latent vector will follow our choice of (lazy) distribution N (0, I). To be slightly more general, we write p(z) as the target distribution. Because KL is a distance (which increases when the two distributions become more dissimilar), we need to put a negative sign in front so that it increases when the two distributions become more similar.

Example. Let’s continue our trivial Gaussian example. We know from our previous derivation that

qϕ(z|x) = N (z | ax + b, 1), pθ(x|z) = N (x | cz + d, c).

To determine θ and ϕ, we need to minimize the prior matching error and maximize the reconstruction term. For the prior matching, we know that

DKL(qϕ(z|x)∥p(z)) = DKL (N (z | ax + b, 1) ∥ N (z | 0, 1)) .

Since

E[x]

=

µ

and

Var[x]

=

σ2,

the

KL-divergence

is

minimized

when

a

=

1 σ

and

b

=

−

µ σ

so

that

ax + b

=

x−µ σ

.

It

then

follows

that

E[ax + b]

=

0,

and

Var[ax + b]

=

1.

For

the

reconstruction

term,

we

know that

(cz + d − µ)2

Eqϕ(z|x)[log pθ(x|z)] = Eqϕ(z|x) −

2c2

.

Since E[z] = 0 and Var[z] = 1, it follows that the term is maximized when c = σ and d = µ.

© 2024 Stanley Chan. All Rights Reserved.

6

To conclude, the encoder and decoder parameters are

x−µ

z = encode(x) =

,

σ

x = decode(z) = σz + µ,

which is fairly easy to understand.

The reconstruction term and the prior matching terms are illustrated in Figure 3. In both cases, and
during training, we assume that we have access to both z and x, where z needs to be sampled from qϕ(z|x). Then for reconstruction, we estimate θ to maximize pθ(x|z). For prior matching, we find ϕ to minimize the KL divergence. The optimization can be challenging, because if you update ϕ, the distribution qϕ(z|x) will change.

Figure 3: Interpreting the reconstruction term and the prior matching term in ELBO for a variational autoencoder.

1.3 Training VAE
Now that we understand the meaning of ELBO, we can discuss how to train the VAE. To train a VAE, we need the ground truth pairs (x, z). We know how to get x; it is just the image from a dataset. But correspondingly what should z be?
Let’s talk about the encoder. We know that z is generated from the distribution qϕ(z|x). We also know that qϕ(z|x) is a Gaussian. Assume that this Gaussian has a mean µ and a covaria

=== DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps (Lu, Cheng; Zhou, Yuhao; Bao, Fan; Chen, Jianfei; Li, Chongxuan; Zhu, Jun) ===
DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps
Cheng Lu†, Yuhao Zhou†, Fan Bao†, Jianfei Chen†∗, Chongxuan Li‡, Jun Zhu†∗ †Dept. of Comp. Sci. & Tech., Institute for AI, BNRist Center, THBI Lab †Tsinghua-Bosch Joint ML Center, Tsinghua University, Beijing, 100084 China ‡Gaoling School of Artificial Intelligence, Renmin University of China, ‡Beijing Key Laboratory of Big Data Management and Analysis Methods, Beijing, China {lucheng.lc15, yuhaoz.cs}@gmail.com; bf19@mails.tsinghua.edu.cn chongxuanli@ruc.edu.cn; {jianfeic, dcszj}@tsinghua.edu.cn
Abstract
Diffusion probabilistic models (DPMs) are emerging powerful generative models. Despite their high-quality generation performance, DPMs still suffer from their slow sampling as they generally need hundreds or thousands of sequential function evaluations (steps) of large neural networks to draw a sample. Sampling from DPMs can be viewed alternatively as solving the corresponding diffusion ordinary differential equations (ODEs). In this work, we propose an exact formulation of the solution of diffusion ODEs. The formulation analytically computes the linear part of the solution, rather than leaving all terms to black-box ODE solvers as adopted in previous works. By applying change-of-variable, the solution can be equivalently simplified to an exponentially weighted integral of the neural network. Based on our formulation, we propose DPM-Solver, a fast dedicated high-order solver for diffusion ODEs with the convergence order guarantee. DPM-Solver is suitable for both discrete-time and continuous-time DPMs without any further training. Experimental results show that DPM-Solver can generate high-quality samples in only 10 to 20 function evaluations on various datasets. We achieve 4.70 FID in 10 function evaluations and 2.87 FID in 20 function evaluations on the CIFAR10 dataset, and a 4 ∼ 16× speedup compared with previous state-of-the-art training-free samplers on various datasets.2
1 Introduction
Diffusion probabilistic models (DPMs) [1–3] are emerging powerful generative models with promising performance on many tasks, such as image generation [4, 5], video generation [6], text-to-image generation [7], speech synthesis [8, 9] and lossless compression [10]. DPMs are defined by discretetime random processes [1, 2] or continuous-time stochastic differential equations (SDEs) [3], which learn to gradually remove the noise added to the data points. Compared with the widely-used generative adversarial networks (GANs) [11] and variational auto-encoders (VAEs) [12], DPMs can not only compute exact likelihood [3], but also achieve even better sample quality for image generation [4]. However, to obtain high-quality samples, DPMs usually need hundreds or thousands of sequential steps of large neural network evaluations, thereby resulting in a much slower sampling speed than the single-step GANs or VAEs. Such inefficiency is becoming a critical bottleneck for the adoption of DPMs in downstream tasks, leading to an urgent request to design fast samplers for DPMs.
∗Corresponding Author.
2Code is available at https://github.com/LuChengTHU/dpm-solver
36th Conference on Neural Information Processing Systems (NeurIPS 2022).
arXiv:2206.00927v3 [cs.LG] 13 Oct 2022


NFE = 10 NFE = 15 NFE = 20 NFE = 100 NFE = 10
(a) DDIM [19] (b) DPM-Solver (ours)
Figure 1: Samples by DDIM [19] with 10, 15, 20, 100 number of function evaluations (NFE), and DPM-Solver (ours) with only 10 NFE, using the pre-trained DPMs on ImageNet 256×256 with classifier guidance [4].
Existing fast samplers for DPMs can be divided into two categories. The first category includes knowledge distillation [13, 14] and noise level or sample trajectory learning [15–18]. Such methods require a possibly expensive training stage before they can be used for efficient sampling. Furthermore, their applicability and flexibility might be limited. It might require nontrivial effort to adapt the method to different models, datasets, and number of sampling steps. The second category consists of training-free [19–21] samplers, which are suitable for all pre-trained DPMs in a simple plug-andplay manner. Training-free samplers include adopting implicit [19] or analytical [21] generation process, advanced differential equation (DE) solvers [3, 20, 22–24] and dynamic programming [18]. However, these methods still require ∼ 50 function evaluations [21] to generate high-quality samples (comparable to those generated by plain samplers in about 1000 function evaluations), thereby are still time-consuming.
In this work, we bring the efficiency of training-free samplers to a new level to produce high-quality samples in the “few-step sampling” regime, where the sampling can be done within around 10 steps of sequential function evaluations. We tackle the alternative problem of sampling from DPMs as solving the corresponding diffusion ordinary differential equations (ODEs) of DPMs, and carefully examine the structure of diffusion ODEs. Diffusion ODEs have a semi-linear structure — they consist of a linear function of the data variable and a nonlinear function parameterized by neural networks. Such structure is omitted in previous training-free samplers [3, 20], which directly use black-box DE solvers. To utilize the semi-linear structure, we derive an exact formulation of the solutions of diffusion ODEs by analytically computing the linear part of the solutions, avoiding the corresponding discretization error. Furthermore, by applying change-of-variable, the solutions can be equivalently simplified to an exponentially weighted integral of the neural network. Such integral is very special and can be efficiently approximated by the numerical methods for exponential integrators [25].
Based on our formulation of solutions, we propose DPM-Solver, a fast dedicated solver for diffusion ODEs by approximating the above integral. Specifically, we propose first-order, second-order and third-order versions of DPM-Solver with convergence order guarantees. We further propose an adaptive step size schedule for DPM-Solver. In general, DPM-Solver is applicable to both continuoustime and discrete-time DPMs, and also conditional sampling with classifier guidance [4]. Fig. 1 demonstrates the speedup performance of a Denoising Diffusion Implicit Models (DDIM) [19] baseline and DPM-Solver, which shows that DPM-Solver can generate high-quality samples with as few as 10 function evaluations and is much faster than DDIM on the ImageNet 256x256 dataset [26]. Our additional experimental results show that DPM-Solver can greatly improve the sampling speed of both discrete-time and continuous-time DPMs, and it can achieve excellent sample quality in around 10 function evaluations, which is much faster than all previous training-free samplers of DPMs.
2 Diffusion Probabilistic Models
We review diffusion probabilistic models and their associated differential equations in this section.
2


2.1 Forward Process and Diffusion SDEs
Assume that we have a D-dimensional random variable x0 ∈ RD with an unknown distribution q0(x0). Diffusion Probabilistic Models (DPMs) [1–3, 10] define a forward process {xt}t∈[0,T ] with T > 0 starting with x0, such that for any t ∈ [0, T ], the distribution of xt conditioned on x0 satisfies
q0t(xt|x0) = N (xt|α(t)x0, σ2(t)I), (2.1)
where α(t), σ(t) ∈ R+ are differentiable functions of t with bounded derivatives, and we denote them as αt, σt for simplicity. The choice for αt and σt is referred to as the noise schedule of a DPM. Let qt(xt) denote the marginal distribution of xt, DPMs choose noise schedules to ensure
that qT (xT ) ≈ N (xT |0, σ ̃2I) for some σ ̃ > 0, and the signal-to-noise-ratio (SNR) αt2/σt2 is strictly decreasing w.r.t. t [10]. Moreover, Kingma et al. [10] prove that the following stochastic differential equation (SDE) has the same transition distribution q0t(xt|x0) as in Eq. (2.1) for any t ∈ [0, T ]:
dxt = f (t)xtdt + g(t)dwt, x0 ∼ q0(x0), (2.2)
where wt ∈ RD is the standard Wiener process, and
f (t) = d log αt
dt , g2(t) = dσt2
dt − 2 d log αt
dt σ2
t . (2.3)
Under some regularity conditions, Song et al. [3] show that the forward process in Eq. (2.2) has an equivalent reverse process from time T to 0, starting with the marginal distribution qT (xT ):
dxt = [f (t)xt − g2(t)∇x log qt(xt)]dt + g(t)dw ̄t, xT ∼ qT (xT ), (2.4)
where w ̄t is a standard Wiener process in the reverse time. The only unknown term in Eq. (2.4) is the score function ∇x log qt(xt) at each time t. In practice, DPMs use a neural network θ(xt, t) parameterized by θ to estimate the scaled score function: −σt∇x log qt(xt). The parameter θ is optimized by minimizing the following objective [2, 3]:
L(θ; ω(t)) := 1
2
∫T
0
ω(t)Eqt(xt)
[
‖θ(xt, t) + σt∇x log qt(xt)‖2
2
]
dt
=1
2
∫T
0
ω(t)Eq0(x0)Eq()
[
‖θ(xt, t) − ‖2
2
]
dt + C,
where ω(t) is a weighting function,  ∼ q() = N (|0, I), xt = αtx0 + σt, and C is a constant independent of θ. As θ(xt, t) can also be regarded as predicting the Gaussian noise added to xt, it is usually called the noise prediction model. Since the ground truth of θ(xt, t) is −σt∇x log qt(xt), DPMs replace the score function in Eq. (2.4) by −θ(xt, t)/σt and define a parameterized reverse
process (diffusion SDE) from time T to 0, starting with xT ∼ N (0, σ ̃2I):
dxt =
[
f (t)xt + g2(t)
σt
θ(xt, t)
]
dt + g(t)dw ̄t, xT ∼ N (0, σ ̃2I). (2.5)
Samples can be generated from DPMs by solving the diffusion SDE in Eq. (2.5) with numerical solvers, which discretize the SDE from T to 0. Song et al. [3] proved that the traditional ancestral sampling method for DPMs [2] can be viewed as a first-order SDE solver for Eq. (2.5). However, these first-order methods usually need hundreds of or thousands of function evaluations to converge [3], leading to extremely slow sampling speed.
2.2 Diffusion (Probability Flow) ODEs
When discretizing SDEs, the step size is limited by the randomness of the Wiener process [27, Chap. 11]. A large step size (small number of steps) often causes non-convergence, especially in high dimensional spaces. For faster sampling, one can consider the associated probability flow ODE [3], which has the same marginal distribution at each time t as that of the SDE. Specifically, for DPMs, Song et al. [3] proved that the probability flow ODE of Eq. (2.4) is
dxt
dt = f (t)xt − 1
2 g2(t)∇x log qt(xt), xT ∼ qT (xT ), (2.6)
3


where the marginal distribution of xt is also qt(xt). By replacing the score function with the noise prediction model, Song et al. [3] defined the following parameterized ODE (diffusion ODE):
dxt
dt = hθ(xt, t) := f (t)xt + g2(t)
2σt
θ(xt, t), xT ∼ N (0, σ ̃2I). (2.7)
Samples can be drawn by solving the ODE from T to 0. Comparing with SDEs, ODEs can be solved with larger step sizes as they have no randomness. Furthermore, we can take advantage of efficient numerical ODE solvers to accelerate the sampling. Song et al. [3] used the RK45 ODE solver [28] for the diffusion ODEs, which generates samples in ∼ 60 function evaluations to reach comparable quality with a 1000-step SDE solver for Eq. (2.5) on the CIFAR-10 dataset [29]. However, existing general-purpose ODE solvers still cannot generate satisfactory samples in the few-step (∼ 10 steps) sampling regime. To the best of our knowledge, there is still a lack of training-free samplers for DPMs in the few-step sampling regime, and the sampling speed of DPMs is still a critical issue.
3 Customized Fast Solvers for Diffusion ODEs
As highlighted in Sec. 2.2, discretizing SDEs is generally difficult in high dimensions [27, Chap. 11] and it is hard to converge within few steps. In contrast, ODEs are easier to solve, yielding a potential for fast samplers. However, as mentioned in Sec. 2.2, the general black-box ODE solver used in previous work [3] empirically fails to converge in few steps. This motivates us to design a dedicated solver for diffusion ODEs to enable fast and high-quality few-step sampling. We start with a detailed investigation of the specific structure of diffusion ODEs.
3.1 Simplified Formulation of Exact Solutions of Diffusion ODEs
The key insight of this work is that given an initial value xs at time s > 0, the solution xt at each time t < s of diffusion ODEs in Eq. (2.7) can be simplified into a very special exact formulation which can be efficiently approximated.
Our first key observation is that a part of the solution xt can be exactly computed by considering the particular structure of diffusion ODEs. The r.h.s. of diffusion ODEs in Eq. (2.7) consists of two parts:
the part f (t)xt is a linear function of xt, and the other part g2(t)
2σt θ(xt, t) is generally a nonlinear function of xt because of the neural network θ(xt, t). This type of ODE is referred to as semi-linear ODE. The black-box ODE solvers adopted by previous work [3] are ignorant of this semi-linear structure as they take the whole hθ(xt, t) in Eq. (2.7) as the input, which causes discretization errors of both the linear and nonlinear term. We note that for semi-linear ODEs, the solution at time t can be exactly formulated by the “variation of constants” formula [30]:
xt = e
∫t
s f (τ )dτ xs +
∫t
s
(
e
∫t
τ f (r)dr g2(τ )
2στ
θ(xτ , τ )
)
dτ. (3.1)
This formulation decouples the linear part and the nonlinear part. In contrast to black-box ODE solvers, the linear part is now exactly computed, which eliminates the approximation error of the linear term. However, the integral of the nonlinear part is still complicated because it couples the coefficients about the noise schedule (i.e., f (τ ), g(τ ), στ ) and the complex neural network θ, which is still hard to approximate.
Our second key observation is that the integral of the nonlinear part can be greatly simplified by introducing a special variable. Let λt := log(αt/σt) (one half of the log-SNR), then λt is a strictly decreasing function of t (due to the definition of DPMs as discussed in Sec. 2.1). We can rewrite g(t) in Eq. (2.3) as
g2(t) = dσt2
dt − 2 d log αt
dt σ2
t = 2σ2
t
( d log σt
dt − d log αt
dt
)
= −2σ2
t
dλt
dt . (3.2)
Combining with f (t) = d log αt/dt in Eq. (2.3), we can rewrite Eq. (3.1) as
xt = αt
αs
xs − αt
∫t
s
( dλτ dτ
) στ
ατ
θ(xτ , τ )dτ. (3.3)
As λ(t) = λt is a strictly decreasing function of t, it has an inverse function tλ(·) satisfying t = tλ(λ(t)). We further change the subscripts of x and θ from t to λ and denote xˆλ := xtλ(λ), ˆθ(xˆλ, λ) := θ(xtλ(λ), tλ(λ)). Rewrite Eq. (3.3) by “change-of-variable” for λ, then we have:
4


Proposition 3.1 (Exact solution of diffusion ODEs). Given an initial value xs at time s > 0, the solution xt at time t ∈ [0, s] of diffusion ODEs in Eq. (2.7) is:
xt = αt
αs
xs − αt
∫ λt
λs
e−λˆθ(xˆλ, λ)dλ. (3.4)
We call the integral ∫ e−λˆθ(xˆλ, λ)dλ the exponentially weighted integral of ˆθ, which is very special and highly related to the exponential integrators in the literature of ODE solvers [25]. To the best of our knowledge, such formulation has not been revealed in prior work of diffusion models.
Eq. (3.4) provides a new perspective for approximating the solutions of diffusion ODEs. Specifically, given xs at time s, According to Eq. (3.4), approximating the solution at time t is equivalent to directly approximating the exponentially weighted integral of ˆθ from λs to λt, which avoids the error of the linear terms and is well-studied in the literature of exponential integrators [25, 31]. Based on this insight, we propose fast solvers for diffusion ODEs, as detailed in the following sections.
3.2 High-Order Solvers for Diffusion ODEs
In this section, we propose high-order solvers for diffusion ODEs with convergence order guarantee by leveraging our proposed solution formulation Eq. (3.4). The proposed solvers and analysis are highly motivated by the methods of exponential integrators [25, 31] in the ODE literature.
Specifically, given an initial value xT at time T and M + 1 time steps {ti}M
i=0 decreasing from t0 = T to tM = 0. Let x ̃t0 = xT be the initial value. The proposed solvers use M steps to iteratively
compute a sequence {x ̃ti }M
i=0 to approximate the true solutions at time steps {ti}M
i=0. In particular, the last iterate x ̃tM approximates the true solution at time 0.
In order to reduce the approximation error between x ̃tM and the true solution at time 0, we need to reduce the approximation error for each x ̃ti at every step [30]. Starting with the previous value x ̃ti−1 at time ti−1, according to Eq. (3.4), the exact solution xti−1→ti at time ti is given by
xti−1→ti = αti
αti−1
x ̃ti−1 − αti
∫ λti
λti−1
e−λˆθ(xˆλ, λ)dλ. (3.5)
Therefore, to compute the value x ̃ti for approximating xti−1→ti , we need to approximate the
exponentially weighted integral of ˆθ from λti−1 to λti . Denote hi := λti −λti−1 , and ˆ(n)
θ (xˆλ, λ) :=
dnˆθ (xˆλ,λ)
dλn as the n-th order total derivative of ˆθ(xˆλ, λ) w.r.t. λ. For k ≥ 1, the (k − 1)-th order
Taylor expansion of ˆθ(xˆλ, λ) w.r.t. λ at λti−1 is
ˆθ(xˆλ, λ) =
k−1
∑
n=0
(λ − λti−1 )n
n! ˆ(n)
θ (xˆλti−1 , λti−1 ) + O((λ − λti−1 )k),
Substituting the above Taylor expansion into Eq. (3.5) yields
xti−1→ti = αti
αti−1
x ̃ti−1 − αti
k−1
∑
n=0
ˆ(n)
θ (xˆλti−1 , λti−1 )
∫ λti
λti−1
e−λ (λ − λti−1 )n
n! dλ + O(hk+1
i ), (3.6)
where the integral ∫ e−λ (λ−λti−1 )n
n! dλ can be analytically computed by repeatedly applying n times
of integration-by-parts (see Appendix B.2). Therefore, to approximate xti−1→ti , we only need to
approximate the n-th order total derivatives ˆ(n)
θ (xˆλ, λ) for n ≤ k − 1, which is a well-studied
problem in the ODE literature [31, 32]. By dropping the O(hk+1
i ) error term and approximating the
first (k − 1)-th total derivatives with the “stiff order conditions” [31, 32], we can derive k-th-order ODE solvers for diffusion ODEs. We name such solvers as DPM-Solver overall, and DPM-Solver-k for a specific order k. Here we take k = 1 for demonstration. In this case, Eq. (3.6) becomes
xti−1→ti = αti
αti−1
x ̃t

=== Flow Matching for Generative Modeling (Lipman, Yaron; Chen, Ricky T. Q.; Ben-Hamu, Heli; Nickel, Maximilian; Le, Matt) ===
Preprint
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
where θ ∈ Rp are its learnable parameters, which in turn leads to a deep parametric model of the flow φt, called a Continuous Normalizing Flow (CNF). A CNF is used to reshape a simple prior density p0 (e.g., pure noise) to a more complicated one, p1, via the push-forward equation
pt = [φt]∗p0 (3)
where the push-forward (or change of variables) operator ∗ is defined by
[φt]∗p0(x) = p0(φ−1
t (x)) det
[ ∂φ−1
t
∂x (x)
]
. (4)
A vector field vt is said to generate a probability density path pt if its flow φt satisfies equation 3. One practical way to test if a vector field generates a probability path is using the continuity equation, which is a key component in our proofs, see Appendix B. We recap more information on CNFs, in particular how to compute the probability p1(x) at an arbitrary point x ∈ Rd in Appendix C.
3 FLOW MATCHING
Let x1 denote a random variable distributed according to some unknown data distribution q(x1). We assume we only have access to data samples from q(x1) but have no access to the density function itself. Furthermore, we let pt be a probability path such that p0 = p is a simple distribution, e.g., the standard normal distribution p(x) = N (x|0, I), and let p1 be approximately equal in distribution to q. We will later discuss how to construct such a path. The Flow Matching objective is then designed to match this target probability path, which will allow us to flow from p0 to p1.
1We use subscript to denote the time parameter, e.g., pt(x).
2


Preprint
Given a target probability density path pt(x) and a corresponding vector field ut(x), which generates pt(x), we define the Flow Matching (FM) objective as
L
FM(θ) = Et,pt(x)‖vt(x) − ut(x)‖2, (5)
where θ denotes the learnable parameters of the CNF vector field vt (as defined in Section 2), t ∼ U [0, 1] (uniform distribution), and x ∼ pt(x). Simply put, the FM loss regresses the vector field ut with a neural network vt. Upon reaching zero loss, the learned CNF model will generate pt(x).
Flow Matching is a simple and attractive objective, but na ̈ıvely on its own, it is intractable to use in practice since we have no prior knowledge for what an appropriate pt and ut are. There are many choices of probability paths that can satisfy p1(x) ≈ q(x), and more importantly, we generally don’t have access to a closed form ut that generates the desired pt. In this section, we show that we can construct both pt and ut using probability paths and vector fields that are only defined per sample, and an appropriate method of aggregation provides the desired pt and ut. Furthermore, this construction allows us to create a much more tractable objective for Flow Matching.
3.1 CONSTRUCTING pt, ut FROM CONDITIONAL PROBABILITY PATHS AND VECTOR FIELDS
A simple way to construct a target probability path is via a mixture of simpler probability paths: Given a particular data sample x1 we denote by pt(x|x1) a conditional probability path such that it satisfies p0(x|x1) = p(x) at time t = 0, and we design p1(x|x1) at t = 1 to be a distribution
concentrated around x = x1, e.g., p1(x|x1) = N (x|x1, σ2I), a normal distribution with x1 mean and a sufficiently small standard deviation σ > 0. Marginalizing the conditional probability paths over q(x1) give rise to the marginal probability path
pt(x) =
∫
pt(x|x1)q(x1)dx1, (6)
where in particular at time t = 1, the marginal probability p1 is a mixture distribution that closely approximates the data distribution q,
p1(x) =
∫
p1(x|x1)q(x1)dx1 ≈ q(x). (7)
Interestingly, we can also define a marginal vector field, by “marginalizing” over the conditional vector fields in the following sense (we assume pt(x) > 0 for all t and x):
ut(x) =
∫
ut(x|x1) pt(x|x1)q(x1)
pt(x) dx1, (8)
where ut(·|x1) : Rd → Rd is a conditional vector field that generates pt(·|x1). It may not seem apparent, but this way of aggregating the conditional vector fields actually results in the correct vector field for modeling the marginal probability path.
Our first key observation is this:
The marginal vector field (equation 8) generates the marginal probability path (equation 6).
This provides a surprising connection between the conditional VFs (those that generate conditional probability paths) and the marginal VF (those that generate the marginal probability path). This connection allows us to break down the unknown and intractable marginal VF into simpler conditional VFs, which are much simpler to define as these only depend on a single data sample. We formalize this in the following theorem.
Theorem 1. Given vector fields ut(x|x1) that generate conditional probability paths pt(x|x1), for any distribution q(x1), the marginal vector field ut in equation 8 generates the marginal probability path pt in equation 6, i.e., ut and pt satisfy the continuity equation (equation 26).
The full proofs for our theorems are all provided in Appendix A. Theorem 1 can also be derived from the Diffusion Mixture Representation Theorem in Peluchetti (2021) that provides a formula for the marginal drift and diffusion coefficients in diffusion SDEs.
3


Preprint
3.2 CONDITIONAL FLOW MATCHING
Unfortunately, due to the intractable integrals in the definitions of the marginal probability path and VF (equations 6 and 8), it is still intractable to compute ut, and consequently, intractable to na ̈ıvely compute an unbiased estimator of the original Flow Matching objective. Instead, we propose a simpler objective, which surprisingly will result in the same optima as the original objective. Specifically, we consider the Conditional Flow Matching (CFM) objective,
L
CFM(θ) = Et,q(x1),pt(x|x1)
∥
∥vt(x) − ut(x|x1)∥
∥
2, (9)
where t ∼ U [0, 1], x1 ∼ q(x1), and now x ∼ pt(x|x1). Unlike the FM objective, the CFM objective allows us to easily sample unbiased estimates as long as we can efficiently sample from pt(x|x1) and compute ut(x|x1), both of which can be easily done as they are defined on a per-sample basis. Our second key observation is therefore:
The FM (equation 5) and CFM (equation 9) objectives have identical gradients w.r.t. θ.
That is, optimizing the CFM objective is equivalent (in expectation) to optimizing the FM objective. Consequently, this allows us to train a CNF to generate the marginal probability path pt—which in particular, approximates the unknown data distribution q at t=1— without ever needing access to either the marginal probability path or the marginal vector field. We simply need to design suitable conditional probability paths and vector fields. We formalize this property in the following theorem.
Theorem 2. Assuming that pt(x) > 0 for all x ∈ Rd and t ∈ [0, 1], then, up to a constant independent of θ, LCFM and LFM are equal. Hence, ∇θLFM(θ) = ∇θLCFM(θ).
4 CONDITIONAL PROBABILITY PATHS AND VECTOR FIELDS
The Conditional Flow Matching objective works with any choice of conditional probability path and conditional vector fields. In this section, we discuss the construction of pt(x|x1) and ut(x|x1) for a general family of Gaussian conditional probability paths. Namely, we consider conditional probability paths of the form
pt(x|x1) = N (x | μt(x1), σt(x1)2I), (10)
where μ : [0, 1] × Rd → Rd is the time-dependent mean of the Gaussian distribution, while σ : [0, 1] × R → R>0 describes a time-dependent scalar standard deviation (std). We set μ0(x1) = 0 and σ0(x1) = 1, so that all conditional probability paths converge to the same standard Gaussian noise distribution at t = 0, p(x) = N (x|0, I). We then set μ1(x1) = x1 and σ1(x1) = σmin, which is set sufficiently small so that p1(x|x1) is a concentrated Gaussian distribution centered at x1.
There is an infinite number of vector fields that generate any particular probability path (e.g., by adding a divergence free component to the continuity equation, see equation 26), but the vast majority of these is due to the presence of components that leave the underlying distribution invariant—for instance, rotational components when the distribution is rotation-invariant—leading to unnecessary extra compute. We decide to use the simplest vector field corresponding to a canonical transformation for Gaussian distributions. Specifically, consider the flow (conditioned on x1)
ψt(x) = σt(x1)x + μt(x1). (11)
When x is distributed as a standard Gaussian, ψt(x) is the affine transformation that maps to a normally-distributed random variable with mean μt(x1) and std σt(x1). That is to say, according to equation 4, ψt pushes the noise distribution p0(x|x1) = p(x) to pt(x|x1), i.e.,
[ψt]∗ p(x) = pt(x|x1). (12)
This flow then provides a vector field that generates the conditional probability path:
d
dt ψt(x) = ut(ψt(x)|x1). (13)
Reparameterizing pt(x|x1) in terms of just x0 and plugging equation 13 in the CFM loss we get
L
CFM(θ) = Et,q(x1),p(x0)
∥ ∥
∥vt(ψt(x0)) − d
dt ψt(x0)
∥ ∥ ∥
2
. (14)
Since ψt is a simple (invertible) affine map we can use equation 13 to solve for ut in a closed form.
Let f ′ denote the derivative with respect to time, i.e., f ′ = d
dt f , for a time-dependent function f .
4


Preprint
Theorem 3. Let pt(x|x1) be a Gaussian probability path as in equation 10, and ψt its corresponding flow map as in equation 11. Then, the unique vector field that defines ψt has the form:
ut(x|x1) = σt′(x1)
σt(x1) (x − μt(x1)) + μ′
t(x1). (15)
Consequently, ut(x|x1) generates the Gaussian path pt(x|x1).
4.1 SPECIAL INSTANCES OF GAUSSIAN CONDITIONAL PROBABILITY PATHS
Our formulation is fully general for arbitrary functions μt(x1) and σt(x1), and we can set them to any differentiable function satisfying the desired boundary conditions. We first discuss the special cases that recover probability paths corresponding to previously-used diffusion processes. Since we directly work with probability paths, we can simply depart from reasoning about diffusion processes altogether. Therefore, in the second example below, we directly formulate a probability path based on the Wasserstein-2 optimal transport solution as an interesting instance.
Example I: Diffusion conditional VFs. Diffusion models start with data points and gradually add noise until it approximates pure noise. These can be formulated as stochastic processes, which have strict requirements in order to obtain closed form representation at arbitrary times t, resulting in Gaussian conditional probability paths pt(x|x1) with specific choices of mean μt(x1) and std σt(x1) (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020b). For example, the reversed (noise→data) Variance Exploding (VE) path has the form
pt(x) = N (x|x1, σ2
1−tI), (16)
where σt is an increasing function, σ0 = 0, and σ1  1. Next, equation 16 provides the choices of μt(x1) = x1 and σt(x1) = σ1−t. Plugging these into equation 15 of Theorem 3 we get
ut(x|x1) = − σ′1−t
σ1−t
(x − x1). (17)
The reversed (noise→data) Variance Preserving (VP) diffusion path has the form
pt(x|x1) = N (x | α1−tx1, (1 − α2
1−t
) I), where αt = e− 1
2 T (t), T (t) =
∫t
0
β(s)ds, (18)
and β is the noise scale function. Equation 18 provides the choices of μt(x1) = α1−tx1 and
σt(x1) =
√
1 − α12−t. Plugging these into equation 15 of Theorem 3 we get
ut(x|x1) = α′1−t
1 − α12−t
(α1−tx − x1) = − T ′(1 − t)
2
[
e−T (1−t)x − e− 1
2 T (1−t)x1
1 − e−T (1−t)
]
. (19)
Our construction of the conditional VF ut(x|x1) does in fact coincide with the vector field previously used in the deterministic probability flow (Song et al. (2020b), equation 13) when restricted to these conditional diffusion processes; see details in Appendix D. Nevertheless, combining the diffusion conditional VF with the Flow Matching objective offers an attractive training alternative—which we find to be more stable and robust in our experiments—to existing score matching approaches.
Another important observation is that, as these probability paths were previously derived as solutions of diffusion processes, they do not actually reach a true noise distribution in finite time. In practice, p0(x) is simply approximated by a suitable Gaussian distribution for sampling and likelihood evaluatio

=== Diffusion on the Probability Simplex (Floto, Griffin; Jonsson, Thorsteinn; Nica, Mihai; Sanner, Scott; Zhu, Eric Zheng) ===
Diffusion on the Probability Simplex
Griffin Floto 1 Thorsteinn Jonsson 1 Mihai Nica 2 Scott Sanner 3 Eric Zhengyu Zhu 3
Abstract
Diffusion models learn to reverse the progressive noising of a data distribution to create a generative model. However, the desired continuous nature of the noising process can be at odds with discrete data. To deal with this tension between continuous and discrete objects, we propose a method of performing diffusion on the probability simplex. Using the probability simplex naturally creates an interpretation where points correspond to categorical probability distributions. Our method uses the softmax function applied to an OrnsteinUnlenbeck Process, a well-known stochastic differential equation. We find that our methodology also naturally extends to include diffusion on the unit cube which has applications for bounded image generation.
1. Introduction
Diffusion models (Sohl-Dickstein et al., 2015) (Ho et al., 2020) (Song & Ermon, 2019) have emerged as a wellestablished class of generative models, finding applications in image (Dhariwal & Nichol, 2021), speech (Jeong et al., 2021), and video (Singer et al., 2022) domains. Diffusion processes work by progressively adding noise to data, which transforms a complex data distribution into a simpler, easyto-sample distribution. Diffusion models are used to reverse the noising process by learning a stochastic differential equation (SDE) parameterized by a neural network that generates the data distribution (Song et al., 2021).
In comparison to other popular methods, such as Generative Adversarial Networks (Goodfellow et al., 2014), diffusion models present a compelling advantage as they have an exact likelihood interpretation and do not require adversarial training that other state-of-the-art generative models require.
*Equal contribution 1EthicalAI 2Department of Mathematics and Statistics, University of Guelph 3Department of Computer Science, University of Toronto. Correspondence to: Griffin Floto <griffin@ethicalairesearch.com>.
Proceedings of the 40 th International Conference on Machine Learning, Honolulu, Hawaii, USA. PMLR 202, 2023. Copyright 2023 by the author(s).
That is, diffusion models enjoy the benefit of having a more stable training process that avoid non-overlapping data and generated distributions (Yang et al., 2023). Furthermore, diffusion models are also advantageous over discretized normalizing flows, which face practical restrictions when computing the determinant of the Jacobian from the change of variables formula (Chen et al., 2018).
Most work with diffusion models assume a continuous data distribution in Rn and noising is performed with Gaussian distributions. This presents a problem for discrete sampling: how would one add continuous Gaussian noise if the underlying categories are discrete? We propose the simple solution to perform diffusion by sampling from k categories on the probability simplex Sk := {x ∈ Rk : 0 ≤ xi ≤ 1, Pk
i=1 xi = 1}. The result of the diffusion is interpreted as the probability that a given category is chosen. By shifting from categories themselves, to the space of probabilities over categories, we effectively turn a discrete problem into a continuous one.
2. Background
2.1. Diffusion with Score-Matching
Score matching as formulated by (Song et al., 2021) considers a continuous time diffusion process. Typically, the forward process does not have parameters and is independent of the data distribution. In particular, the forward process is described by an SDE
dxt = f (xt, t)dt + G(xt, t)dwt (1)
where w is the standard Wiener process (also know as Brownian motion), f (·, t) : Rd → Rd is the drift term and G(·, t) : Rd → Rd×d is the diffusion coefficient. The process maps a data distribution, pt=0(xt) ∈ Rd into some limiting distribution pt=1(xt). The limiting distribution is chosen to be easy to sample from, and independent from the data distribution. Classical results in the theory of stochastic processes then tell us that the time reverse of this process is itself an SDE and obeys
1
arXiv:2309.02530v2 [cs.LG] 12 Sep 2023


Lifting Discrete Diffusion to the Probability Simplex
dxt = f (xt, t)dt − 1
2 ∇ · [G(xt, t)G(xt, t)⊤]dt
−1
2 G(xt, t)G(xt, t)⊤∇xlog pt(xt)dt + G(xt, t)dw ̄
(2)
where time now flows backwards from t = 1 to t = 0 and ∇ · F(x) := [∇ · f1(x), · · · , ∇ · fd(x)]⊤ for a matrixvalued function F(x) = [f1(x), · · · , fd(x)]⊤. The goal of diffusion models is to approximate the score ∇xlog pt(xt) and use the reverse SDE to sample from the generative model. The score can be approximated by sθ(xt, t) which provides the following objective
θ∗ = argminθEt∼U [0,1]Ex0∼p0(x)Ext∼p0t(xt|x0)
λ(t) ∥sθ(xt, t) − ∇xt log p0t(xt|x0)∥2
2
 (3)
where λ(t) is a weighting function and pst(xt|xs) is the transition kernel from x(s) to x(t). We note that a number of other objectives can be used to learn the score function (Song et al., 2021). A common practice when using diffusion models is to discretize time into uniform steps (Ho et al., 2020).
3. Method
3.1. The Logistic-Normal Distribution on the Probability Simplex
Recall the definition of the probability simplex Sk. We interpret points in the probability simplex as probability distributions over k categories.
The logistic-normal distribution is an example of a probability distribution over the probability simplex. It is defined as the probability distribution of a random variable whose multinomial logit is a normal distribution, (or equivalently it is the distribution of the softmax function applied to a Gaussian, see (5)). The probability density function of the logistic normal is
p(x; μ, Σ) = 1
|(2π)d−1Σ|
1
Qd
i=1 xi
exp − 1
2

log
  ̄xd
xd

−μ
⊤
Σ−1

log
  ̄xd
xd

−μ

!
(4)
where x ∈ Sd and x ̄d = [x1, . . . , xd−1]⊤. In the d = 2 dimensional case, the distribution can be understood as mapping a Gaussian distribution on R to [0, 1] via the sigmoid function.
Figure 1: Examples of the Logistic-Normal distribution (PDF values) on S3 with parameters μ = [0, 0], [0.2, 0.35] and σ = [0.5, 0, 5], [0.6, 0.8] respectively.
To constructively sample from this distrubution, we map a point y ∈ Rd−1 to a point in the probability simplex x ∈ Sd using the additive logistic transformation σ : Rd−1 → Sd defined by
xi = σi(y) :=

   
   
eyi
1 + Pd−1
k=1 eyk
, if i ∈ {1, . . . , d − 1}
1
1 + Pd−1
k=1 eyk
, if i = d
(5)
Where we note that 1 − Pd−1
i=1 xi = (1 + Pd−1
k=1 eyk )−1.
Conversely, the unique inverse map from Sd to Rd−1 is
yi = log
 xi
xd

, i ∈ {1, . . . , d − 1}.
3.2. The Ornstein-Unlenbeck Process
The Ornstein-Unlenbeck (OU) process is a real-valued stochastic process used in financial mathematics and physical sciences. Originally, it was developed to model the velocity of a Brownian particle under the force of friction. The process can be described by the following stochastic differential equation:
dYt = −θYtdt + σdWt
where θ > 0 and σ > 0 are parameters and Wt is the WWiener process. The distribution at time t of the process is given by a normal distribution
Yt
=d N

Y0e−θt, 1
2θ 1 − e−2θt I

.
In the limit as t → ∞ the process has a distribution of N 0, 1
2θ
, meaning that θ uniquely determines the limiting distribution.
2


Lifting Discrete Diffusion to the Probability Simplex
3.3. Diffusion on the Probability Simplex
Our main contribution is a novel diffusion process that operates on the probability simplex. Our method works by first defining the forward process by using the additive logistic transformation from equation 5 to map an OU process from Rd to Sd.
Xt = σ(Yt)
In our case we are able to get an exact solution for St by pushing forward the solution of the OU process, meaning that Xt ∼ σ N Y0e−θt, 1
2θ 1 − e−2θt. In other words, at each point t we have a closed form representation of the transition kernel pt0(xt|x0) which is a logistic Gaussian distribution that we can efficiently sample from. Moreover, one can obtain the SDE for Xt by applying Ito’s lemma to the SDE for Yt. Carrying this out (see appendix A.2) gives
dXt = f (Xt, t)dt + G(Xt, t)dWt (6)
where the diffusion coefficient matrix G can be written as:
Gij(x, t) =
(
xi(1 − xi), i = j
−xixj, i ̸= j
and the drift term f can be written as:
fi(x, t) = −θxi

(1 − xi)ai + X
j̸=i
xj aj


where aj = xj + 1
2 (1 − 2xj).
In order to train the score-matching model, we must also have a closed form solution of ∇xlog p(x)i, which we show in Appendix A.1. The results of the derivation is that the score of the logistic-normal distribution is
∇xlog p(x)i = − 1
v
1
xd
d−1
X
k=1
σμ
k (x) + 1
xd
σμ
i (x)
!
+ xi − xd
xixd
(7)
where we write σμ
k (x) = log
h xi xd
i
− μ. Finally, the calcu
lation for deriving ∇ · [G(xt, t)G(xt, t)⊤] is performed in Appendix A.3.
3.4. Implementation Considerations
An example application of this model is for modelling discrete data. A dataset with k different categories, can naturally be modelled with the simplex in Sk. The data distribution could then be represented as a linear combination
of Dirac delta functions centered at the corners of the simplex at t = 0. In other words, each data sample would correspond to a one-hot vector. In practice we relax this condition such that at the beginning of the forward process, data samples are mapped to vectors x = [α, β, · · · , β]⊤, where β = 1−α
d−2 . For example, a reasonable choice of α
would be 0.9 if k = 6.
During the optimization process, the score suffers from numerical instability in perimeter regions on the simplex. Furthermore, the region around the perimeter increases as the dimension of the simplex d grows. To deal with this problem, we notice that we directly predict the term −1
2 G(x, t)G(x, t)⊤∇xlog pt(x) from the reverse diffusion SDE.
0.0 0.2 0.4 0.6 0.8 1.0 x
−2
0
2
4
6
8
10
p(x)
∇ log p(x)
g2(x)∇ log p(x)
Figure 2: A comparison between the regular score, ∇xlog pt(x), and the reverse SDE term, g2(x, t)∇xlog pt(x), in the one-dimensional case. The reverse SDE term is bounded at the border of the interval [0, 1], unlike the score. The PDF of the logistic-normal distribution is plotted for clarity, along with a dotted line around the score for visual clarity.
4. Results
We present initial results of the Simplex Diffusion model using the MNIST dataset. We create a discrete version of the dataset which maps the pixel values that are typically in [0, 1, · · · , 255] to [0, 1, 2] for a total of k = 3 unique categories. In our experiments we use the following parameters: θ = 20, α = 0.9 and t ∈ [0.01, 0.25]. We parameterize the score function by a U-Net (Ronneberger et al., 2015) model with 35 million parameters.
3


Lifting Discrete Diffusion to the Probability Simplex
When samples are generated, they must be converted from vectors on the probability simplex, to one of k discrete categories. We choose to take the argmax of the sampled vectors to convert from points on the simplex to discrete categories. Qualitative results from this initial experiment can be found in Figure 3.
Figure 3: Random samples from a Simplex Diffusion model. Samples are taken at the beginning, middle and end of the reverse process and correspond to the top middle and bottom row respectively. Sampling is done with T = 1000 denoising steps
5. Discussion
Our methodology is related to recent works extending diffusion to the bounded domains of the probability simplex and the unit cube. In this section we compare these methodologies with our proposed model to highlight important differences.
5.1. Simplex Diffusion
Categorical SDEs with Simplex Diffusion (Richemond et al., 2022) use a diffusion process of Gamma random variables to sample from a Dirichlet distribution over the simplex. The Dirichlet distribution is an appealing choice as it is the conjugate prior of the categorical distribution . The forward process used is the Cox-Ingersoll-Ross process, which is
defined by the SDE dθ = b(a − θ)dt + σ√2bθdw, where θ(t = 0) ≥ 0 and a, b, σ > 0. A drawback of this approach is that while the process has a limiting distribution that is Dirichlet, this is not the case during the transient regime of the process dynamics.
Our proposed diffusion with the OU process and the LogitNormal distribution remains a Logistic-Normal distribution throughout the diffusion process due to the correspondence between diffusion spaces in Rd and Sd via Ito’s lemma.
5.2. Unit-Cube Diffusion
Reflected Diffusion (Lou & Ermon, 2023) is a method of performing diffusion on the unit cube [0, 1]d that is motivated by applications to pixel-based diffusion models. When image based diffusion models are used with Gaussian noise, sampling errors often compound and result in pixel values that are outside the valid data range of the unit cube. To mitigate this problem, thresholding is often performed to keep generated images to reasonable values via knowledge of the data distribution constraints (Ho et al., 2020) (Dhariwal & Nichol, 2021). While thresholding is popular in many image based diffusion models, it is theoretically unsound as there is a disconnect between the training and generative processes. The authors address this problem by using a reflected diffusion process that reflects particle trajectories into the interior of a data domain Ω that would normally extend outside the domain.
An interesting property of our Simplex Diffusion Model is that it can be naturally extended to higher dimensions by performing diffusion on the unit cube. By taking the product of d one-dimensional processes that we have developed, we create a diffusion process that is contained to the unit cube. A drawback of the Reflected Diffusion approach is that the resulting score from the forward process cannot be written in closed form. The authors use a combination of two approximations to apply their model in practice. On the other hand, our method maintains an closed form score function that is easy to implement.
6. Conclusion
We introduce a novel method to perform diffusion on the probability simplex and the unit cube. In both cases our method allows for an exact solution for the SDE dynamics, and fits into the common diffusion training paradigm.
Future work involves testing the method on more complex datasets and evaluating the properties on the categorical distribution. For example, if the entropy can be utilized as a natural notation of aleatoric uncertainty over generated values.
References
Chen, R. T. Q., Rubanova, Y., Bettencourt, J., and Duvenaud, D. K. Neural ordinary differential equations. In Advances in Neural Information Processing Systems, 2018.
Dhariwal, P. and Nichol, A. Diffusion models beat gans on image synthesis, 2021.
Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio,
4


Lifting Discrete Diffusion to the Probability Simplex
Y. Generative adversarial nets. In Advances in Neural Information Processing Systems, 2014.
Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems, 2020.
Jeong, M., Kim, H., Cheon, S. J., Choi, B. J., and Kim, N. S. Diff-tts: A denoising diffusion model for text-to-speech, 2021.
Lou, A. and Ermon, S. Reflected diffusion models, 2023.
Richemond, P. H., Dieleman, S., and Doucet, A. Categorical sdes with simplex diffusion, 2022.
Ronneberger, O., Fischer, P., and Brox, T. U-net: Convolutional networks for biomedical image segmentation. abs/1505.04597, 2015.
Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., Hu, Q., Yang, H., Ashual, O., Gafni, O., Parikh, D., Gupta, S., and Taigman, Y. Make-a-video: Text-to-video generation without text-video data, 2022.
Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics, 2015.
Song, Y. and Ermon, S. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 2019.
Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations, 2021.
Yang, L., Zhang, Z., Song, Y., Hong, S., Xu, R., Zhao, Y., Zhang, W., Cui, B., and Yang, M.-H. Diffusion models: A comprehensive survey of methods and applications, 2023.
5


Lifting Discrete Diffusion to the Probability Simplex
A. Mathematical calculations
A.1. Score Derivation
We want to calculate ∇xlog p(x) where
log p(x) = −log [Z] − log
"d Y
i−1
xi
#
−1
2v log
 x ̄d
xd

−μ
2
2
We first find the gradient of second term, given that the log normalizing constant doesn’t have a gradient.
α := −∇xlog
"d Y
i=1
xi
#
αi = − ∂
∂xi
d−1
X
i=1
log [xi] + log
"
a−
d−1
X
k=1
xk
#!
=−1
xi
+1
a − Pd−1
k=1 xk
=1
xd
−1
xi = xi − xd
xixd
Next, we deal with the exponential term:
β := − 1
2v ∇x log
 x ̄d
xd

−μ
2
2
βi = − 1
2v
∂ ∂xi
d−1
X
k=1

log
 xk
xd

−μ
2!
=− 1
2v
d−1
X
k=1
∂
∂u u2 ∂
∂xi
u

, u = log
 xk
xd

−μ
Working with κ := ∂
∂u u2 ∂
∂xi u we get
κ := ∂
∂u u2 ∂
∂xi
u
= 2u ∂
∂xi
log [xk] − ∂
∂xi
log
"
a−
d−1
X
k=1
xk
#!
= 2u

δik
1
xi
+1
xd

Combining terms again we get:
6


Lifting Discrete Diffusion to the Probability Simplex
βi = − 1
v
d−1
X
k=1

δik
1
xi
+1
xd

log
  ̄xd
xd

−μ

=− 1
vxd
d−1
X
k=1

log
 xk
xd

−μ

−1
vxi

log
 xi
xd

−μ

=− 1
vxd
d−1
X
k=1
γk
μ(x) − 1
vxi
γi
μ(x)
where we write γiμ(x) = log
h xi xd
i
−μ
For the final results, we must combine the α and β terms together to get:
∇xlog pa(x)i = − 1
vxd
d−1
X
k=1
γk
μ(x) − 1
vxi
γi
μ(x) + xi − xd
xixd
A.2. Sampling and Ito’s Lemma
We are working with an OU process of the following form:
dYt = −θYtdt + dBt
with a corresponding process on the simplex:
Xt = σ(Yt)
To keep this section self-contained the definition of σ is:
σi(y) = eyi
1 + Pd−1
k=1 eyk
, i ∈ {1, . . . , d − 1}
We must write Xt in a form where Xt = f (X, t)dt + G(X, t)dBt. This can be done via Ito’s Lemma:
dXi = −θ(∇X σi(X))⊤

=== Denoising Diffusion Probabilistic Models (Ho, Jonathan; Jain, Ajay; Abbeel, Pieter) ===
Denoising Diffusion Probabilistic Models

Jonathan Ho UC Berkeley jonathanho@berkeley.edu

Ajay Jain UC Berkeley ajayj@berkeley.edu

Pieter Abbeel UC Berkeley pabbeel@cs.berkeley.edu

Abstract
We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding. On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 and a state-of-the-art FID score of 3.17. On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN. Our implementation is available at https://github.com/hojonathanho/diffusion.
1 Introduction
Deep generative models of all kinds have recently exhibited high quality samples in a wide variety of data modalities. Generative adversarial networks (GANs), autoregressive models, ﬂows, and variational autoencoders (VAEs) have synthesized striking image and audio samples [14, 27, 3, 58, 38, 25, 10, 32, 44, 57, 26, 33, 45], and there have been remarkable advances in energy-based modeling and score matching that have produced images comparable to those of GANs [11, 55].

Figure 1: Generated samples on CelebA-HQ 256 × 256 (left) and unconditional CIFAR10 (right) 34th Conference on Neural Information Processing Systems (NeurIPS 2020), Vancouver, Canada.

xT
<latexit sha1_base64="l4LvSgM7PR7I/kkuy5soikK4gpU=">AAAEoXictVLditNAFE7XqGv92a5eejOYLexKLU0VFKRQ9EYvhCrb3YUklOlk2g6dnzBzYrcb8zK+lU/gazhJK6atuiB4YODM+T/n+8YJZwY6nW+1vRvuzVu39+/U7967/+CgcfjwzKhUEzokiit9McaGcibpEBhwepFoisWY0/Px/G3hP/9MtWFKnsIyoZHAU8kmjGCwplHjeygwzAjThNM4Kz/jSXaZj05zFHIlp5pNZ4C1VgsUkliB2TX/oQLYCpe/4rJwZhJM6NPMJyLPt9IM0SwBA0tOUaVGBs/8/J8mWVRH6eSjhtdpd0pBu4q/VjxnLYPR4d7XMFYkFVQC4diYwO8kEGVYA7P183qYGmr3meMpDawqsaAmykpEctS0lhhNlLZPAiqt1YwMC2OWYmwjiynNtq8w/s4XpDB5FWVMJilQSVaNJilHoFABL4qZpgT40irYntTOisgMa0zAkqC+0QbY/MquIfCcYssbsBH1UNIFUUJgGVePGfhR1qyj1YETXAaH/SqAnp836/lGftUfdNcFiqbBT8L2jouQdvE9iVAoVUyDWONFa5XVYlJSjezEPT+BlmCSiVQgw65or2vBaE0Y5z1e4D/VeBmhstwJyo5C0YeZ53vdo/z19lhVjly71+K6xRb/ZbO/rbLCS8HMwmVZ7W9zeFc567b95+3uxxde/82a3/vOY+eJc+z4zkun77xzBs7QIbUPNVP7Ustdz33vDtxPq9C92jrnkbMhbvAD81mObw==</latexit>

! · · · ! xt

p✓ (xt
<latexit sha1_base64="XVzP503G8Ma8Lkwk3KKGZcZJbZ0=">AAACEnicbVC7SgNBFJ2Nrxhfq5Y2g0FICsNuFEwZsLGMYB6QLMvsZDYZMvtg5q4Y1nyDjb9iY6GIrZWdf+Mk2SImHrhwOOde7r3HiwVXYFk/Rm5tfWNzK79d2Nnd2z8wD49aKkokZU0aiUh2PKKY4CFrAgfBOrFkJPAEa3uj66nfvmdS8Si8g3HMnIAMQu5zSkBLrlmO3R4MGZBSLyAw9Pz0YeKmcG5P8CNekKDsmkWrYs2AV4mdkSLK0HDN714/oknAQqCCKNW1rRiclEjgVLBJoZcoFhM6IgPW1TQkAVNOOntpgs+00sd+JHWFgGfq4kRKAqXGgac7p0eqZW8q/ud1E/BrTsrDOAEW0vkiPxEYIjzNB/e5ZBTEWBNCJde3YjokklDQKRZ0CPbyy6ukVa3YF5Xq7WWxXsviyKMTdIpKyEZXqI5uUAM1EUVP6AW9oXfj2Xg1PozPeWvOyGaO0R8YX7+bCp4F</latexit>

1|xt)
!

xt

1

! · · · ! x0

!

<latexit sha1_base64="7yFrn0YPyuP5dVIvc7Tl2zcbS/g=">AAAB+HicbVBNSwMxEJ2tX7V+dNWjl2ARPJXdKuix6MVjBfsB7VKyaXYbmk2WJKvU0l/ixYMiXv0p3vw3pu0etPXBwOO9GWbmhSln2njet1NYW9/Y3Cpul3Z29/bL7sFhS8tMEdokkkvVCbGmnAnaNMxw2kkVxUnIaTsc3cz89gNVmklxb8YpDRIcCxYxgo2V+m65x6WIFYuHBislH/tuxat6c6BV4uekAjkafferN5AkS6gwhGOtu76XmmCClWGE02mpl2maYjLCMe1aKnBCdTCZHz5Fp1YZoEgqW8Kgufp7YoITrcdJaDsTbIZ62ZuJ/3ndzERXwYSJNDNUkMWiKOPISDRLAQ2YosTwsSWYKGZvRWSIFSbGZlWyIfjLL6+SVq3qn1drdxeV+nUeRxGO4QTOwIdLqMMtNKAJBDJ4hld4c56cF+fd+Vi0Fpx85gj+wPn8AXOGk5o=</latexit>

q(xt|xt <latexit sha1_base64="eAZ87UuTmAQoJ4u19RGH5tA+bCI=">AAACC3icbVC7TgJBFJ31ifhatbSZQEywkOyiiZQkNpaYyCMBspkdZmHC7MOZu0ay0tv4KzYWGmPrD9j5N87CFgieZJIz59ybe+9xI8EVWNaPsbK6tr6xmdvKb+/s7u2bB4dNFcaSsgYNRSjbLlFM8IA1gINg7Ugy4ruCtdzRVeq37plUPAxuYRyxnk8GAfc4JaAlxyzclbo+gaHrJQ8TB/AjnvsmcGZPTh2zaJWtKfAysTNSRBnqjvnd7Yc09lkAVBClOrYVQS8hEjgVbJLvxopFhI7IgHU0DYjPVC+Z3jLBJ1rpYy+U+gWAp+p8R0J8pca+qyvTRdWil4r/eZ0YvGov4UEUAwvobJAXCwwhToPBfS4ZBTHWhFDJ9a6YDokkFHR8eR2CvXjyMmlWyvZ5uXJzUaxVszhy6BgVUAnZ6BLV0DWqowai6Am9oDf0bjwbr8aH8TkrXTGyniP0B8bXL+1hmu8=</latexit>

1)

Figure 2: The directed graphical model considered in this work.

This paper presents progress in diffusion probabilistic models [53]. A diffusion probabilistic model (which we will call a “diffusion model” for brevity) is a parameterized Markov chain trained using variational inference to produce samples matching the data after ﬁnite time. Transitions of this chain are learned to reverse a diffusion process, which is a Markov chain that gradually adds noise to the data in the opposite direction of sampling until signal is destroyed. When the diffusion consists of small amounts of Gaussian noise, it is sufﬁcient to set the sampling chain transitions to conditional Gaussians too, allowing for a particularly simple neural network parameterization.
Diffusion models are straightforward to deﬁne and efﬁcient to train, but to the best of our knowledge, there has been no demonstration that they are capable of generating high quality samples. We show that diffusion models actually are capable of generating high quality samples, sometimes better than the published results on other types of generative models (Section 4). In addition, we show that a certain parameterization of diffusion models reveals an equivalence with denoising score matching over multiple noise levels during training and with annealed Langevin dynamics during sampling (Section 3.2) [55, 61]. We obtained our best sample quality results using this parameterization (Section 4.2), so we consider this equivalence to be one of our primary contributions.
Despite their sample quality, our models do not have competitive log likelihoods compared to other likelihood-based models (our models do, however, have log likelihoods better than the large estimates annealed importance sampling has been reported to produce for energy based models and score matching [11, 55]). We ﬁnd that the majority of our models’ lossless codelengths are consumed to describe imperceptible image details (Section 4.3). We present a more reﬁned analysis of this phenomenon in the language of lossy compression, and we show that the sampling procedure of diffusion models is a type of progressive decoding that resembles autoregressive decoding along a bit ordering that vastly generalizes what is normally possible with autoregressive models.

2 Background

Diffusion models [53] are latent variable models of the form pθ(x0) := pθ(x0:T ) dx1:T , where x1, . . . , xT are latents of the same dimensionality as the data x0 ∼ q(x0). The joint distribution pθ(x0:T ) is called the reverse process, and it is deﬁned as a Markov chain with learned Gaussian transitions starting at p(xT ) = N (xT ; 0, I):

T

pθ(x0:T ) := p(xT ) pθ(xt−1|xt),
t=1

pθ(xt−1|xt) := N (xt−1; µθ(xt, t), Σθ(xt, t)) (1)

What distinguishes diffusion models from other types of latent variable models is that the approximate
posterior q(x1:T |x0), called the forward process or diffusion process, is ﬁxed to a Markov chain that gradually adds Gaussian noise to the data according to a variance schedule β1, . . . , βT :

T

q(x1:T |x0) := q(xt|xt−1), q(xt|xt−1) := N (xt; 1 − βtxt−1, βtI)

(2)

t=1

Training is performed by optimizing the usual variational bound on negative log likelihood:

E [− log pθ(x0)] ≤ Eq

−

log

pθ(x0:T ) q(x1:T |x0)

= Eq

−

log

p(xT

)

−

t≥1

log

pθ (xt−1 |xt ) q(xt|xt−1)

=: L (3)

The forward process variances βt can be learned by reparameterization [33] or held constant as

hyperparameters, and expressiveness of the reverse process is ensured in part by the choice of

Gaussian conditionals in pθ(xt−1|xt), because both processes have the same functional form when

βt are small [53]. A notable property of the forward process is that it admits sampling xt at an

arbitrary

timestep

t

in

closed

form:

using

the

notation √

αt

:=

1

−

βt

and

α¯t

:=

t s=1

αs,

we

have

q(xt|x0) = N (xt; α¯tx0, (1 − α¯t)I)

(4)

2

Efﬁcient training is therefore possible by optimizing random terms of L with stochastic gradient descent. Further improvements come from variance reduction by rewriting L (3) as:

Eq DKL(q(xT |x0) p(xT )) + DKL(q(xt−1|xt, x0) pθ(xt−1|xt)) − log pθ(x0|x1) (5)

LT

t>1

Lt−1

L0

(See Appendix A for details. The labels on the terms are used in Section 3.) Equation (5) uses KL

divergence to directly compare pθ(xt−1|xt) against forward process posteriors, which are tractable

when conditioned on x0:

q(xt−1|xt, where µ˜ t(xt,

x0) x0)

= :=

N√1(α¯x−tt−−α¯11βt; tµ˜xt0(x+t,

√x0α),t(β˜1t

I), − α¯t−1)

1 − α¯t

xt

and

β˜t

:=

1 − α¯t−1 1 − α¯t

βt

(6) (7)

Consequently, all KL divergences in Eq. (5) are comparisons between Gaussians, so they can be calculated in a Rao-Blackwellized fashion with closed form expressions instead of high variance Monte Carlo estimates.

3 Diffusion models and denoising autoencoders

Diffusion models might appear to be a restricted class of latent variable models, but they allow a large number of degrees of freedom in implementation. One must choose the variances βt of the forward process and the model architecture and Gaussian distribution parameterization of the reverse process. To guide our choices, we establish a new explicit connection between diffusion models and denoising score matching (Section 3.2) that leads to a simpliﬁed, weighted variational bound objective for diffusion models (Section 3.4). Ultimately, our model design is justiﬁed by simplicity and empirical results (Section 4). Our discussion is categorized by the terms of Eq. (5).

3.1 Forward process and LT
We ignore the fact that the forward process variances βt are learnable by reparameterization and instead ﬁx them to constants (see Section 4 for details). Thus, in our implementation, the approximate posterior q has no learnable parameters, so LT is a constant during training and can be ignored.

3.2 Reverse process and L1:T −1

Now we discuss we set Σθ(xt, t)

o=urσct2hIotioceusnitnrapinθe(xdtt−im1|exdt)ep=enNde(nxtt−co1n; µstθa(nxtst.,

t), Σθ(xt, t)) for 1 < Experimentally, both

t ≤ T . First, σt2 = βt and

σt2

=

β˜t

=

β 1−α¯t−1
1−α¯t t

had

similar

results.

The

ﬁrst

choice

is optimal for x0

∼

N (0, I),

and the

second is optimal for x0 deterministically set to one point. These are the two extreme choices

corresponding to upper and lower bounds on reverse process entropy for data with coordinatewise

unit variance [53].

Second, to represent following analysis of

the Lt.

mean With

pµθθ((xxtt−,1t|)x, tw)e=prNop(oxste−1a;sµpθe(cxiﬁt,ctp),aσrat2mI)e,twereizcaatinown rmitoe:tivated

by

the

Lt−1 = Eq

1 2σt2

µ˜ t(xt, x0) − µθ(xt, t) 2

+C

(8)

where C is a constant that does not depend on θ. So, we see that the most straightforward parameteri-

zation of µθ is Eq. (8) further

a model that predicts by reparameterizing

µ˜ t, Eq.

the (4)

forward process√posterior as xt(x0, ) = α¯tx0 +

m√ean. 1−

However, we can expand α¯t for ∼ N (0, I) and

applying the forward process posterior formula (7):

Lt−1 − C = Ex0,

1 2σt2

µ˜ t

xt(x0,

),

√1 α¯t

(xt(x0,

√ ) − 1 − α¯t

)

2
− µθ(xt(x0, ), t)

(9)

= Ex0,

1 2σt2

1 √αt

xt(x0,

)

−

√ βt 1−

α¯t

2
− µθ(xt(x0, ), t)

(10)

3

Algorithm 1 Training

1: repeat

2: x0 ∼ q(x0)

3: t ∼ Uniform({1, . . . , T })

4: ∼ N (0, I)

5:

Take gradient des√cent step o√n ∇θ − θ( α¯tx0 + 1 − α¯t

, t)

2

6: until converged

Algorithm 2 Sampling

1: xT ∼ N (0, I) 2: for t = T, . . . , 1 do 3: z ∼ N (0, I) if t > 1, else z = 0

4:

xt−1

=

√1 αt

5: end for

6: return x0

xt

−

√1−αt 1−α¯t

θ(xt, t)

+ σtz

Equation (10) reveals that µθ must predict √1αt

xt

−

√ βt 1−α¯t

given xt. Since xt is available as

input to the model, we may choose the parameterization

µθ(xt, t) = µ˜ t

xt,

√1 α¯t

(xt

−

√ 1

−

α¯t

θ (xt ))

= √1αt

xt

−

√ βt 1−

α¯t

θ(xt, t)

(11)

where θ is a function approximator intended to predict from xt. To sample xt−1 ∼ pθ(xt−1|xt) is

to compute xt−1 = √1αt

xt

−

√ βt 1−α¯t

θ(xt, t)

+ σtz, where z ∼ N (0, I). The complete sampling

procedure, Algorithm 2, resembles Langevin dynamics with θ as a learned gradient of the data

density. Furthermore, with the parameterization (11), Eq. (10) simpliﬁes to:

Ex0,

βt2 2σt2αt(1 − α¯t)

−

√

√

θ( α¯tx0 + 1 − α¯t

, t)

2

(12)

which resembles denoising score matching over multiple noise scales indexed by t [55]. As Eq. (12)

is equal to (one term of) the variational bound for the Langevin-like reverse process (11), we see

that optimizing an objective resembling denoising score matching is equivalent to using variational

inference to ﬁt the ﬁnite-time marginal of a sampling chain resembling Langevin dynamics.

To summarize, we can train the reverse process mean function approximator µθ to predict µ˜ t, or by modifying its parameterization, we can train it to predict . (There is also the possibility of predicting
x0, but we found this to lead to worse sample quality early in our experiments.) We have shown that the -prediction parameterization both resembles Langevin dynamics and simpliﬁes the diffusion
model’s variational bound to an objective that resembles denoising score matching. Nonetheless, it is just another parameterization of pθ(xt−1|xt), so we verify its effectiveness in Section 4 in an ablation where we compare predicting against predicting µ˜ t.

3.3 Data scaling, reverse process decoder, and L0

We assume that image data consists of integers in {0, 1, . . . , 255} scaled linearly to [−1, 1]. This

ensures that the neural network reverse process operates on consistently scaled inputs starting from

the standard normal prior p(xT ). To obtain discrete log likelihoods, we set the last term of the reverse process to an independent discrete decoder derived from the Gaussian N (x0; µθ(x1, 1), σ12I):

D

δ+ (xi0 )

pθ(x0|x1) =

N (x; µiθ(x1, 1), σ12) dx

i=1 δ−(xi0)

(13)

δ+(x) =

∞

x

+

1 255

if x = 1 if x < 1

δ−(x) =

−∞

x

−

1 255

if x = −1 if x > −1

where D is the data dimensionality and the i superscript indicates extraction of one coordinate.

(It would be straightforward to instead incorporate a more powerful decoder like a conditional

autoregressive model, but we leave that to future work.) Similar to the discretized continuous

distributions used in VAE decoders and autoregressive models [34, 52], our choice here ensures that

the variational bound is a lossless codelength of discrete data, without need of adding noise to the

data or incorporating the Jacobian of the scaling operation into the log likelihood. At the end of

sampling, we display µθ(x1, 1) noiselessly.

3.4 Simpliﬁed training objective
With the reverse process and decoder deﬁned above, the variational bound, consisting of terms derived from Eqs. (12) and (13), is clearly differentiable with respect to θ and is ready to be employed for

4

Table 1: CIFAR10 results. NLL measured in bits/dim.

Model
Conditional
EBM [11] JEM [17] BigGAN [3] StyleGAN2 + ADA (v1) [29]
Unconditional
Diffusion (original) [53] Gated PixelCNN [59] Sparse Transformer [7] PixelIQN [43] EBM [11] NCSNv2 [56] NCSN [55] SNGAN [39] SNGAN-DDLS [4] StyleGAN2 + ADA (v1) [29] Ours (L, ﬁxed isotropic Σ) Ours (Lsimple)

IS
8.30 8.76 9.22 10.06
4.60
5.29 6.78
8.87±0.12 8.22±0.05 9.09±0.10 9.74 ± 0.05 7.67±0.13 9.46±0.11

FID
37.9 38.4 14.73 2.67
65.93
49.46 38.2 31.75 25.32 21.7 15.42 3.26 13.51 3.17

NLL Test (Train)
Table 2: Unconditional CIFAR10 reverse process parameterization and training objective ablation. Blank entries were unstable to train and generated poor samples with out-ofrange scores.

Objective

IS

FID

≤ 5.40 3.03 (2.90)
2.80

µ˜ prediction (baseline)
L, learned diagonal Σ L, ﬁxed isotropic Σ
µ˜ − µ˜ θ 2
prediction (ours)

7.28±0.10 8.06±0.09
–

23.69 13.22
–

≤ 3.70 (3.69) ≤ 3.75 (3.72)

L, learned diagonal Σ
L, ﬁxed isotropic Σ ˜ − θ 2 (Lsimple)

– 7.67±0.13 9.46±0.11

– 13.51 3.17

training. However, we found it beneﬁcial to sample quality (and simpler to implement) to train on the following variant of the variational bound:

Lsimple(θ) := Et,x0,

−

√

√

θ( α¯tx0 + 1 − α¯t

, t)

2

(14)

where t is uniform between 1 and T . The t = 1 case corresponds to L0 with the integral in the discrete decoder deﬁnition (13) approximated by the Gaussian probability density function times the bin width, ignoring σ12 and edge effects. The t > 1 cases correspond to an unweighted version of Eq. (12), analogous to the loss weighting used by the NCSN denoising score matching model [55]. (LT does not appear because the forward process variances βt are ﬁxed.) Algorithm 1 displays the complete training procedure with this simpliﬁed objective.
Since our simpliﬁed objective (14) discards the weighting in Eq. (12), it is a weighted variational bound that emphasizes different aspects of reconstruction compared to the standard variational bound [18, 22]. In particular, our diffusion process setup in Section 4 causes the simpliﬁed objective to down-weight loss terms corresponding to small t. These terms train the network to denoise data with very small amounts of noise, so it is beneﬁcial to down-weight them so that the network can focus on more difﬁcult denoising tasks at larger t terms. We will see in our experiments that this reweighting leads to better sample quality.

4 Experiments
We set T = 1000 for all experiments so that the number of neural network evaluations needed during sampling matches previous work [53, 55]. We set the forward process variances to constants increasing linearly from β1 = 10−4 to βT = 0.02. These constants were chosen to be small relative to data scaled to [−1, 1], ensuring that reverse and forward processes have approximately the same functional form while keeping the signal-to-noise ratio at xT as small as possible (LT = DKL(q(xT |x0) N (0, I)) ≈ 10−5 bits per dimension in our experiments).
To represent the reverse process, we use a

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

=== One Step Diffusion via Shortcut Models (Abbeel, Pieter; Levine, Sergey; Frans, Kevin; Hafner, Danijar) ===
Under review as a conference paper at ICLR 2025
ONE STEP DIFFUSION VIA SHORTCUT MODELS
Anonymous authors
Paper under double-blind review
ABSTRACT
Diffusion models and flow-matching models have enabled generating diverse and realistic images by learning to transfer noise to data. However, sampling from these models involves iterative denoising over many neural network passes, making generation slow and expensive. Previous approaches for speeding up sampling require complex training regimes, such as multiple training phases, multiple networks, or fragile scheduling. We introduce shortcut models, a family of generative models that use a single network and training phase to produce high-quality samples in a single or multiple sampling steps. Shortcut models condition the network not only on the current noise level but also on the desired step size, allowing the model to skip ahead in the generation process. Across a wide range of sampling step budgets, shortcut models consistently produce higher quality samples than previous approaches, such as consistency models and reflow. Compared to distillation, shortcut models reduce complexity to a single network and training phase and additionally allow varying step budgets at inference time.
1 INTRODUCTION
Iterative denoising methods such as diffusion (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020) and flow-matching (Lipman et al., 2022; Liu et al., 2022) have seen remarkable success in modelling diverse images (Rombach et al., 2022; Esser et al., 2024), video (Ho et al., 2022; BarTal et al., 2024), audio (Kong et al., 2020), and proteins (Abramson et al., 2024). Yet, their weakness lies in expensive inference. Despite producing high-quality samples, these methods require an iterative inference procedure—often requiring dozens to hundreds of forward passes of the neural network—making generation slow and expensive. We posit that there exists a generative modelling objective which retains the benefits of diffusion training, yet can denoise in a single step.
Flow Matching Shortcut Models (ours)
128 Steps
Four Steps
One Step
Figure 1: Generations of flow-matching models and shortcut models for different inference budgets. Shortcut models generate high-quality images across a wide range of inference budgets, including using a single forward pass, drastically reducing sampling time compared to diffusion and flow-matching models. With too few steps, diffusion and flow-matching models predict the dataset mean. The same starting noise used within each column and two models are trained on CelebA-HQ and Imagenet-256 (class conditioned).
1


Under review as a conference paper at ICLR 2025
We consider the end-to-end setting, in which one-step denoising is acquired by a single model over a single training run. Closely related are previous two-stage methods which take existing diffusion models and later distill one-step capabilities into them. These stages introduce complexity and require either generating a large synthetic dataset (Luhman & Luhman, 2021; Liu et al., 2022) or propagating through a series of teacher and student networks (Ho et al., 2020; Meng et al., 2023). Consistency models (Song et al., 2023) are step closer to the end-to-end setting, but their dependency on large amounts of bootstrapping requires a careful learning schedule throughout training. Twostage or tightly-scheduled procedures suffer from a need to specify when to end training and begin distillation. In contrast, end-to-end methods can be trained indefinitely to continually improve.
We present shortcut models, a class of end-to-end generative models that produce high-quality generations under any inference budget, including in a single sampling step. Our key insight is to condition the neural network not only on the noise level but also the desired step size, enabling it to accurately jump ahead in the denoising process. Shortcut models can be seen as performing selfdistillation during training time, and thus do not require a separate distillation step and are trained over a single run. No schedules or careful warmups are necessary. Shortcut models are efficient to train, requiring only ∼ 16% more compute than that of a base diffusion model.
Empirical evaluations display that shortcut models satisfy a number of useful desiderata. On the commonly used CelebA-HQ and Imagenet-256 benchmarks, a single shortcut model can handle many-step, few-step, and one-step generation. Accuracy is not sacrificed —- in fact, many-step generation quality matches those of baseline diffusion models. At the same time, shortcut models can consistently match or outperform two-stage distillation methods in the few- and one-step settings.
The key contributions of this paper are summarized as follows:
• We introduce shortcut models, a class of generative models that generate high-quality samples in a single forward pass, by conditioning the model on the desired step size. Unlike distillation or consistency models, shortcut models are trained in a single training run without a schedule. • We perform a comprehensive comparison of shortcut models to previous diffusion and flowmatching approaches on CelebAHQ-256 and ImageNet-256 under fixed architecture and compute. Shortcut models match or exceed the distillation methods that require multiple training phases and significantly outperform previous end-to-end methods across inference budgets. • To demonstrate the generality of shortcut models beyond image generation, we apply them to robotic control and replace diffusion policies with shortcut policies. We observe that shortcut models maintain comparable performance under an order-of-magnitude lower inference cost. • We release model checkpoints and the full training code for replicating our experimental results: https://anonymous.4open.science/r/shortcut-BDFB/
2 BACKGROUND
Diffusion and flow-matching. A recent family of models, including diffusion (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020) and flow-matching1 (Lipman et al., 2022; Liu et al., 2022) models, approach the generative modelling problem by learning an ordinary differential equation (ODE) that transforms noise into data. In this work, we adopt the optimal transport flow-matching objective (Liu et al., 2022) for simplicity. We define xt as a linear interpolation between a data point x1 ∼ D and a noise point x0 ∼ N (0, I) of the same dimensionality. The velocity vt is the direction from the noise to the data point:
xt = (1 − t) x0 + t x1 and vt = x1 − x0. (1)
Given x0 and x1, the velocity vt is fully determined. But given only xt, there are multiple plausible pairs (x0, x1) and thus different values the velocity can take on, rendering vt a random variable. Flow models learn a neural network to estimate the expected value v ̄t = E[vt | xt] that averages over all plausible velocities at xt. The flow model can be optimized by regressing the empirical velocity of randomly sampled pairings of noise x0 and data x1 pairs:
v ̄θ(xt, t) ≈ Ex0,x1∼D [vt | xt] LF(θ) = Ex0,x1∼D
||v ̄θ(xt, t) − (x1 − x0)||2 (2)
1We consider flow-matching as a special case of diffusion modelling (Kingma & Gao, 2024), and use the terms interchangeably.
2


Under review as a conference paper at ICLR 2025
data
noise
Training pairings Learned ODE 4 Steps 2 Steps 1 Step
Flow Matching Training Flow Matching Inference
Figure 2: Naive diffusion and flow-matching models fail at few-step generation. Left: Training paths are created by randomly pairing data and noise. Note that the paths overlap; there is inherent uncertainty about the direction vt to the data point, given only xt. Right: While flow-matching models learn a deterministic ODE, its paths are not straight and have to be followed closely. The predicted directions vt point towards the average of plausible data points. The fewer inference steps, the more the generations are biased towards the dataset mean, causing them to go off track. At the first sampling step, the model points towards the dataset mean and thus cannot generate multi-modal data in a single step (see red circles).
To sample from a flow model, a noise point x0 is first sampled from the normal distribution. This point is then iteratively updated from x0 to x1 following the the denoising ODE defined as following the learned flow model v ̄θ(xt, t). In practice, this process is approximated using Euler sampling over small discrete time intervals.
Few-step ambiguity. While a perfectly trained ODE deterministically maps the noise distribution to the data distribution in continuous time, this guarantee is lost under finite step sizes. As illustrated in Figure 2, flow-matching learns to predict the average direction from xt towards the data, so following the prediction with a large step size will jump to an average of multiple data points. At t = 0 the model receives pure noise as input and (x0, x1) are randomly paired during training, so the predicted velocity at t = 0 points towards the dataset mean. Thus, even at the optimum of the flow matching objective, one step generation will fail for any multi-modal data distribution.
3 SHORTCUT MODELS FOR FEW STEP GENERATION
We introduce shortcut models, a new family of denoising generative models that overcomes the large number of sampling steps required by diffusion and flow-matching models. Our key intuition is that we can train a single model that supports different sampling budgets, by conditioning the model not only on the timestep t but also on a desired step size d.
As shown in Figure 2, flow-matching learns an ODE that maps noise to data along curved paths. Naively taking large sampling steps leads to large discretization error and in the single-step case, to catastrophic failure. Conditioning on d allows shortcut models to account for the future curvature, and jump to the correct next point rather than going off track. We refer to the normalized direction from xt towards the correct next point x′
t+d as the shortcut s(xt, t, d):
x′
t+d = xt + s(xt, t, d) d. (3)
Our aim is to train a shortcut model sθ(xt, t, d) to learn the shortcut for all combinations of xt, t, and d. Shortcut models can thus be seen as a generalization of flow-matching models to larger step sizes: whereas flow-matching models only learn the instantaneous velocity, shortcut models additionally learn to make larger jumps. At d → 0, the shortcut is equivalent to the flow.
A naive way to compute targets for training sθ(xt, t, d) would be to fully simulate the ODE forward with a small enough step size (Luhman & Luhman, 2021; Liu et al., 2022). However, this approach is computationally expensive, especially for end-to-end training. Instead, we leverage an inherent self-consistency property of shortcut models, namely that one shortcut step equals two consecutive shortcut steps of half the size:
s(xt, t, 2d) = s(xt, t, d)/2 + s(x′
t+d, t, d)/2 (4)
This allows us to train shortcut models using self-consistency targets for d > 0 and using the flowmatching loss (Equation 2) as a base case for d = 0. In principle, we can train the model on any
3


Under review as a conference paper at ICLR 2025
a) Diffusion / Flow Matching
b) Shortcut Models
Regress
Train towards two smaller steps
Figure 3: Overview of shortcut model training. At d ≈ 0, the shortcut objective is equivalent to the flow-matching objective, and can be trained by regressing onto empirical E[vt|xt] samples. Targets for larger d shortcuts are constructed by concatenating a sequence of two d/2 shortcuts. Both objectives can be trained jointly; shortcut models do not require a two-stage procedure or discretization schedule.
distribution of d ∼ p(d). In practice, we split the batch into a fraction that is trained with d = 0 and another fraction with randomly sampled d > 0 targets. We thus arrive at the combined shortcut model loss function:
LS(θ) = Ex0∼N, x1∼D, (t,d)∼p(t,d)
h
∥sθ(xt, t, 0) − (x1 − x0)∥2
| {z }
Flow-Matching
+ ∥sθ(xt, t, 2d) − starget∥2
| {z }
Self-Consistency
i
,
where starget = sθ(xt, t, d)/2 + sθ(x′
t+d, t, d)/2 and x′
t+d = xt + sθ(xt, t, d)d.
(5)
Intuitively, the above objective learns a mapping from noise to data which is consistent when queried under any sequence of step sizes, including directly in a single step. The flow-matching portion of the objective grounds the shortcut model at small step size to match empirical velocity samples. This ensures that the shortcut model develops a base generation capability when queried with many steps, exactly as an equivalent flow-matching model does. In the self-consistency portion, appropriate targets for larger step-sizes are constructed by concatenating a sequence of two smaller shortcuts. This propagates the generation capability from multi-step to few-step to one-step. The combined objective can be trained jointly, using a single model and over a single end-to-end training run.
3.1 TRAINING DETAILS
We now present a simple framework for training shortcut models via the objective described above. At each stage, we opt for design decisions which encourage training stability and simplicity.
Regressing onto empirical samples. As d → 0, the shortcut is equivalent to instantaneous flow. Thus, we can train the shortcut model at d = 0 using the loss given by Equation 2, i.e. by sampling random (x0, x1) pairs and fitting the expectation over vt. This term can be seen as grounding the small-step shortcuts to match the data denoising ODE. We find that sampling t ∼ U (0, 1) uniformly is the simplest and works as well as any other sampling scheme.
Enforcing self-consistency. Given that the shortcut model is accurate at small step-size, our next goal is to ensure that the shortcut model maintains this behavior at larger step-size. We rely on self-generated bootstrap targets for this purpose. To limit compounding approximation error, it is desirable to limit the total length of the bootstrap paths. We therefore opt for a binary recursive formulation in which two shortcuts are used to construct a twice-as-large shortcut (Figure 3).
We must decide on a number of steps M to represent the smallest unit of time for approximating the ODE; we use 128 in our experiments. This creates log2(128) + 1 = 8 possible shortcut lengths according to d ∈ (1/128, 1/64 ... 1/2, 1). During each training step, we sample xt, t, and a random d < 1, then take two sequential steps with the shortcut model. The concatenation of these two steps is then used as the target to train the model at 2d.
Note that the second step is queried at x′
t+d under the denoising ODE and not the empirical data
pairing, i.e. it is constructed by adding the predicted first shortcut to xt, and not by interpolating
4


Under review as a conference paper at ICLR 2025
Algorithm 1 Shortcut Model Training
while not converged do
x0 ∼ N (0, I), x1 ∼ D, (d, t) ∼ p(d, t)
xt ← (1 − t) x0 + t x1 Corrupt data point
for first k batch elements do starget ← x1 − x0 Flow-matching target d ←0
for other batch elements do st ← sθ(xt, t, d) First small step xt+d ← xt + st d Follow ODE st+d ← sθ(xt+d, t + d, d) Second small step starget ← stopgrad(st + st+d)/2 Self-consistency target
θ ← ∇θ||sθ(xt, t, 2d) − starget||2
Algorithm 2 Sampling
x ∼ N (0, I) d ← 1/M t ←0 for n ∈ [0, . . . , M − 1] do x ← x + sθ(x, t, d) d t ←t+d return x
towards x1 from the dataset. When d is at the smallest value (e.g. 1/128), we instead query the model at d = 0.
Joint optimization. Equation 5 consists of an empirical flow-matching objective and a selfconsistency objective, which are jointly optimized during training. The variance of the empirical term is much higher, as it regresses onto random noise pairings with inherent uncertainty, whereas the self-consistency term uses deterministic bootstrap targets. We found it helpful to construct a batch with significantly more empirical targets than self-consistency targets.
The above behavior also gives us room for computational efficiency. Training requires less selfconsistency targets than empirical targets, and self-consistency targets are also more expensive to generate (requiring two additional forward passes). We can therefore construct a training batch by combining a ratio of 1 − k empirical targets with k self-consistency targets. We find k = (1/4) to be reasonable. In this way, we can reduce the training cost of a shortcut model to be roughly only ∼ 16% more than that of an equivalent diffusion model2.
Guidance. Classifier-free guidance (CFG; Ho & Salimans, 2022) has proven to be an essential tool for diffusion models to reach high generation fidelity. CFG provides a linear approximation of a tradeoff between the class-conditional and -unconditional denoising ODE. We find that CFG helps at small step sizes but is error-prone at larger steps when linear approximation is not appropriate. We therefore use CFG when evaluating the shortcut model at d = 0 but forgo it elsewhere. A limitation of CFG in shortcut models is that the CFG scale must be specified before training.
Exponential moving average weights. Many recent diffusion models use an exponential moving average (EMA) over weight parameters to improve sample quality. EMA induces a smoothing effect on the generations, which is especially helpful in the in diffusion modelling since the objective has inherent variance. We find that similarly in shortcut models, variance from loss at the d = 0 level can result in large oscillations in the output at d = 1. Utilizing EMA parameters for generating self-consistency targets alleviates this issue.
Weight decay. We find that weight decay is crucial for enabling stability, especially early on in training. When the shortcut model is near initialization, the self-consistency targets it generates are largely noise. The model can latch on to these incoherent targets, resulting in artifacting and bad feature learning. We find that proper weight decay causes these issues to disappear, and enables us to bypass the need for discretization sch

=== Denoising Diffusion Implicit Models (Ermon, Stefano; Meng, Chenlin; Song, Jiaming) ===
Published as a conference paper at ICLR 2021
DENOISING DIFFUSION IMPLICIT MODELS
Jiaming Song, Chenlin Meng & Stefano Ermon Stanford University
{tsong,chenlin,ermon}@cs.stanford.edu
ABSTRACT
Denoising diffusion probabilistic models (DDPMs) have achieved high quality image generation without adversarial training, yet they require simulating a Markov chain for many steps in order to produce a sample. To accelerate sampling, we present denoising diffusion implicit models (DDIMs), a more efficient class of iterative implicit probabilistic models with the same training procedure as DDPMs. In DDPMs, the generative process is defined as the reverse of a particular Markovian diffusion process. We generalize DDPMs via a class of non-Markovian diffusion processes that lead to the same training objective. These non-Markovian processes can correspond to generative processes that are deterministic, giving rise to implicit models that produce high quality samples much faster. We empirically demonstrate that DDIMs can produce high quality samples 10× to 50× faster in terms of wall-clock time compared to DDPMs, allow us to trade off computation for sample quality, perform semantically meaningful image interpolation directly in the latent space, and reconstruct observations with very low error.
1 INTRODUCTION
Deep generative models have demonstrated the ability to produce high quality samples in many domains (Karras et al., 2020; van den Oord et al., 2016a). In terms of image generation, generative adversarial networks (GANs, Goodfellow et al. (2014)) currently exhibits higher sample quality than likelihood-based methods such as variational autoencoders (Kingma & Welling, 2013), autoregressive models (van den Oord et al., 2016b) and normalizing flows (Rezende & Mohamed, 2015; Dinh et al., 2016). However, GANs require very specific choices in optimization and architectures in order to stabilize training (Arjovsky et al., 2017; Gulrajani et al., 2017; Karras et al., 2018; Brock et al., 2018), and could fail to cover modes of the data distribution (Zhao et al., 2018).
Recent works on iterative generative models (Bengio et al., 2014), such as denoising diffusion probabilistic models (DDPM, Ho et al. (2020)) and noise conditional score networks (NCSN, Song & Ermon (2019)) have demonstrated the ability to produce samples comparable to that of GANs, without having to perform adversarial training. To achieve this, many denoising autoencoding models are trained to denoise samples corrupted by various levels of Gaussian noise. Samples are then produced by a Markov chain which, starting from white noise, progressively denoises it into an image. This generative Markov Chain process is either based on Langevin dynamics (Song & Ermon, 2019) or obtained by reversing a forward diffusion process that progressively turns an image into noise (Sohl-Dickstein et al., 2015).
A critical drawback of these models is that they require many iterations to produce a high quality sample. For DDPMs, this is because that the generative process (from noise to data) approximates the reverse of the forward diffusion process (from data to noise), which could have thousands of steps; iterating over all the steps is required to produce a single sample, which is much slower compared to GANs, which only needs one pass through a network. For example, it takes around 20 hours to sample 50k images of size 32 × 32 from a DDPM, but less than a minute to do so from a GAN on a Nvidia 2080 Ti GPU. This becomes more problematic for larger images as sampling 50k images of size 256 × 256 could take nearly 1000 hours on the same GPU.
To close this efficiency gap between DDPMs and GANs, we present denoising diffusion implicit models (DDIMs). DDIMs are implicit probabilistic models (Mohamed & Lakshminarayanan, 2016) and are closely related to DDPMs, in the sense that they are trained with the same objective function.
1
arXiv:2010.02502v4 [cs.LG] 5 Oct 2022


Published as a conference paper at ICLR 2021
Figure 1: Graphical models for diffusion (left) and non-Markovian (right) inference models.
In Section 3, we generalize the forward diffusion process used by DDPMs, which is Markovian, to non-Markovian ones, for which we are still able to design suitable reverse generative Markov chains. We show that the resulting variational training objectives have a shared surrogate objective, which is exactly the objective used to train DDPM. Therefore, we can freely choose from a large family of generative models using the same neural network simply by choosing a different, nonMarkovian diffusion process (Section 4.1) and the corresponding reverse generative Markov Chain. In particular, we are able to use non-Markovian diffusion processes which lead to ”short” generative Markov chains (Section 4.2) that can be simulated in a small number of steps. This can massively increase sample efficiency only at a minor cost in sample quality.
In Section 5, we demonstrate several empirical benefits of DDIMs over DDPMs. First, DDIMs have superior sample generation quality compared to DDPMs, when we accelerate sampling by 10× to 100× using our proposed method. Second, DDIM samples have the following “consistency” property, which does not hold for DDPMs: if we start with the same initial latent variable and generate several samples with Markov chains of various lengths, these samples would have similar high-level features. Third, because of “consistency” in DDIMs, we can perform semantically meaningful image interpolation by manipulating the initial latent variable in DDIMs, unlike DDPMs which interpolates near the image space due to the stochastic generative process.
2 BACKGROUND
Given samples from a data distribution q(x0), we are interested in learning a model distribution pθ(x0) that approximates q(x0) and is easy to sample from. Denoising diffusion probabilistic models (DDPMs, Sohl-Dickstein et al. (2015); Ho et al. (2020)) are latent variable models of the form
pθ(x0) =
∫
pθ(x0:T )dx1:T , where pθ(x0:T ) := pθ(xT )
T
∏
t=1
p(t)
θ (xt−1|xt) (1)
where x1, . . . , xT are latent variables in the same sample space as x0 (denoted as X ). The parameters θ are learned to fit the data distribution q(x0) by maximizing a variational lower bound:
mθax Eq(x0)[log pθ(x0)] ≤ mθax Eq(x0,x1,...,xT ) [log pθ(x0:T ) − log q(x1:T |x0)] (2)
where q(x1:T |x0) is some inference distribution over the latent variables. Unlike typical latent variable models (such as the variational autoencoder (Rezende et al., 2014)), DDPMs are learned with a fixed (rather than trainable) inference procedure q(x1:T |x0), and latent variables are relatively high dimensional. For example, Ho et al. (2020) considered the following Markov chain with Gaussian transitions parameterized by a decreasing sequence α1:T ∈ (0, 1]T :
q(x1:T |x0) :=
T
∏
t=1
q(xt|xt−1), where q(xt|xt−1) := N
(√ αt
αt−1
xt−1,
(
1 − αt
αt−1
)
I
)
(3)
where the covariance matrix is ensured to have positive terms on its diagonal. This is called the forward process due to the autoregressive nature of the sampling procedure (from x0 to xT ). We call the latent variable model pθ(x0:T ), which is a Markov chain that samples from xT to x0, the generative process, since it approximates the intractable reverse process q(xt−1|xt). Intuitively, the forward process progressively adds noise to the observation x0, whereas the generative process progressively denoises a noisy observation (Figure 1, left).
A special property of the forward process is that
q(xt|x0) :=
∫
q(x1:t|x0)dx1:(t−1) = N (xt; √αtx0, (1 − αt)I);
2


Published as a conference paper at ICLR 2021
so we can express xt as a linear combination of x0 and a noise variable :
xt = √αtx0 + √1 − αt, where  ∼ N (0, I). (4)
When we set αT sufficiently close to 0, q(xT |x0) converges to a standard Gaussian for all x0, so it is natural to set pθ(xT ) := N (0, I). If all the conditionals are modeled as Gaussians with trainable
mean functions and fixed variances, the objective in Eq. (2) can be simplified to1:
Lγ (θ) :=
T
∑
t=1
γtEx0∼q(x0),t∼N (0,I)
[
‖(t)
θ (√αtx0 + √1 − αtt) − t‖2
2
]
(5)
where θ := {(t)
θ }tT=1 is a set of T functions, each (t)
θ : X → X (indexed by t) is a function with trainable parameters θ(t), and γ := [γ1, . . . , γT ] is a vector of positive coefficients in the objective that depends on α1:T . In Ho et al. (2020), the objective with γ = 1 is optimized instead to maximize generation performance of the trained model; this is also the same objective used in noise conditional score networks (Song & Ermon, 2019) based on score matching (Hyva ̈rinen, 2005; Vincent, 2011). From a trained model, x0 is sampled by first sampling xT from the prior pθ(xT ), and then sampling xt−1 from the generative processes iteratively.
The length T of the forward process is an important hyperparameter in DDPMs. From a variational perspective, a large T allows the reverse process to be close to a Gaussian (Sohl-Dickstein et al., 2015), so that the generative process modeled with Gaussian conditional distributions becomes a good approximation; this motivates the choice of large T values, such as T = 1000 in Ho et al. (2020). However, as all T iterations have to be performed sequentially, instead of in parallel, to obtain a sample x0, sampling from DDPMs is much slower than sampling from other deep generative models, which makes them impractical for tasks where compute is limited and latency is critical.
3 VARIATIONAL INFERENCE FOR NON-MARKOVIAN FORWARD PROCESSES
Because the generative model approximates the reverse of the inference process, we need to rethink the inference process in order to reduce the number of iterations required by the generative model. Our key observation is that the DDPM objective in the form of Lγ only depends on the marginals2 q(xt|x0), but not directly on the joint q(x1:T |x0). Since there are many inference distributions (joints) with the same marginals, we explore alternative inference processes that are non-Markovian, which leads to new generative processes (Figure 1, right). These non-Markovian inference process lead to the same surrogate objective function as DDPM, as we will show below. In Appendix A, we show that the non-Markovian perspective also applies beyond the Gaussian case.
3.1 NON-MARKOVIAN FORWARD PROCESSES
Let us consider a family Q of inference distributions, indexed by a real vector σ ∈ RT
≥0:
qσ(x1:T |x0) := qσ(xT |x0)
T
∏
t=2
qσ(xt−1|xt, x0) (6)
where qσ(xT |x0) = N (√αT x0, (1 − αT )I) and for all t > 1,
qσ(xt−1|xt, x0) = N
(√αt−1x0 +
√
1 − αt−1 − σt2 · xt − √αtx0
√1 − αt
, σ2
tI
)
. (7)
The mean function is chosen to order to ensure that qσ(xt|x0) = N (√αtx0, (1 − αt)I) for all t (see Lemma 1 of Appendix B), so that it defines a joint inference distribution that matches the “marginals” as desired. The forward process3 can be derived from Bayes’ rule:
qσ(xt|xt−1, x0) = qσ(xt−1|xt, x0)qσ(xt|x0)
qσ(xt−1|x0) , (8)
1Please refer to Appendix C.2 for details. 2We slightly abuse this term (as well as joints) when only conditioned on x0. 3We overload the term “forward process” for cases where the inference model is not a diffusion.
3


Published as a conference paper at ICLR 2021
which is also Gaussian (although we do not use this fact for the remainder of this paper). Unlike the diffusion process in Eq. (3), the forward process here is no longer Markovian, since each xt could depend on both xt−1 and x0. The magnitude of σ controls the how stochastic the forward process is; when σ → 0, we reach an extreme case where as long as we observe x0 and xt for some t, then xt−1 become known and fixed.
3.2 GENERATIVE PROCESS AND UNIFIED VARIATIONAL INFERENCE OBJECTIVE
Next, we define a trainable generative process pθ(x0:T ) where each p(t)
θ (xt−1|xt) leverages knowl
edge of qσ(xt−1|xt, x0). Intuitively, given a noisy observation xt, we first make a prediction4 of the corresponding x0, and then use it to obtain a sample xt−1 through the reverse conditional distribution qσ(xt−1|xt, x0), which we have defined.
For some x0 ∼ q(x0) and t ∼ N (0, I), xt can be obtained using Eq. (4). The model (t)
θ (xt) then
attempts to predict t from xt, without knowledge of x0. By rewriting Eq. (4), one can then predict the denoised observation, which is a prediction of x0 given xt:
f (t)
θ (xt) := (xt − √1 − αt · (t)
θ (xt))/√αt. (9)
We can then define the generative process with a fixed prior pθ(xT ) = N (0, I) and
p(t)
θ (xt−1|xt) =
{
N (f (1)
θ (x1), σ12I) if t = 1
qσ(xt−1|xt, f (t)
θ (xt)) otherwise, (10)
where qσ(xt−1|xt, f (t)
θ (xt)) is defined as in Eq. (7) with x0 replaced by f (t)
θ (xt). We add some
Gaussian noise (with covariance σ12I) for the case of t = 1 to ensure that the generative process is supported everywhere.
We optimize θ via the following variational inference objective (which is a functional over θ):
Jσ(θ) := Ex0:T ∼qσ(x0:T )[log qσ(x1:T |x0) − log pθ(x0:T )] (11)
= Ex0:T ∼qσ (x0:T )
[
log qσ(xT |x0) +
T
∑
t=2
log qσ(xt−1|xt, x0) −
T
∑
t=1
log p(t)
θ (xt−1|xt) − log pθ(xT )
]
where we factorize qσ(x1:T |x0) according to Eq. (6) and pθ(x0:T ) according to Eq. (1).
From the definition of Jσ, it would appear that a different model has to be trained for every choice of σ, since it corresponds to a different variational objective (and a different generative process). However, Jσ is equivalent to Lγ for certain weights γ, as we show below.
Theorem 1. For all σ > 0, there exists γ ∈ RT>0 and C ∈ R, such that Jσ = Lγ + C.
The variational objective Lγ is special in the sense that if parameters θ of the models (t)
θ are not
shared across different t, then the optimal solution for θ will not depend on the weights γ (as global optimum is achieved by separately maximizing each term in the sum). This property of Lγ has two implications. On the one hand, this justified the use of L1 as a surrogate objective function for the variational lower bound in DDPMs; on the other hand, since Jσ is equivalent to some Lγ from Theorem 1, the optimal solution of Jσ is also the same as that of L1. Therefore, if parameters are not shared across t in the model θ, then the L1 objective used by Ho et al. (2020) can be used as a surrogate objective for the variational objective Jσ as well.
4 SAMPLING FROM GENERALIZED GENERATIVE PROCESSES
With L1 as the objective, we are not only learning a generative process for the Markovian inference process considered in Sohl-Dickstein et al. (2015) and Ho et al. (2020), but also generative processes for many non-Markovian forward processes parametrized by σ that we have described. Therefore, we can essentially use pretrained DDPM models as the solutions to the new objectives, and focus on finding a generative process that is better at producing samples subject to our needs by changing σ.
4Learning a distribution over the predictions is also possible, but empirically we found little benefits of it.
4


Published as a conference paper at ICLR 2021
Figure 2: Graphical model for accelerated generation, where τ = [1, 3].
4.1 DENOISING DIFFUSION IMPLICIT MODELS
From pθ(x1:T ) in Eq. (10), one can generate a sample xt−1 from a sample xt via:
xt−1 = √αt−1
(
xt − √1 − αt(t)
θ (xt)
√αt
)
} {{ }
“ predicted x0”
+
√
1 − αt−1 − σt2 · (t)
θ (xt)
} {{ }
“direction pointing to xt”
+ σtt
}{{}
random noise
(12)
where t ∼ N (0, I) is standard Gaussian noise independent of xt, and we define α0 := 1. Different choices of σ values results in different generative processes, all while using the same model θ, so re-training the model is unnecessary. When σt = √(1 − αt−1)/(1 − αt)√1 − αt/αt−1 for all t, the forward process becomes Markovian, and the generative process becomes a DDPM.
We note another special case when σt = 0 for all t5; the forward process becomes deterministic given xt−1 and x0, except for t = 1; in the generative process, the coefficient before the random noise t becomes zero. The resulting model becomes an implicit probabilistic model (Mohamed & Lakshminarayanan, 2016), where samples are generated from latent variables with a fixed procedure (from xT to x0). We name this the denoising diffusion implicit model (DDIM, pronounced /d:Im/), because it is an implicit probabilistic model trained with the DDPM objective (despite the forward process no longer being a diffusion).
4.2 ACCELERATED GENERATION PROCESSES
In the previous sections, the generative process is considered as the approximation to the reverse process; since of the forward process has T steps, the generative process is also forced to sample T steps. However, as the denoising objective L1 does not depend on the specific forward procedure as long as qσ(xt|x0) is fixed, we may also consider forward processes with lengths smaller than T , which accelerates the corresponding generative processes without having to train a different model.
Let us consider the forward process as defined not on all the latent variables x1:T , but on a subset {xτ1 , . . . , xτS }, where τ is an increasing sub-sequence of [1, . . . , T ] of length S. In particular, we define the sequential forward process over xτ1 , . . . , xτS such that q(xτi |x0) =
N (√ατi x0, (1 − ατi )I) matches the “marginals” (see Figure 2 for an illustration). The generative process now samples latent variables according to reversed(τ ), which we term (sampling) trajectory. When the length of the sampling trajectory is much smaller than T , we may achieve significant increases in computational efficiency due to the iterative nature of the sampling process.
Using a similar argument as in Section 3, we can justify using the model trained with the L1 objective, so no changes are needed in training. We show that only slight changes to the updates in Eq. (12) are needed to obtain the new, faster generative processes, which applies to DDPM, DDIM, as well as all generative processes considered in Eq. (10). We include these details in Appendix C.1.
In principle, this means that we can train a model with an arbitrary number o

=== Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow (Liu, Xingchao; Gong, Chengyue; Liu, Qiang) ===
Flow Straight and Fast:
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
However, compared with the traditional one-step models like GAN and VAE, a key drawback of continuoustimes models is the high computational cost in inference time: drawing a single point (e.g., image) requires to solve the ODE/SDE with a numerical solver that needs to repeatedly call the expensive neural drift function. In addition, the existing denoising diffusion techniques require substantial hyper-parameter search in an involved design space and are still poorly understood both empirically and theoretically [29].
In existing approaches, generative modeling and domain transfer are typically treated separately. It often requires to extend or customize a generative learning techniques to solve domain transfer problems; see e.g., Cycle GAN [100] and diffusion-based image-to-image translation [e.g., 75, 97]. One framework that naturally unifies both domains is optimal transport (OT) [e.g., 85, 2, 15, 59], which endows a collection of techniques for finding optimal couplings with minimum transport costs of form E[c(Z1 − Z0)] w.r.t. a cost function c : Rd → R, yielding natural applications to both generative and transfer learning. However, the existing OT techniques are slow for problems with high dimensional and large volumes of data [59]. Furthermore, as the transport costs do not perfectly align with the actual learning performance, methods that faithfully find the optimal transport maps do not necessarily have better learning performance [34].
2


Figure 1: The trajectories of rectified flows for image generation (π0: standard Gaussian noise, π1: cat faces, top two rows), and image transfer between human and cat faces (π0: human faces, π1: cat faces, bottom two rows), when simulated using Euler method with step size 1/N for N steps. The first rectified flow induced from the training data (1-rectified flow) yields good results with a very small number (e.g., ≥ 2) of steps; the straightened reflow induced from 1-rectified flow (denoted as 2-rectified flow) has nearly straight line trajectories and yield good results even with one discretization step.
Contribution
We introduce rectified flow, a surprisingly simple approach to the transport mapping problem, which unifiedly solves both generative modeling and domain transfer. The rectified flow is an ODE model that transport distribution π0 to π1 by following straight line paths as much as possible. The straight paths are preferred both theoretically because it is the shortest path between two end points, and computationally because it can be exactly simulated without time discretization. Hence, flows with straight paths bridge the gap between one-step and continuous-time models.
Algorithmically, the rectified flow is trained with a simple and scalable unconstrained least squares optimization procedure, which avoids the instability issues of GANs, the intractable likelihood of MLE methods, and the subtle hyper-parameter decisions of denoising diffusion models. The procedure of obtaining the rectified flow from the training data has the attractive theoretical property of 1) yielding a coupling with non-increasing transport cost jointly for all convex cost c, and 2) making the paths of flow increasingly straight and hence incurring lower error with numerical solvers. Therefore, with a reflow procedure that iteratively trains new rectified flows with the data simulated from the previously obtained rectified flow, we obtain nearly straight flows that yield good results even with the coarsest time discretization, i.e., one Euler step. Our method is purely ODE-based, and is both conceptually simpler and practically faster in inference time than the SDE-based approaches of [23, 73, 70].
3


(a) Linear interpolation
Xt = tX1 + (1 − t)X0
(b) Rectified flow Zt
induced by (X0, X1)
(c) Linear interpolation
Zt = tZ1 + (1 − t)Z0
(d) Rectified flow Zt′
induced by (Z0, Z1)
Figure 2: (a) Linear interpolation of data input (X0, X1) ∼ π0 × π1. (b) The rectified flow Zt induced by (X0, X1); the trajectories are “rewired” at the intersection points to avoid the crossing. (c) The linear interpolation of the end points (Z0, Z1) of flow Zt. (d) The rectified flow induced from (Z0, Z1), which follows straight paths.
Empirically, rectified flow can yield high-quality results for image generation when simulated with a very few number of Euler steps (see Figure 1, top row). Moreover, with just one step of reflow, the flow becomes nearly straight and hence yield good results with a single Euler discretization step (Figure 1, the second row). This substantially improves over the standard denoising diffusion methods. Quantitatively, we claim a state-of-the-art result of FID (4.85) and recall (0.51) on CIFAR10 for one-step fast diffusion/flow models [5, 48, 91, 99, 47]. The same algorithm also achieves superb result on domain transfer tasks such as image-toimage translation (see the bottom two rows of Figure 1) and transfer learning.
2 Method
We provide a quick overview of the method in Section 2.1, followed with some discussion and remarks in Section 2.2. We introduce a nonlinear extension of our method in Section 2.3, with which we clarify the connection and advantages of our method with the method of probability flow ODEs [73] and DDIM [70].
2.1 Overview
Rectified flow Given empirical observations of X0 ∼ π0, X1 ∼ π1, the rectified flow induced from (X0, X1) is an ordinary differentiable model (ODE) on time t ∈ [0, 1],
dZt = v(Zt, t)dt,
which converts Z0 from π0 to a Z1 following π1. The drift force v : Rd → Rd is set to drive the flow to follow the direction (X1 − X0) of the linear path pointing from X0 to X1 as much as possible, by solving a simple least squares regression problem:
mvin
∫1
0
E
[∥
∥(X1 − X0) − v(Xt, t)∥
∥
2
]
dt, with Xt = tX1 + (1 − t)X0, (1)
where Xt is the linear interpolation of X0 and X1. Naviely, Xt follows the ODE of dXt = (X1 − X0)dt, which is non-causal (or anticipating) as the update of Xt requires the information of the final point X1. By fitting the drift v with X1 − X0, the rectified flow causalizes the paths of linear interpolation Xt, yielding an ODE flow that can be simulated without seeing the future.
In practice, we parameterize v with a neural network or other nonlinear models and solve (1) with any offthe-shelf stochastic optimizer, such as stochastic gradient descent, with empirical draws of (X0, X1). See
4


Algorithm 1. After we get v, we solve the ODE starting from Z0 ∼ π0 to transfer π0 to π1, backwardly starting from Z1 ∼ π1 to transfer π1 to π0. Specifically, for backward sampling, we simply solve dX ̃t = −v(X ̃t, t)dt initialized from X ̃0 ∼ π1 and set Xt = X ̃1−t. The forward and backward sampling are equally favored by the training algorithm, because the objective in (1) is time-symmetric in that it yields the equivalent problem if we exchange X0 and X1 and flip the sign of v.
Flows avoid crossing A key to understanding the method is the non-crossing property of flows: the different paths following a well defined ODE dZt = v(Zt, t)dt, whose solution exists and is unique, cannot cross each other at any time t ∈ [0, 1). Specifically, there exists no location z ∈ Rd and time t ∈ [0, 1), such that two paths go across z at time t along different directions, because otherwise the solution of the ODE would be non-unique. On the other hand, the paths of the interpolation process Xt may intersect with each other (Figure 2a), which makes it non-causal. Hence, as shown in Figure 2b, the rectified flow rewires the individual trajectories passing through the intersection points to avoid crossing, while tracing out the same density map as the linear interpolation paths due to the optimization of (1). We can view the linear interpolation Xt as building roads (or tunnels) to connect π0 and π1, and the rectified flow as traffics of particles passing through the roads in a myopic, memoryless, non-crossing way, which allows them to ignore the global path information of how X0 and X1 are paired, and rebuild a more deterministic pairing of (Z0, Z1).
Rectified flows reduce transport costs If (1) is solved exactly, the pair (Z0, Z1) of the rectified flow is guaranteed to be a valid coupling of π0, π1 (Theorem 3.3), that is, Z1 follows π1 if Z0 ∼ π0. Moreover, (Z0, Z1) guarantees to yield no larger transport cost than the data pair (X0, X1) simultaneously for all convex cost functions c (Theorem 3.5). The data pair (X0, X1) can be an arbitrary coupling of π0, π1, typically independent (i.e., (X0, X1) ∼ π0 × π1) as dictated by the lack of meaningfully paired observations in practical problems. In comparison, the rectified coupling (Z0, Z1) has a deterministic dependency as it is constructed from an ODE model. Denote by (Z0, Z1) = Rectify((X0, X1)) the mapping from (X0, X1) to (Z0, Z1). Hence, Rectify(·) converts an arbitrary coupling into a deterministic coupling with lower convex transport costs.
Straight line flows yield fast simulation Following Algorithm 1, denote by Z = RectFlow((X0, X1)) the rectified flow induced from (X0, X1). Applying this operator recursively yields a sequence of rectified flows Zk+1 = RectFlow((Z0k, Z1k)) with (Z00, Z10) = (X0, X1), where Zk is the k-th rectified flow, or simply k-rectified flow, induced from (X0, X1).
This reflow procedure not only decreases transport cost, but also has the important effect of straightening paths of rectified flows, that is, making the paths of the flow more straight. This is highly attractive computationally as flows with nearly straight paths incur small time-discretization error in numerical simulation. Indeed, perfectly straight paths can be simulated exactly with a single Euler step and is effectively a onestep model. This addresses the very bottleneck of high inference cost in existing continuous-time ODE/SDE models.
2.2 Main Results and Properties
We provide more in-depth discussions on the main properties of rectified flow. We keep the discussion informal to highlight the intuitions in this section and defer the full course theoretical analysis to Section 3.
5


Algorithm 1 Rectified Flow: Main Algorithm
Procedure: Z = RectFlow((X0, X1)):
Inputs: Draws from a coupling (X0, X1) of π0 and π1; velocity model vθ : Rd → Rd with parameter θ.
Training: θˆ = arg min
θ
E
[
‖X1 − X0 − v(tX1 + (1 − t)X0, t)‖2]
, with t ∼ Uniform([0, 1]).
Sampling: Draw (Z0, Z1) following dZt = vθˆ(Zt, t)dt starting from Z0 ∼ π0 (or backwardly Z1 ∼ π1). Return: Z = {Zt : t ∈ [0, 1]}.
Reflow (optional): Zk+1 = RectFlow((Z0k, Z1k)), starting from (Z00, Z10) = (X0, X1).
Distill (optional): Learn a neural network Tˆ to distill the k-rectified flow, such that Z1k ≈ Tˆ(Z0k).
First, for a given input coupling (X0, X1), it is easy to see that the exact minimum of (1) is achieved if
vX (x, t) = E[X1 − X0 | Xt = x], (2)
which is the expectation of the line directions X1 − X0 that pass through x at time t. We discuss below the property of rectified flow dZt = vX (Zt, t)dt with Z0 ∼ π0, assuming that the ODE has an unique solution.
Marginal preserving property [Theorem 3.3] The pair (Z0, Z1) is a coupling of π0 and π1. In fact, the marginal law of Zt equals that of Xt at every time t, that is, Law(Zt) = Law(Xt), ∀t ∈ [0, 1].
Intuitively, this is because, by the definition of vX in (2), the expected amount of mass that passes through every infinitesmal volume at all location and time are equal under the dynamics of Xt and Zt, which ensures that they trace out the same marginal distributions:
Flow in & out
()
= Flow in & out
()
, ∀time & location =⇒ Law(Zt) = Law(Xt), ∀t.
On the other hand, the joint distributions of the whole trajectory of Zt and that of Xt are different in general. In particular, Xt is in general a non-causal, non-Markov process, with (X0, X1) a stochastic coupling, and Zt causalizes, Markovianizes and derandomizes Xt, while preserving the marginal distributions at all time.
Reducing transport costs [Theorem 3.5] The coupling (Z0, Z1) yields lower or equal convex transport costs than the input (X0, X1) in that E[c(Z1 − Z0)] ≤ E[c(X1 − X0)] for any convex cost c : Rd → R.
The transport costs measure the expense of transporting the mass of one distribution to another following the assignment relation specified by the coupling and is a central topic in optimal transport [e.g., 

=== Structured Denoising Diffusion Models in Discrete State-Spaces (Ho, Jonathan; Austin, Jacob; Johnson, Daniel D.; Tarlow, Daniel; Berg, Rianne va) ===
Structured Denoising Diffusion Models in Discrete State-Spaces
Jacob Austin,∗Daniel D. Johnson,∗Jonathan Ho, Daniel Tarlow & Rianne van den Berg† Google Research, Brain Team
{jaaustin,ddjohnson,jonathanho,dtarlow,riannevdberg}@google.com
Abstract
Denoising diffusion probabilistic models (DDPMs) [19] have shown impressive results on image and waveform generation in continuous state spaces. Here, we introduce Discrete Denoising Diffusion Probabilistic Models (D3PMs), diffusionlike generative models for discrete data that generalize the multinomial diffusion model of Hoogeboom et al. [20], by going beyond corruption processes with uniform transition probabilities. This includes corruption with transition matrices that mimic Gaussian kernels in continuous space, matrices based on nearest neighbors in embedding space, and matrices that introduce absorbing states. The third allows us to draw a connection between diffusion models and autoregressive and mask-based generative models. We show that the choice of transition matrix is an important design decision that leads to improved results in image and text domains. We also introduce a new loss function that combines the variational lower bound with an auxiliary cross entropy loss. For text, this model class achieves strong results on character-level text generation while scaling to large vocabularies on LM1B. On the image dataset CIFAR-10, our models approach the sample quality and exceed the log-likelihood of the continuous-space DDPM model.
1 Introduction
Generative modeling is a core problem in machine learning, useful both for benchmarking our ability to capture statistics of natural datasets and for downstream applications that require generating high-dimensional data like images, text, and speech waveforms. There has been a great deal of progress with the development of methods like GANs [15, 4], VAEs [25, 35], large autoregressive neural network models [51, 50, 52], normalizing flows [34, 12, 24, 32], and others, each with their own tradeoffs in terms of sample quality, sampling speed, log-likelihoods, and training stability.
Recently, diffusion models [43] have emerged as a compelling alternative for image [19, 46] and audio [7, 26] generation, achieving comparable sample quality to GANs and log-likelihoods comparable to autoregressive models with fewer inference steps. A diffusion model is a parameterized Markov chain trained to reverse a predefined forward process, which is a stochastic process constructed to gradually corrupt training data into pure noise. Diffusion models are trained using a stable objective closely related to both maximum likelihood and score matching [21, 53], and they admit faster sampling than autoregressive models by using parallel iterative refinement [30, 45, 47, 44].
Although diffusion models have been proposed in both discrete and continuous state spaces [43], most recent work has focused on Gaussian diffusion processes that operate in continuous state spaces (e.g. for real-valued image and waveform data). Diffusion models with discrete state spaces have been explored for text and image segmentation domains [20], but they have not yet been demonstrated as a competitive model class for large scale text or image generation.
35th Conference on Neural Information Processing Systems (NeurIPS 2021).
∗Equal contributions †Now at Microsoft Research
arXiv:2107.03006v3 [cs.LG] 22 Feb 2023


Figure 1: D3PM forward and (learned) reverse process applied to a quantized swiss roll. Each dot represents a 2D categorical variable. Top: samples from the uniform, discretized Gaussian, and absorbing state D3PM model forward processes, along with corresponding transition matrices Q. Bottom: samples from a learned discretized Gaussian reverse process.
Our aim in this work is to improve and extend discrete diffusion models by using a more structured categorical corruption process to shape data generation, as illustrated in Figure 1. Our models do not require relaxing or embedding discrete data (including images) into continuous spaces, and can embed structure or domain knowledge into the transition matrices used by the forward process. We achieve significantly improved results by taking advantage of this flexibility. We develop structured corruption processes appropriate for text data, using similarity between tokens to enable gradual corruption and denoising. Expanding further, we also explore corruption processes that insert [MASK] tokens, which let us draw parallels to autoregressive and mask-based generative models. Finally, we study discrete diffusion models for quantized images, taking inspiration from the locality exploited by continuous diffusion models. This leads to a particular choice of discrete corruption process that diffuses preferentially to more similar states and leads to much better results in the image domain.
Overall, we make a number of technical and conceptual contributions. Beyond designing several new structured diffusion models, we introduce a new auxiliary loss which stabilizes training of D3PMs and a family of noise schedules based on mutual information that lead to improved performance. We strongly outperform various non-autoregressive baselines for text generation on character-level text generation, and successfully scale discrete diffusion models to large vocabularies and long sequence lengths. We also achieve strong results on the image dataset CIFAR-10, approaching or exceeding the Gaussian diffusion model from Ho et al. [19] on log-likelihoods and sample quality.
2 Background: diffusion models
Diffusion models [43] are latent variable generative models characterized by a forward and a reverse Markov process. The forward process q(x1:T |x0) = ∏T
t=1 q(xt|xt−1) corrupts the data x0 ∼
q(x0) into a sequence of increasingly noisy latent variables x1:T = x1, x2, ..., xT . The learned reverse Markov process pθ(x0:T ) = p(xT ) ∏T
t=1 pθ(xt−1|xt) gradually denoises the latent variables towards the data distribution. For example, for continuous data, the forward process typically adds Gaussian noise, which the reverse process learns to remove.
In order to optimize the generative model pθ(x0) to fit the data distribution q(x0), we typically optimize a variational upper bound on the negative log-likelihood:
Lvb = Eq(x0)
[
DKL[q(xT |x0)||p(xT )]
} {{ }
LT
+
T
∑
t=2
Eq (xt |x0 )
[DKL[q(xt−1|xt, x0)||pθ(xt−1|xt)]]
} {{ } Lt−1
−Eq(x1|x0)[log pθ(x0|x1)]
} {{ }
L0
]
. (1)
2


When the number of time steps T goes to infinity, both the forward process and the reverse process share the same functional form [13], allowing the use of a learned reverse process from the same class of distributions as that of the forward process. Furthermore, for several choices of the forward process the distribution q(xt|x0) converges to a stationary distribution π(x) in the limit t → ∞ independent of the value of x0. When the number of time steps T is large enough and we choose π(x) as the prior p(xT ), we can guarantee that the LT term in (1) will approach zero regardless of the data distribution q(x0). (Alternatively, one can use a learned prior pθ(xT ).)
While q(xt|xt−1) can in theory be arbitrary, efficient training of pθ is possible when q(xt|xt−1):
1. Permits efficient sampling of xt from q(xt|x0) for an arbitrary time t, allowing us to randomly sample timesteps and optimize each Lt−1 term individually with stochastic gradient descent,
2. Has a tractable expression for the forward process posterior q(xt−1|xt, x0), which allows us to compute the KL divergences present in the Lt−1 term of (1).
The majority of recent work in continuous spaces [19, 44, 7, 30] defines the forward
and reverse distributions as q(xt|xt−1) = N (xt|√1 − βtxt−1, βtI) and pθ(xt−1|xt) = N (xt−1|μθ(xt, t), Σθ(xt, t)), respectively. The aforementioned properties hold in the case of these Gaussian diffusion models: the forward process q(xt|x0) converges to a stationary distribution, motivating the choice p(xT ) = N (xT |0, I), and both q(xt|x0) and q(xt−1|xt, x0) are tractable Gaussian distributions for which the KL divergence can be computed analytically.
3 Diffusion models for discrete state spaces
Diffusion models with discrete state spaces were first introduced by Sohl-Dickstein et al. [43], who considered a diffusion process over binary random variables. Hoogeboom et al. [20] extended the model class to categorical random variables with transition matrices characterized by uniform transition probabilities. In their supplementary material, Song et al. [44] also derived this extension, although no experiments were performed with this model class. Here, we briefly describe a more general framework for diffusion with categorical random variables which includes these models as special cases.
For scalar discrete random variables with K categories xt, xt−1 ∈ 1, ..., K the forward transition probabilities can be represented by matrices: [Qt]ij = q(xt = j|xt−1 = i). Denoting the one-hot version of x with the row vector x, we can write
q(xt|xt−1) = Cat(xt; p = xt−1Qt), (2)
where Cat(x; p) is a categorical distribution over the one-hot row vector x with probabilities given by the row vector p, and xt−1Qt is to be understood as a row vector-matrix product. We assume that Qt is applied to each pixel of an image or each token in a sequence independently, and that q factorizes over these higher dimensions as well; we thus write q(xt|xt−1) in terms of a single element. Starting from x0, we obtain the following t-step marginal and posterior at time t − 1:
q(xt|x0) = Cat (xt; p = x0Qt
) , with Qt = Q1Q2 . . . Qt
q(xt−1|xt, x0) = q(xt|xt−1, x0)q(xt−1|x0)
q(xt|x0) = Cat
(
xt−1; p = xtQt> x0Qt−1
x0Qtxt>
)
. (3)
Note that due to the Markov property of the forward process q(xt|xt−1, x0) = q(xt|xt−1). Assuming that the reverse process pθ(xt|xt−1) is also factorized as conditionally independent over the image or sequence elements, the KL divergence between q and pθ can be computed by simply summing over all possible values of each random variable; we thus satisfy criteria 1 and 2 discussed in Section 2. Depending on Qt, the cumulative products Qt can often be computed in closed form, or simply precomputed for all t. However, for large K and large T this may be prohibitive. In Appendix A.4 we discuss how to ensure Qt can still be computed efficiently in this case, allowing the framework to scale to a larger number of categories.
In the next section we discuss the choice of the Markov transition matrices Qt and corresponding stationary distributions. From here on, we refer to the general class of diffusion models with discrete state spaces as Discrete Denoising Diffusion Probabilistic Models (D3PMs).
3


3.1 Choice of Markov transition matrices for the forward process
An advantage of the D3PM framework described above is the ability to control the data corruption and denoising process by choosing Qt, in notable contrast to continuous diffusion, for which only additive Gaussian noise has received significant attention. Besides the constraint that the rows of Qt must sum to one to conserve probability mass, the only other constraint in choosing Qt is that the rows of Qt = Q1Q2 . . . Qt must converge to a known stationary distribution3 when t becomes large, which can be guaranteed while imposing minimal restrictions on Qt (see Appendix A.1).
We argue that for most real-world discrete data, including images and text, it makes sense to add domain-dependent structure to the transition matrices Qt as a way of controlling the forward corruption process and the learnable reverse denoising process. Below we briefly discuss the uniform transition matrices that have been studied in prior work [20], along with a set of structured transition matrices we have explored for our image and text dataset experiments; see Appendix A.2 for more details on each matrix type. We also note that this set is not exhaustive, and many other transition matrices could also be used within the D3PM framework.
Uniform (Appendix A.2.1). Sohl-Dickstein et al. [43] considered a simple 2 × 2 transition matrix for binary random variables. Hoogeboom et al. [20] later extended this to categorical variables, proposing a transition matrix Qt = (1 − βt)I + βt/K 11T with βt ∈ [0, 1]. Since this transition matrix is doubly stochastic with strictly positive entries, the stationary distribution is uniform. Because the transition probability to any other state is uniform, in this paper we equivalently refer to this discrete diffusion instance as D3PM-uniform.
Absorbing state (Appendix A.2.2). Motivated by the success of BERT [11] and recent work on Conditional Masked Language Models (CMLMs) in text, we consider a transition matrix with an absorbing state (called [MASK]), such that each token either stays the same or transitions to [MASK] with some probability βt. This does not impose particular relationships between categories, similar to uniform diffusion, but still allows corrupted tokens to be distinguished from original ones. Moreover, the stationary distribution is not uniform but has all the mass on the [MASK] token. For images, we reuse the grey pixel as the [MASK] absorbing token.
Discretized Gaussian (Appendix A.2.3). Instead of transitioning uniformly to any other state, for ordinal data we propose imitating a continuous space diffusion model by using a discretized, truncated Gaussian distribution. We choose a normalization such that the transition matrix is doubly stochastic, leading to a uniform stationary distribution. This transition matrix will transition between more similar states with higher probability, and is well suited for quantized ordinal data such as images.
Token embedding distance (Appendix A.2.4). Textual data does not have ordinal structure, but there may still be interesting semantic relationships. For instance, in a character level vocabulary vowels may be more similar to each other than they are to consonants. As a demonstration of the generality of the D3PM framework, we explore using similarity in an embedding space to guide the forward process, and construct a doubly-stochastic transition matrix that transitions more frequently between tokens that have similar embeddings while maintaining a uniform stationary distribution.
For uniform and absorbing-state diffusion, the cumulative products Qt can be computed in closed form (see Appendix A.4.1); the remainder can be precomputed.
3.2 Noise schedules
We consider several different options for the noise schedule of the forward process. For discretized Gaussian diffusion, we explore linearly increasing the variance of the Gaussian before discretizing it. (Note that a linear schedule for Qt leads to a nonlinear amount of cumulative noise in Qt.) For uniform diffusion we use the cosine schedule which sets the cumulative probability of a transition to a cosine function, as introduced by Nichol and Dhariwal [30] and adapted by Hoogeboom et al. [20]. For a general set of transition matrices Qt (such as the one based on token embeddings), previously proposed schedules may not be directly applicable. We consider linearly interpolating the mutual information between xt and x0 to zero, i.e. I(xt; x0) ≈ (1 − t
T ) H(x0). Interestingly, for the
3If a stationary distribution is not known, we can introduce a learned prior pθ(xT ); we note that this is equivalent to extending the forward process by appending a rank-one matrix QT +1 that ignores xT and produces a deterministic xT +1, then learning the reverse step pθ(xT |xT +1) = pθ(xT ).
4


specific case of absorbing-state D3PMs, this schedule reduces to exactly the (T − t + 1)−1 schedule proposed by Sohl-Dickstein et al. [43] for a Bernoulli diffusion process. See Appendix A.7 for more details.
3.3 Parameterization of the reverse process
While it is possible to directly predict the logits of pθ(xt−1|xt) using a neural network nnθ(xt), we follow Ho et al. [19] and Hoogeboom et al. [20] and focus on using a neural network nnθ(xt) to predict the logits of a distribution p ̃θ(x ̃0|xt), which we combine with q(xt−1|xt, x0) and a summation over one-hot representations of x0 to obtain the following parameterization
pθ(xt−1|xt) ∝
∑
x ̃0
q(xt−1, xt|x ̃0)p ̃θ(x ̃0|xt). (4)
We note that under this x0-parameterization the KL divergence DKL[q(xt−1|xt, x0)||pθ(xt−1|xt)] will be zero if p ̃θ(x ̃0|xt) places all of its probability mass on the original value x0. The decomposition of q(xt−1|xt, x0) in (3) also provides us with a motivation for this parameterization. According to (3), in a given state xt, the optimal reverse process only takes into account transitions to states for which q(xt|xt−1) is non-zero. Therefore, the sparsity pattern of Qt determines the sparsity pattern of the ideal reverse transition probabilities in pθ(xt−1|xt). The parameterization in (4) automatically ensures that the learned reverse probability distribution pθ(xt−1|xt) has the correct sparsity pattern dictated by the choice of the Markov transition matrix Qt. This parameterization also lets us perform inference with k steps at a time, by predicting pθ(xt−k|xt) = ∑ q(xt−k, xt|x ̃0)p ̃θ(x ̃0|xt).
Finally, when modeling ordinal discrete data, instead of predicting the logits of p ̃θ(x ̃0|xt) directly with the output of a neural net, another option is to model the probabilities with a truncated discretized logistic distribution (see Appendix A.8). This provides an extra ordinal inductive bias to the reverse model and boosts FID and log-likelihood scores for images.
3.4 Loss function
While the original diffusion models introduced by Sohl-Dickstein et al. [43] were optimized with the negative variational lower bound Lvb of (1), more recent diffusion models are optimized with different objectives. For instance, Ho et al. [19] derive a simplified loss function (Lsimple) that reweights the negative variational bound, and Nichol and Dhariwal [30] explore a hybrid loss Lhybrid = Lsimple + λLvb (using one term to learn the predic

=== Understanding Diffusion Models: A Unified Perspective (Luo, Calvin) ===
arXiv:2208.11970v1 [cs.LG] 25 Aug 2022

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

Directly computing and maximizing the likelihood p(x) is diﬃcult because it either involves integrating out all latent variables z in Equation 1, which is intractable for complex models, or it involves having access to a ground truth latent encoder p(z|x) in Equation 2. However, using these two equations, we can derive a term called the Evidence Lower Bound (ELBO), which as its name suggests, is a lower bound of the evidence. The evidence is quantiﬁed in this case as the log likelihood of the observed data. Then, maximizing the ELBO becomes a proxy objective with which to optimize a latent variable model; in the best case, when the ELBO is powerfully parameterized and perfectly optimized, it becomes exactly equivalent to the evidence. Formally, the equation of the ELBO is:

p(x, z)

Eqφ (z |x)

log qφ(z|x)

(3)

2

To make the relationship with the evidence explicit, we can mathematically write:

p(x, z)

log p(x) ≥ Eqφ(z|x)

log qφ(z|x)

(4)

Here, qφ(z|x) is a ﬂexible approximate variational distribution with parameters φ that we seek to optimize. Intuitively, it can be thought of as a parameterizable model that is learned to estimate the true distribution over latent variables for given observations x; in other words, it seeks to approximate true posterior p(z|x). As we will see when exploring the Variational Autoencoder, as we increase the lower bound by tuning the parameters φ to maximize the ELBO, we gain access to components that can be used to model the true data distribution and sample from it, thus learning a generative model. For now, let us try to dive deeper into why the ELBO is an objective we would like to maximize.

Let us begin by deriving the ELBO, using Equation 1:

log p(x) = log p(x, z)dz

(Apply Equation 1)

(5)

= log p(x, z)qφ(z|x) dz qφ(z|x)

p(x, z) = log Eqφ(z|x) qφ(z|x)

p(x, z)

≥ Eqφ(z|x)

log qφ(z|x)

(Multiply by 1 = qφ(z|x) )

(6)

qφ(z|x)

(Deﬁnition of Expectation)

(7)

(Apply Jensen’s Inequality)

(8)

In this derivation, we directly arrive at our lower bound by applying Jensen’s Inequality. However, this does not supply us much useful information about what is actually going on underneath the hood; crucially, this proof gives no intuition on exactly why the ELBO is actually a lower bound of the evidence, as Jensen’s Inequality handwaves it away. Furthermore, simply knowing that the ELBO is truly a lower bound of the data does not really tell us why we want to maximize it as an objective. To better understand the relationship between the evidence and the ELBO, let us perform another derivation, this time using Equation 2:

log p(x) = log p(x) qφ(z|x)dz

(Multiply by 1 = qφ(z|x)dz) (9)

= qφ(z|x)(log p(x))dz

(Bring evidence into integral) (10)

= Eqφ(z|x) [log p(x)]

p(x, z)

= Eqφ(z|x)

log p(z|x)

(Deﬁnition of Expectation)

(11)

(Apply Equation 2)

(12)

= Eqφ(z|x)

log p(x, z)qφ(z|x) p(z|x)qφ(z|x)

(Multiply by 1 = qφ(z|x) )

(13)

qφ(z|x)

= Eqφ(z|x)

p(x, z) log
qφ(z|x)

+ Eqφ(z|x)

log qφ(z|x) p(z|x)

(Split the Expectation)

(14)

p(x, z)

= Eqφ(z|x)

log qφ(z|x)

+ DKL(qφ(z|x)

p(z|x))

(Deﬁnition of KL Divergence)

(15)

p(x, z)

≥ Eqφ(z|x)

log qφ(z|x)

(KL Divergence always ≥ 0)

(16)

From this derivation, we clearly observe from Equation 15 that the evidence is equal to the ELBO plus the KL Divergence between the approximate posterior qφ(z|x) and the true posterior p(z|x). In fact, it was this KL Divergence term that was magically removed by Jensen’s Inequality in Equation 8 of the ﬁrst derivation. Understanding this term is the key to understanding not only the relationship between the ELBO and the evidence, but also the reason why optimizing the ELBO is an appropriate objective at all.

Firstly, we now know why the ELBO is indeed a lower bound: the diﬀerence between the evidence and the ELBO is a strictly non-negative KL term, thus the value of the ELBO can never exceed the evidence.

3

Figure 1: A Variational Autoencoder graphically represented. Here, encoder q(z|x) deﬁnes a distribution over latent variables z for observations x, and p(x|z) decodes latent variables into observations.

Secondly, we explore why we seek to maximize the ELBO. Having introduced latent variables z that we would like to model, our goal is to learn this underlying latent structure that describes our observed data. In other words, we want to optimize the parameters of our variational posterior qφ(z|x) to exactly match the true posterior distribution p(z|x), which is achieved by minimizing their KL Divergence (ideally to zero). Unfortunately, it is intractable to minimize this KL Divergence term directly, as we do not have access to the ground truth p(z|x) distribution. However, notice that on the left hand side of Equation 15, the likelihood of our data (and therefore our evidence term log p(x)) is always a constant with respect to φ, as it is computed by marginalizing out all latents z from the joint distribution p(x, z) and does not depend on φ whatsoever. Since the ELBO and KL Divergence terms sum up to a constant, any maximization of the ELBO term with respect to φ necessarily invokes an equal minimization of the KL Divergence term. Thus, the ELBO can be maximized as a proxy for learning how to perfectly model the true latent posterior distribution; the more we optimize the ELBO, the closer our approximate posterior gets to the true posterior. Additionally, once trained, the ELBO can be used to estimate the likelihood of observed or generated data as well, since it is learned to approximate the model evidence log p(x).

Variational Autoencoders
In the default formulation of the Variational Autoencoder (VAE) [1], we directly maximize the ELBO. This approach is variational, because we optimize for the best qφ(z|x) amongst a family of potential posterior distributions parameterized by φ. It is called an autoencoder because it is reminiscent of a traditional autoencoder model, where input data is trained to predict itself after undergoing an intermediate bottlenecking representation step. To make this connection explicit, let us dissect the ELBO term further:

p(x, z)

Eqφ (z |x)

log qφ(z|x)

= Eqφ(z|x)

log pθ(x|z)p(z) qφ(z|x)

= Eqφ(z|x) [log pθ(x|z)] + Eqφ(z|x)

p(z) log
qφ(z|x)

= Eqφ(z|x) [log pθ(x|z)] − DKL(qφ(z|x) p(z))

reconstruction term

prior matching term

(Chain Rule of Probability) (17)

(Split the Expectation)

(18)

(Deﬁnition of KL Divergence) (19)

In this case, we learn an intermediate bottlenecking distribution qφ(z|x) that can be treated as an encoder ; it transforms inputs into a distribution over possible latents. Simultaneously, we learn a deterministic function pθ(x|z) to convert a given latent vector z into an observation x, which can be interpreted as a decoder.
The two terms in Equation 19 each have intuitive descriptions: the ﬁrst term measures the reconstruction likelihood of the decoder from our variational distribution; this ensures that the learned distribution is modeling eﬀective latents that the original data can be regenerated from. The second term measures how similar the learned variational distribution is to a prior belief held over latent variables. Minimizing this term encourages the encoder to actually learn a distribution rather than collapse into a Dirac delta function. Maximizing the ELBO is thus equivalent to maximizing its ﬁrst term and minimizing its second term.

4

A deﬁning feature of the VAE is how the ELBO is optimized jointly over parameters φ and θ. The encoder of the VAE is commonly chosen to model a multivariate Gaussian with diagonal covariance, and the prior is often selected to be a standard multivariate Gaussian:

qφ(z|x) = N (z; µφ(x), σφ2 (x)I)

(20)

p(z) = N (z; 0, I)

(21)

Then, the KL divergence term of the ELBO can be computed analytically, and the reconstruction term can be approximated using a Monte Carlo estimate. Our objective can then be rewritten as:

L

arg max Eqφ(z|x) [log pθ(x|z)] − DKL(qφ(z|x) p(z)) ≈ arg max log pθ(x|z(l)) − DKL(qφ(z|x) p(z)) (22)

φ,θ

φ,θ l=1

where latents {z(l)}Ll=1 are sampled from qφ(z|x), for every observation x in the dataset. However, a problem arises in this default setup: each z(l) that our loss is computed on is generated by a stochastic sampling procedure, which is generally non-diﬀerentiable. Fortunately, this can be addressed via the reparameterization trick when qφ(z|x) is designed to model certain distributions, including the multivariate Gaussian.
The reparameterization trick rewrites a random variable as a deterministic function of a noise variable; this allows for the optimization of the non-stochastic terms through gradient descent. For example, samples from a normal distribution x ∼ N (x; µ, σ2) with arbitrary mean µ and variance σ2 can be rewritten as:

x = µ + σ with ∼ N ( ; 0, I)

In other words, arbitrary Gaussian distributions can be interpreted as standard Gaussians (of which is a sample) that have their mean shifted from zero to the target mean µ by addition, and their variance stretched by the target variance σ2. Therefore, by the reparameterization trick, sampling from an arbitrary Gaussian distribution can be performed by sampling from a standard Gaussian, scaling the result by the target standard deviation, and shifting it by the target mean.
In a VAE, each z is thus computed as a deterministic function of input x and auxiliary noise variable :

z = µφ(x) + σφ(x)

with ∼ N ( ; 0, I)

where represents an element-wise product. Under this reparameterized version of z, gradients can then be computed with respect to φ as desired, to optimize µφ and σφ. The VAE therefore utilizes the reparameterization trick and Monte Carlo estimates to optimize the ELBO jointly over φ and θ.
After training a VAE, generating new data can be performed by sampling directly from the latent space p(z) and then running it through the decoder. Variational Autoencoders are particularly interesting when the dimensionality of z is less than that of input x, as we might then be learning compact, useful representations. Furthermore, when a semantically meaningful latent space is learned, latent vectors can be edited before being passed to the decoder to more precisely control the data generated.

Hierarchical Variational Autoencoders
A Hierarchical Variational Autoencoder (HVAE) [2, 3] is a generalization of a VAE that extends to multiple hierarchies over latent variables. Under this formulation, latent variables themselves are interpreted as generated from other higher-level, more abstract latents. Intuitively, just as we treat our three-dimensional observed objects as generated from a higher-level abstract latent, the people in Plato’s cave treat threedimensional objects as latents that generate their two-dimensional observations. Therefore, from the perspective of Plato’s cave dwellers, their observations can be treated as modeled by a latent hierarchy of depth two (or more).
Whereas in the general HVAE with T hierarchical levels, each latent is allowed to condition on all previous latents, in this work we focus on a special case which we call a Markovian HVAE (MHVAE). In a MHVAE, the generative process is a Markov chain; that is, each transition down the hierarchy is Markovian, where

5

Figure 2: A Markovian Hierarchical Variational Autoencoder with T hierarchical latents. The generative process is modeled as a Markov chain, where each latent zt is generated only from the previous latent zt+1.

decoding each latent zt only conditions on previous latent zt+1. Intuitively, and visually, this can be seen as simply stacking VAEs on top of each other, as depicted in Figure 2; another appropriate term describing this model is a Recursive VAE. Mathematically, we represent the joint distribution and the posterior of a Markovian HVAE as:

T
p(x, z1:T ) = p(zT )pθ(x|z1) pθ(zt−1|zt)
t=2 T
qφ(z1:T |x) = qφ(z1|x) qφ(zt|zt−1)
t=2

(23) (24)

Then, we can easily extend the ELBO to be:

log p(x) = log p(x, z1:T )dz1:T

(Apply Equation 1)

(25)

= log

p(x,

z1:T )qφ qφ(z1:T

(z1:T |x)

|x)

dz1:T

= log Eqφ(z1:T |x)

p(x, z1:T ) qφ(z1:T |x)

≥ Eqφ(z1:T |x)

log p(x, z1:T ) qφ(z1:T |x)

(Multiply by 1 = qφ(z1:T |x) )

(26)

qφ(z1:T |x)

(Deﬁnition of Expectation)

(27)

(Apply Jensen’s Inequality)

(28)

We can then plug our joint distribution (Equation 23) and posterior (Equation 24) into Equation 28 to produce an alternate form:

Eqφ(z

=== Step-by-Step Diffusion: An Elementary Tutorial (Nakkiran, Preetum; Bradley, Arwen; Zhou, Hattie; Advani, Madhu) ===
Step-by-Step Diffusion: An Elementary Tutorial
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
tion into a noise distribution. Equation (1) defines a joint distribution over all (x0, x1, . . . , xT), and we let {pt}t∈[T] denote the marginal distributions of each xt. Notice that at large step count T, the distribution pT is nearly Gaussian3, so we can approximately sample from pT by just sampling a Gaussian.

1 These stand for Denoising Diffusion Probabilistic Models (DDPM) and Denoising Diffusion Implicit Models (DDIM), following Ho et al. [2020] and Song et al. [2021].
2 One benefit of using this particular forward process is computational: we can directly sample xt given x0 in constant time.
3 Formally, pT is close in KL divergence to N (0, Tσ2), assuming p0 has bounded moments.

Figure 1: Probability distributions defined by diffusion forward process on one-dimensional target distribution p0.

step-by-step diffusion: an elementary tutorial 4

Now, suppose we can solve the following subproblem:

“Given a sample marginally distributed as pt, produce a sample marginally distributed as pt−1”.

We will call a method that does this a reverse sampler4, since it tells us how to sample from pt−1 assuming we can already sample from pt. If we had a reverse sampler, we could sample from our target p0 by simply starting with a Gaussian sample from pT, and iteratively applying the reverse sampling procedure to get samples from pT−1, pT−2, . . . and finally p0 = p∗.
The key insight of diffusion is, learning to reverse each intermediate step can be easier than learning to sample from the target distribution in one step5. There are many ways to construct reverse samplers, but for concreteness let us first see the standard diffusion sampler which we will call the DDPM sampler6.k
The Ideal DDPM sampler uses the obvious strategy: At time t, given input z (which is promised to be a sample from pt), we output a sample from the conditional distribution

p(xt−1 | xt = z).

(2)

This is clearly a correct reverse sampler. The problem is, it requires learning a generative model for the conditional distribution p(xt−1 | xt) for every xt, which could be complicated. But if the per-step noise σ is sufficiently small, then it turns out this conditional distribution becomes simple:

Fact 1 (Diffusion Reverse Process). For small σ, and the Gaussian diffu-

sion process defined in (1), the conditional distribution p(xt−1 | xt) is itself close to Gaussian. That is, for all times t and conditionings z ∈ Rd, there

exists some mean parameter µ ∈ Rd such that

p(xt−1 | xt = z) ≈ N (xt−1; µ , σ2).

(3)

This is not an obvious fact; we will derive it in Section 2.1. This fact enables a drastic simplification: instead of having to learn an

4 Reverse samplers will be formally defined in Section 1.2 below.
5 Intuitively this is because the distributions (pt−1, pt) are already quite close, so the reverse sampler does not need to do much. 6 This is the sampling strategy originally proposed in Sohl-Dickstein et al. [2015].

Figure 2: Illustration of Fact 1. The prior distribution p(xt−1), leftmost, defines a joint distribution (xt−1, xt) where p(xt | xt−1) = N (0, σ2). We plot the reverse conditional distributions p(xt−1 | xt) for a fixed condi-
tioning xt, and varying noise levels σ. Notice these distributions become close to Gaussian for small σ.

step-by-step diffusion: an elementary tutorial 5

arbitrary distribution p(xt−1 | xt) from scratch, we now know everything about this distribution except its mean, which we denote7 µt−1(xt). The fact that we can approximate the posterior distribution as Gaussian when σ is sufficiently small is illustrated in Fig 2. This is
an important point, so to re-iterate: for a given time t and conditioning value xt, learning the mean of p(xt−1 | xt) is sufficient to learn the full conditional distribution p(xt−1 | xt).
Learning the mean of p(xt−1 | xt) is a much simpler problem than learning the full conditional distribution, because we can solve it by regression. To elaborate, we have a joint distribution (xt−1, xt) from which we can easily sample, and we would like to estimate E[xt−1 | xt]. This can be done by optimizing a standard regression loss8:

µt−1(z) := E[xt−1 | xt = z]

(4)

=⇒ µt−1 = argmin
f :Rd→Rd

E
xt ,xt−1

||

f

(xt)

−

xt−1 ||22

(5)

=

argmin
f :Rd→Rd

E
xt−1,η

||

f

(xt−1

+ ηt) − xt−1)||22,

(6)

where the expectation is taken over samples x0 from our target distribution p∗.9 This particular regression problem is well-studied in certain settings. For example, when the target p∗ is a distribution on
images, then the corresponding regression problem (Equation 6) is
exactly an image denoising objective, which can be approached with
familiar methods (e.g. convolutional neural networks).

Stepping back, we have seen something remarkable: we have reduced the problem of learning to sample from an arbitrary distribution to the standard problem of regression.

1.2 Diffusions in the Abstract
Let us now abstract away the Gaussian setting, to define diffusionlike models in a way that will capture their many instantiations (including deterministic samplers, discrete domains, and flowmatching).
Abstractly, here is how to construct a diffusion-like generative model: We start with our target distribution p∗, and we pick some base distribution q(x) which is easy to sample from, e.g. a standard Gaussian or i.i.d bits. We then try to construct a sequence of distributions which interpolate between our target p∗ and the base distribution q. That is, we construct distributions

p0 , p1 , p2 , . . . , pT,

(7)

7 We denote the mean as a function µt−1 : Rd → Rd because the mean of p(xt−1 | xt) depends on the time t as well as the conditioning xt, as described in Fact 1.
8 Recall the generic fact that for any distribution over (x, y), we have: argmin f E || f (x) − y||2 = E[y | x]
9 Notice that we simulate samples of (xt−1, xt) by adding noise to the samples of x0, as defined in Equation 1.

step-by-step diffusion: an elementary tutorial 6

such that p0 = p∗ is our target, pT = q the base distribution, and adjacent distributions (pt−1, pt) are marginally “close” in some appropriate sense. Then, we learn a reverse sampler which transforms distributions pt to pt−1. This is the key learning step, which presumably is made easier by the fact that adjacent distributions are “close.” Formally, reverse samplers are defined below.

Definition 1 (Reverse Sampler). Given a sequence of marginal distributions pt, a reverse sampler for step t is a potentially stochastic function Ft such that if xt ∼ pt, then the marginal distribution of Ft(xt) is exactly pt−1:

{Ft(z) : z ∼ pt} ≡ pt−1.

(8)

There are many possible reverse samplers10, and it is even possible to construct reverse samplers which are deterministic. In the remainder of this tutorial we will see three popular reverse samplers more formally: the DDPM sampler discussed above (Section 2.1), the DDIM sampler (Section 3), which is deterministic, and the family of flow-matching models (Section 4), which can be thought of as a generalization of DDIM.11

1.3 Discretization

Before we proceed further, we need to be more precise about what we mean by adjacent distributions pt, pt−1 being “close". We want to think of the sequence p0, p1, . . . , pT as the discretization of some (well-behaved) time-evolving function p(x, t), that starts from the target distribution p0 at time t = 0 and ends at the noisy distribution pT at time t = 1:

p(x, k∆t) = pk(x),

where

∆t

=

1 T

.

(9)

The number of steps T controls the fineness of the discretization

(hence the closeness of adjacent distributions).12

In order to ensure that the variance of the final distribution, pT, is

independent of the number of discretization steps, we also need to

be more specific about the variance of each increment. Note that if

xk = xk−1 + N (0, σ2), then xT ∼ N (x0, Tσ2). Therefore, we need to scale the variance of each increment by ∆t = 1/T, that is, choose

√

σ = σq ∆t,

(10)

where σq2 is the desired terminal variance. This choice√ensures that the variance of pT is always σq2, regardless of T. (The ∆t scaling will turn out to be important in our arguments for the correctness of our reverse solvers in the next chapter, and also connects to the SDE formulation in Section 2.4.)

10 Notice that none of this abstraction is specific to the case of Gaussian noise— in fact, it does not even require the concept of “adding noise”. It is even possible to instantiate in discrete settings, where we consider distributions p∗ over a finite set, and define corresponding “interpolating distributions” and reverse samplers. 11 Given a set of marginal distributions {pt}, there are many possible joint distributions consistent with these marginals (such joint distributions are called couplings). There is therefore no canonical reverse sampler for a given set of marginals {pt} — we are free to chose whichever coupling is most convenient.
12 This naturally suggests taking the continuous-time limit, which we discuss in Section 2.4, though it is not needed for most of our arguments.

step-by-step diffusion: an elementary tutorial 7

At this point, it is convenient to adjust our notation. From here on, t will represent a continuous-value in the interval [0, 1] (specifically, taking one of the values 0, ∆t, 2∆t, . . . , T∆t = 1). Subscripts will indicate time rather than index, so for example xt will now denote x at a discretized time t. That is, Equation 1 becomes:

xt+∆t := xt + ηt, ηt ∼ N (0, σq2∆t),

(11)

which also implies that

√

xt ∼ N (x0, σt2), where σt := σq t,

(12)

since the total noise added up to time t (i.e. ∑τ∈{0,∆t,2∆t,...,t−∆t} ητ) is also Gaussian with mean zero and variance ∑τ σq2∆t = σq2t.

step-by-step diffusion: an elementary tutorial 8

2 Stochastic Sampling: DDPM

In this section we review the DDPM-like reverse sampler discussed in Section 1, and heuristically prove its correctness. This sampler is conceptually the same as the sampler popularized in Denoising Diffusion Probabilistic Models (DDPM) by Ho et al. [2020] and originally introduced by Sohl-Dickstein et al. [2015], when adapted to our simplified setting. However, a word of warning for the reader familiar with Ho et al. [2020]: Although the overall strategy of our sampler is identical to Ho et al. [2020], certain technical details (like constants, etc) are slightly different13.
We consider the setup from Section 1.3, with some target distribution p∗ and the joint distribution of noisy samples (x0, x∆t, . . . , x1) defined by Equation (11). The DDPM sampler will require estimates of the following conditional expectations:

µt(z) := E[xt | xt+∆t = z].

(13)

This is a set of functions {µt}, one for every time step t ∈ {0, ∆t, . . . , 1 − ∆t}. In the training phase, we estimate these functions from i.i.d. samples of x0, by optimizing the denoising regression objective

µt = argmin
f :Rd→Rd

E
xt,xt+∆t

|| f (xt+∆t)

−

xt ||22

,

(14)

typically with a neural-network14 parameterizing f . Then, in the inference phase, we use the estimated functions in the following reverse sampler.

Algorithm 1: Stochastic Reverse Sampler (DDPM-like) For input sample xt, and timestep t, output:

xt−∆t ← µt−∆t(xt) + N (0, σq2∆t)

(15)

To actually generate a sample, we first sample x1 as an isotropic Gaussian x1 ∼ N (0, σq2), and then run the iteration of Algorithm 1 down to t = 0, to produce a generated sample x0. (Recall that in our discretized notation (12), x1 is the fully-noised terminal distribution, and the iteration takes steps of size ∆t.) Explicit pseudocode for these
algorithms are given in Section 2.2.
We want to reason about correctness of this entire procedure: why
does iterating Algorithm 1 produce a sample from [approximately] our target distribution p∗? The key missing piece is, we need to prove
some version of Fact 1: that the true conditional p(xt−∆t | xt) can be well-approximated by a Gaussian, and this approximation gets better as we scale ∆t → 0.

13 For the experts, the main difference is we use the “Variance Exploding” diffusion forward process. We also use a constant noise schedule, and we do not discuss how to parameterize the predictor (“predicting x0 vs. xt−1 vs. noise η”). We elaborate on the latter point in Section 2.3.
14 In practice, it is comm

=== Mean Flows for One-step Generative Modeling (Geng, Zhengyang; Deng, Mingyang; Bai, Xingjian; Kolter, J. Zico; He, Kaiming) ===
arXiv:2505.13447v1 [cs.LG] 19 May 2025
Mean Flows for One-step Generative Modeling
Zhengyang Geng1∗ Mingyang Deng2 Xingjian Bai2 J. Zico Kolter1 Kaiming He2
1CMU 2MIT
Abstract
We propose a principled and effective framework for one-step generative modeling. We introduce the notion of average velocity to characterize flow fields, in contrast to instantaneous velocity modeled by Flow Matching methods. A well-defined identity between average and instantaneous velocities is derived and used to guide neural network training. Our method, termed the MeanFlow model, is self-contained and requires no pre-training, distillation, or curriculum learning. MeanFlow demonstrates strong empirical performance: it achieves an FID of 3.43 with a single function evaluation (1-NFE) on ImageNet 256×256 trained from scratch, significantly outperforming previous state-of-the-art one-step diffusion/flow models. Our study substantially narrows the gap between one-step diffusion/flow models and their multi-step predecessors, and we hope it will motivate future research to revisit the foundations of these powerful models.
235 236 237 238 239 240
Training Compute (GFLOPs, log-scale)
0
5
10
15
20
25
30
35
1-step FID
MF-B MF-M
iCT-XL
Shortcut-XL
IMM-XL
MF-L MF-XL
B: 131M M: 308M
L: 459M
XL: 675M Figure 1: One-step generation on ImageNet 256×256 from scratch. Our MeanFlow (MF) model achieves significantly better generation quality than previous state-of-the-art one-step diffusion/flow methods. Here, iCT [43], Shortcut [13], and our MF are all 1-NFE generation, while IMM’s 1-step result [52] involves 2-NFE guidance. Detailed numbers are in Tab. 2. Images shown are generated by our 1-NFE model.
1 Introduction
The goal of generative modeling is to transform a prior distribution into the data distribution. Flow Matching [28, 2, 30] provides an intuitive and conceptually simple framework for constructing flow paths that transport one distribution to another. Closely related to diffusion models [42, 44, 19], Flow Matching focuses on the velocity fields that guide model training. Since its introduction, Flow Matching has seen widespread adoption in modern generative modeling [11, 33, 35].
Both Flow Matching and diffusion models perform iterative sampling during generation. Recent research has paid significant attention to few-step—and in particular, one-step, feedforward—generative models. Pioneering this direction, Consistency Models [46, 43, 15, 31] introduce a consistency constraint to network outputs for inputs sampled along the same path. Despite encouraging results, the consistency constraint is imposed as a property of the network’s behavior, while the properties of the underlying ground-truth field that should guide learning remain unknown. Consequently, training can be unstable and requires a carefully designed “discretization curriculum” [46, 43, 15] to progressively constrain the time domain.
∗Work partly done when visiting MIT.


In this work, we propose a principled and effective framework, termed MeanFlow, for one-step generation. The core idea is to introduce a new ground-truth field representing the average velocity, in contrast to the instantaneous velocity typically modeled in Flow Matching. Average velocity is defined as the ratio of displacement to a time interval, with displacement given by the time integral of the instantaneous velocity. Solely originated from this definition, we derive a well-defined, intrinsic relation between the average and instantaneous velocities, which naturally serves as a principled basis for guiding network training.
Building on this fundamental concept, we train a neural network to directly model the average velocity field. We introduce a loss function that encourages the network to satisfy the intrinsic relation between average and instantaneous velocities. No extra consistency heuristic is needed. The existence of the ground-truth target field ensures that the optimal solution is, in principle, independent of the specific network, which in practice can lead to more robust and stable training. We further show that our framework can naturally incorporate classifier-free guidance (CFG) [18] into the target field, incurring no additional cost at sampling time when guidance is used.
Our MeanFlow Models demonstrate strong empirical performance in one-step generative modeling. On ImageNet 256×256 [7], our method achieves an FID of 3.43 using 1-NFE (Number of Function Evaluations) generation. This result significantly outperforms previous state-of-the-art methods in its class by a relative margin of 50% to 70% (Fig. 1). In addition, our method stands as a selfcontained generative model: it is trained entirely from scratch, without any pre-training, distillation, or curriculum learning. Our study largely closes the gap between one-step diffusion/flow models and their multi-step predecessors, and we hope it will inspire future work to reconsider the foundations of these powerful models.
2 Related Work
Diffusion and Flow Matching. Over the past decade, diffusion models [42, 44, 19, 45] have been developed into a highly successful framework for generative modeling. These models progressively add noise to clean data and train a neural network to reverse this process. This procedure involves solving stochastic differential equations (SDE), which is then reformulated as probability flow ordinary differential equations (ODE) [45, 22]. Flow Matching methods [28, 2, 30] extend this framework by modeling the velocity fields that define flow paths between distributions. Flow Matching can also be viewed as a form of continuous-time Normalizing Flows [36].
Few-step Diffusion/Flow Models. Reducing sampling steps has become an important consideration from both practical and theoretical perspectives. One approach is to distill a pre-trained many-step diffusion model into a few-step model, e.g., [39, 14, 41] or score distillation [32, 50, 53]. Early explorations into training few-step models [46] are built upon the evolution of distillation-based methods. Meanwhile, Consistency Models [46] are developed as a standalone generative model that does not require distillation. These models impose consistency constraints on network outputs at different time steps, encouraging them to produce the same endpoints along the trajectory. Various consistency models and training strategies [46, 43, 15, 31, 49] have been investigated.
In recent work, several methods have focused on characterizing diffusion-/flow-based quantities with respect to two time-dependent variables. In [3], a Flow Map is defined as the integral of the flow between two time steps, with several forms of matching losses developed for learning. In comparison to the average velocity our method is based on, the Flow Map corresponds to displacement. Shortcut Models [13] introduce a self-consistency loss function in addition to Flow Matching, which captures relationships between the flows at different discrete time intervals. Inductive Moment Matching [52] models the self-consistency of stochastic interpolants at different time steps.
3 Background: Flow Matching
Flow Matching [28, 30, 1] is a family of generative models that learn to match the flows, represented by velocity fields, between two probabilistic distributions. Formally, given data x ∼ pdata(x) and prior ε ∼ pprior(ε), a flow path can be constructed as zt = atx + btε with time t, where at and bt are
predefined schedules. The velocity vt is defined as vt = zt′ = a′tx + b′tε, where ′ denotes the time
derivative. This velocity is referred to as the conditional velocity in [28], denoted by vt = vt(zt | x).
See Fig. 2 left. A commonly used schedule is at = 1 − t and bt = t, which leads to vt = ε − x.
2


vt
zt
v(zt, t)
Figure 2: Velocity fields in Flow Matching [28]. Left: conditional flows [28]. A given zt can arise from different (x, ε) pairs, resulting in different conditional velocities vt. Right: marginal flows [28], obtained by marginalizing over all possible conditional velocities. The marginal velocity field serves as the underlying ground-truth field for network training. All velocities shown here are essentially instantaneous velocities. Illustration follows [12]. (Gray dots: samples from prior; red dots: samples from data.)
Because a given zt and its vt can arise from different x and ε, Flow Matching essentially models the expectation over all possibilities, called the marginal velocity [28] (Fig. 2 right):
v(zt, t) ≜ Ept(vt|zt)[vt]. (1)
A neural network vθ parameterized by θ is learned to fit the marginal velocity field: LFM(θ) =
Et,pt(zt)∥vθ(zt, t) − v(zt, t)∥2. Although computing this loss function is infeasible due to the marginalization in Eq. (1), it is proposed to instead evaluate the conditional Flow Matching loss [28]: LCFM(θ) = Et,x,ε∥vθ(zt, t) − vt(zt | x)∥2, where the target vt is the conditional velocity.
Minimizing LCFM is equivalent to minimizing LFM [28].
Given a marginal velocity field v(zt, t), samples are generated by solving an ODE for zt:
d
dt zt = v(zt, t) (2)
starting from z1 = ε ∼ pprior. The solution can be written as: zr = zt − R t
r v(zτ , τ )dτ , where
we use r to denote another time step. In practice, this integral is approximated numerically over discrete time steps. For example, the Euler method, a first-order ODE solver, computes each step as: zti+1 = zti + (ti+1 − ti)v(zti , ti). Higher-order solvers can also be applied.
It is worth noting that even when the conditional flows are designed to be straight (“rectified") [28, 30], the marginal velocity field (Eq. (1)) typically induces a curved trajectory. See Fig. 2 for illustration. We also emphasize that this non-straightness is not only a result of neural network approximation, but rather arises from the underlying ground-truth marginal velocity field. When applying coarse discretizations over curved trajectories, numerical ODE solvers lead to inaccurate results.
4 MeanFlow Models
4.1 Mean Flows
The core idea of our approach is to introduce a new field representing average velocity, whereas the velocity modeled in Flow Matching represents the instantaneous velocity.
Average Velocity. We define average velocity as the displacement between two time steps t and r (obtained by integration) divided by the time interval. Formally, the average velocity u is:
u(zt, r, t) ≜ 1
t−r
Zt
r
v(zτ , τ )dτ. (3)
To emphasize the conceptual difference, throughout this paper, we use the notation u to denote average velocity, and v to denote instantaneous velocity. u(zt, r, t) is a field that is jointly dependent on (r, t). The field of u is illustrated in Fig. 3. Note that in general, the average velocity u is the result of a functional of the instantaneous velocity v: that is, u = F[v] ≜ 1
t−r
Rt
r vdτ . It is a field induced
by v, not depending on any neural network. Conceptually, just as the instantaneous velocity v serves as the ground-truth field in Flow Matching, the average velocity u in our formulation provides an underlying ground-truth field for learning.
By definition, the field of u satisfies certain boundary conditions and “consistency” constraints (generalizing the terminology of [46]). As r→t, we have: limr→t u = v. Moreover, a form of “consistency" is naturally satisfied: taking one larger step over [r, t] is “consistent" with taking two smaller consecutive steps over [r, s] and [s, t], for any intermediate time s. To see this, observe that (t − r)u(zt, r, t) = (s − r)u(zs, r, s) + (t − s)u(zt, s, t), which follows directly from the additivity
3


t
r u(z, r, t)
(t−r)u(z, r, t)
t = 0.5
u(z, r, t)
t = 0.7
u(z, r, t)
t = 1.0
u(z, r, t)
v
Figure 3: The field of average velocity u(z, r, t). Leftmost: While the instantaneous velocity v determines the tangent direction of the path, the average velocity u(z, r, t), defined in Eq. (3), is generally not aligned with v. The average velocity is aligned with the displacement, which is (t − r)u(z, r, t). Right three subplots: The field u(z, r, t) is conditioned on both r and t, and is shown here for t = 0.5, 0.7, and 1.0.
of the integral: R t
r vdτ = R s
r vdτ + R t
s vdτ . Thus, a network that accurately approximates the true u
is expected to satisfy the consistency relation inherently, without the need for explicit constraints.
The ultimate aim of our MeanFlow model will be to approximate the average velocity using a neural network uθ(zt, r, t). This has the notable advantage that, assuming we approximate this quantity accurately, we can approximate the entire flow path using a single evaluation of uθ(ε, 0, 1). In other words, and as we will also demonstrate empirically, the approach is much more amenable to single or few-step generation, as it does not need to explicitly approximate a time integral at inference time, which was required when modeling instantaneous velocity. However, directly using the average velocity defined by Eq. (3) as ground truth for training a network is intractable, as it requires evaluating an integral during training. Our key insight is that the definitional equation of average velocity can be manipulated to construct an optimization target that is ultimately amenable to training, even when only the instantaneous velocity is accessible.
The MeanFlow Identity. To have a formulation amenable to training, we rewrite Eq. (3) as:
(t − r)u(zt, r, t) =
Zt
r
v(zτ , τ )dτ. (4)
Now we differentiate both sides with respect to t, treating r as independent of t. This leads to:
d
dt (t − r)u(zt, r, t) = d
dt
Zt
r
v(zτ , τ )dτ =⇒ u(zt, r, t) + (t − r) d
dt u(zt, r, t) = v(zt, t), (5)
where the manipulation of the left hand side employs the product rule and the right hand side uses the fundamental theorem of calculus2. Rearranging terms, we obtain the identity:
u(zt, r, t)
| {z }
average vel.
= v(zt, t)
| {z }
instant. vel.
−(t − r) d
dt u(zt, r, t)
| {z }
time derivative
(6)
We refer to this equation as the “MeanFlow Identity", which describes the relation between v and u. It is easy to show that Eq. (6) and Eq. (4) are equivalent (see Appendix B.3).
The right hand side of Eq. (6) provides a “target" form for u(zt, r, t), which we will leverage to construct a loss function to train a neural network. To serve as a suitable target, we must also further decompose the time derivative term, which we discuss next.
Computing Time Derivative. To compute the d
dt u term in Eq. (6), note that d
dt denotes a total
derivative, which can be expanded in terms of partial derivatives:
d
dt u(zt, r, t) = dzt
dt ∂zu + dr
dt ∂ru + dt
dt ∂tu. (7)
With dzt
dt = v(zt, t) (see Eq. (2)), dr
dt = 0, and dt
dt = 1, we have another relation between u and v:
d
dt u(zt, r, t) = v(zt, t)∂zu + ∂tu, (8)
2If r depends on t, the Leibniz rule [26] gives: d
dt
Rt
r v(zτ , τ )dτ = v(zt, t) − v(zr, r) dr
dt .
4


This equation shows that the total derivative is given by the Jacobian-vector product (JVP) between [∂zu, ∂ru, ∂tu] (the Jacobian matrix of the function u) and the tangent vector [v, 0, 1]. In modern libraries, this can be efficiently computed by the jvp interface, such as torch.func.jvp in PyTorch or jax.jvp in JAX, which we discuss later.
Training with Average Velocity. Up to this point, the formulations are independent of any network parameterization. We now introduce a model to learn u. Formally, we parameterize a network uθ and encourage it to satisfy the MeanFlow Identity (Eq. (6)). Specifically, we minimize this objective:
L(θ) = E uθ(zt, r, t) − sg(utgt) 2
2, (9)
where utgt = v(zt, t) − (t − r) (v(zt, t)∂zuθ + ∂tuθ) , (10)
The term utgt serves as the effective regression target, which is driven by Eq. (6). This target uses the instantaneous velocity v as the only ground-truth signal; no integral computation is needed. While the target should involve derivatives of u (that is, ∂u), they are replaced by their parameterized counterparts (that is, ∂uθ). In the loss function, a stop-gradient (sg) operation is applied on the target utgt, following common practice [46, 43, 15, 31, 13]: in our case, it eliminates the need for “double backpropagation” through the Jacobian-vector product, thereby avoiding higher-order optimization. Despite these practices for optimizability, if uθ were to achieve zero loss, it is easy to show that it would satisfy the MeanFlow Identity (Eq. (6)), and thus satisfy the original definition (Eq. (3)).
The velocity v(zt, t) in Eq. (10) is the marginal velocity in Flow Matching [28] (see Fig. 2 right). We follow [28] to replace it with the conditional velocity (Fig. 2 left). With this, the target is:
utgt = vt − (t − r) vt∂zuθ + ∂tuθ . (11)
Recall that vt = a′tx + b′tε is the conditional velocity [28], and by default, vt = ε − x.
Pseudocode for minimizing the loss function Eq. (9) is presented in Alg. 1. Overall, our method is conceptually simple: it behaves similarly to Flow Matching, with the key difference that the matching target is modified by −(t−r) (vt∂zuθ + ∂tuθ), arising from our consideration of the average velocity. In particular, note that if we were to restrict to the condition t = r, then the second term vanishes, and the method would exactly match standard Flow Matching.
Algorithm 1 MeanFlow: Training.
Note: in PyTorch and JAX, jvp returns the function output and JVP.
# fn(z, r, t): function to predict u # x: training batch
t, r = sample_t_r() e = randn_like(x)
z = (1 - t) * x + t * e v=e-x
u, dudt = jvp(fn, (z, r, t), (v, 0, 1))
u_tgt = v - (t - r) * dudt error = u - stopgrad(u_tgt)
loss = metric(error)
Algorithm 2 MeanFlow: 1-step Sampling
e = randn(x_shape) x = e - fn(e, r=0, t=1)
In Alg. 1, the jvp operation is highly efficient. In essence, computing d
dt u via jvp requires only
a single backward pass, similar to standard backpropagation in neural networks. Because d
dt u is
part of the target utgt and thus subject to stopgrad (w.r.t. θ), the backpropagation for neural network optimization (w.r.t. θ) treats d
dt u as a constant,
incurring no higher-order gradient computation. Consequently, jvp intro

=== Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design (Yim, Jason; Barzilay, Regina; Jaakkola, Tommi; Campbell, Andrew; Rainforth, Tom;) ===
Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design

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
Fig. 1A provides an overview of DFMs. We first define a probability flow pt that linearly interpolates from noise to data. We then generate new data by simulating a sequence trajectory xt that follows pt across time which requires training a denoising neural network with cross-entropy. The sequence trajectory could have many transitions or few, a property we term CTMC Stochasticity (Fig. 1B). Prior discrete diffusion models are equivalent to picking a specific stochasticity at training time, whereas we can adjust it at inference: enhancing sample quality and exerting control

Discrete Flow Models

Figure 1. Overview. (A.) A DFM trajectory with masking over a 3-dim. sequence with 4 possible states. (B.) CTMC stochasticity controls the number of transitions in a sequence trajectory while respecting the flow pt. Shown is a 1-dim. sequence with 5 states. (C.) Sampling with Multiflow can start from noise (bottom left) or with either the structure or sequence given (top left and bottom right). Any sampling tasks (structure/sequence generation, forward/inverse folding, co-generation) can be achieved with a single Multiflow model.

over sample distributional properties.
Using DFMs, we are then able to create a multimodal flow model by defining factorized flows for each data modality. We apply this capability to the task of protein co-design by developing a novel continuous structure and discrete sequence generative model named Multiflow. We combine a DFM for sequence generation and a flow-based structure generation method developed in Yim et al. (2023a). Previous multimodal approaches either generated only the sequence or only the structure and then used a prediction model to infer the remaining modality (see Sec. 5). Our single model can jointly generate sequence and structure while being able to condition on either modality.
In our experiments (Sec. 6), we first verify on small scale text data that DFMs outperform the discrete diffusion alternative, D3PM (Austin et al., 2021) through their expanded sample time flexibility. We then move to our main focus, assessing Multiflow’s performance on the co-design task of jointly generating protein structure and sequence. Multiflow achieves state-of-the-art co-design performance while data distillation allows for obtaining state-of-the-art structure generation. We find CTMC stochasticity enables controlling sample properties such as secondary structure composition and diversity. Preliminary results on inverse and forward folding show Multiflow is a promising path towards a general-purpose protein generative model.
Our contributions are summarized as follows:
• We present Discrete Flow Models (DFMs), a novel discrete generative modeling method built through a CTMC simulating a probability flow.
• We combine DFMs with continuous flow-based methods to create a multimodal generative modeling framework.
• We use our multimodal framework to develop Multiflow, a state-of-the-art generative protein co-design model with the flexibility of multimodal protein generation.

2. Background
We aim to model discrete data where a sequence x ∈ {1, . . . , S}D has D dimensions, each taking on one of S states. For ease of exposition, we will assume D = 1; all results hold for D > 1 as discussed in App. E. We first explain a class of continuous time discrete stochastic processes called Continuous Time Markov Chains (CTMCs) (Norris, 1998) and then describe the link to probability flows.

2.1. Continuous Time Markov Chains.

A sequence trajectory xt over time t ∈ [0, 1] that follows a CTMC alternates between resting in its current state and
periodically jumping to another randomly chosen state. We
show example trajectories in Fig. 1B. The frequency and
destination of the jumps are determined by the rate matrix Rt ∈ RS×S with the constraint its off-diagonal elements are non-negative. The probability xt will jump to a different state j is Rt(xt, j)dt for the next infinitesimal time step dt . We can write the transition probability as

pt+dt|t(j|xt) =

Rt(xt, j)dt 1 + Rt(xt, xt)dt

for j ̸= xt for j = xt

(1)

= δ {xt, j} + Rt(xt, j)dt

(2)

where δ {i, j} is the Kronecker delta which is 1 when i = j
and is otherwise 0 and Rt(xt, xt) := − k̸=x Rt(xt, k) in order for pt+dt|t(·|i) to sum to 1. We use compact notation Eq. (2) in place of Eq. (1). Therefore, pt+dt|t is a Categorical distribution with probabilities δ {xt, ·} + Rt(xt, ·)dt that we denote as Cat(δ {xt, j} + Rt(xt, j)dt):

j ∼ pt+dt|t(j|xt) ⇐⇒ j ∼ Cat(δ {xt, j} + Rt(xt, j)dt).

In practice, we need to simulate the sequence trajectory with finite time intervals ∆t. A sequence trajectory can be simulated with Euler steps (Sun et al., 2023b)

xt+∆t ∼ Cat(δ {xt, xt+∆t} + Rt(xt, xt+∆t)∆t), (3)

Discrete Flow Models

where the sequence starts from an initial sample x0 ∼ p0 at time t = 0. The rate matrix Rt along with an initial distribution p0 together define the CTMC.

2.2. Kolmogorov equation

For a sequence trajectory following the dynamics of a CTMC, we write its marginal distribution at time t as pt(xt). The Kolmogorov equation allows us to relate the rate matrix Rt to the change in pt(xt). It has the form:

∂tpt(xt) = Rt(j, xt)pt(j) − Rt(xt, j)pt(xt) (4)

j̸=xt

j̸=xt

incoming

outgoing

The difference between the incoming and outgoing
probability mass is the time derivative of the marginal ∂tpt(xt). Using our definition of Rt(xt, xt), Eq. (4) can be succinctly written as ∂tpt = Rt⊤pt where the marginals are treated as probability mass vectors: pt ∈ [0, 1]S. This defines an Ordinary Differential Equation (ODE) in a vector space. We refer to the series of distributions pt ∀t ∈ [0, 1] satisfying the ODE as a probability flow.

Key terms: A CTMC is defined by an initial distribution
p0 and rate matrix Rt. Samples along CTMC dynamics are called a sequence trajectory xt. The probability flow pt is the marginal distribution of xt at every time t. We say Rt generates pt if ∂tpt = Rt⊤pt ∀t ∈ [0, 1].

3. Discrete Flow Models
A Discrete Flow Model (DFM) is a Discrete data generative model built around a probability Flow that interpolates from noise to data. To sample new datapoints, we simulate a sequence trajectory that matches the noise to data probability flow. The flow construction allows us to combine DFM with continuous data flow models to define a multimodal generative model. Proofs for all propositions are in App. B.

3.1. A Flow Model for Sampling Discrete Data

We start by constructing the data generating probability flow referred to as the generative flow, pt, that we will later sample from using a CTMC. The generative flow interpolates from noise to data where p0(x0) = pnoise(x0) and p1(x1) = pdata(x1). Since pt is complex to consider directly, the insight of flow matching is to define pt using a simpler datapoint conditional flow, pt|1(·|x1) that we will be able to write down explicitly. We can then define pt as

pt(xt) := Epdata(x1) pt|1(xt|x1) .

(5)

The conditional flow, pt|1(·|x1) interpolates from noise to the datapoint x1. The conditioning allows us to write the flow down in closed form. We are free to define pt|1(·|x1)

as needed for the specific application. The conditional flows we use in this paper linearly interpolate towards x1 from a uniform prior or an artificially introduced mask state, M :

put|n1if

(xt|x1)

=

Cat(tδ

{x1,

xt}

+

(1

−

t)

1 S

),

(6)

pmt|1ask(xt|x1) = Cat(tδ {x1, xt} + (1 − t)δ {M, xt}).

We require our conditional flow to converge on the datapoint

x1 at t = 1, i.e. pt|1(xt|x1) = δ {x1, xt}. We also require

that the conditional flow starts from noise at t = 0, i.e.

pt|1(xt|x1) = pnoise(xt). In our examples, punnoiisfe(xt) =

1 S

and pmnoaisske (xt)

=

δ {M, xt}.

These two requirements

ensure our generative flow, pt, defined in Eq. (5) interpolates

from pnoise at t = 0 towards pdata at t = 1 as desired. Next,

we will show how to sample from the generative flow by

exploiting pt’s decomposition into conditional flows.

3.1.1. SAMPLING

To sample from pdata using the generative flow, pt, we need access to a rate matrix Rt(xt, j) that generates pt. Given a Rt(xt, j), we could use Eq. (3) to simulate a sequence trajectory that begins with marginal distribution pnoise at t = 0 and ends with marginal distribution pdata at t = 1. The definition of pt in Eq. (5) suggests Rt(xt, j) can also be derived as an expectation over a simpler conditional rate

matrix. Define Rt(xt, j|x1) as a datapoint conditional rate matrix that generates pt|1(xt|x1). We now show Rt(xt, j) can indeed be defined as an expectation over Rt(xt, j|x1).

Proposition 3.1. If Rt(xt, j|x1) is a rate matrix that gener-

ates the conditional flow pt|1(xt|x1), then

Rt(xt, j) := Ep1|t(x1|xt) [Rt(xt, j|x1)]

(7)

is a rate matrix that generates pt defined in Eq. (5). The

expectation

is

taken

over

p1|t(x1|xt)

=

. pt|1 (xt |x1 )pdata (x1 )
pt (xt )

Our aim now is to calculate Rt(xt, j|x1) and p1|t(x1|xt) to plug into Eq. (7). p1|t(x1|xt) is the distribution predicting clean data x1 from noisy data xt and in Sec. 3.1.2, we will train a neural network pθ1|t(x1|xt) to approximate it. In Sec. 3.2, we will show how to derive Rt(xt, j|x1) in closed
form. Sampling pseudo-code is provided in Alg. 1.

Algorithm 1 DFM Sampling
1: init t = 0, x0 ∼ p0, choice of Rt(xt, ·|x1) (Sec. 3.2) 2: while t < 1 do 3: Rtθ(xt, ·) ← Epθ1|t(x1|xt) [Rt(xt, ·|x1)] 4: xt+∆t ∼ Cat δ {xt, xt+∆t} + Rtθ(xt, xt+∆t)∆t 5: t ← t + ∆t
6: end while 7: return x1

We discuss further CTMC sampling methods in App. G. Our construction of the generative flow from conditional flows

Discrete Flow Models

Table 1. Comparison between continuous space linear interpolant flow models and DFMs with masking. Both start with a conditional flow pt|1(xt|x1) interpolating between data and noise. For continuous, pt|1(xt|x1) = N (tx1, (1 − t)2I) and for discrete we use pm t|1ask. Solving the Fokker-Planck or Kolmogorov equations with pt|1(xt|x1) gives a data conditioned process, specified either by the velocity field (νt) or the rate matrix (Rt). We train a model to learn the unconditional process – written analytically as the expected value of the
conditional quantity – which is then used for sampling. The side-by-side comparison reveals the similar forms of each quantity.

QUANTITY
FOKKER-PLANCK-KOLMOGOROV CONDITIONAL PROCESS GENERATIVE PROCESS GENERATIVE SAMPLING

CONTINUOUS

∂tpt = −∇ · (vtpt)

νt(xt|x1)

=

xt −x1 1−t

νt(xt) = Ep1|t(x1|xt) [νt(xt|x1)]

xt+∆t = xt + vt(xt)∆t

DISCRETE

∂tpt = Rt⊤pt

Rt(xt, j|x1)

=

δ{j,x1 } 1−t

δ

{xt,

M

}

Rt(xt, j) = Ep1|t(x1|xt) [R(xt, j|x1)]

xt+∆t ∼ Cat(δ {xt, xt+∆t} + Rt(xt, xt+∆t)∆t)

is analogous to the construction of generative probability paths from conditional probability paths in Lipman et al. (2023), where instead of a continuous vector field generating the probability path, we have a rate matrix generating the probability flow. We expand on these links in Table. 1.

3.1.2. TRAINING
We train a neural network with parameters θ, pθ1|t(xt|x1), to approximate the true denoising distribution using the standard cross-entropy i.e. learning to predict the clean datapoint x1 when given noisy data xt ∼ pt|1(xt|x1).

L = E ce

pdata(x1)U (t;0,1)pt|1(xt|x1)

log pθ1|t(x1|xt)

(8)

where U(t; 0, 1) is a uniform distribution on [0, 1]. xt can be sampled from pt|1(xt|x1) in a simulation-free manner by using the explicit form we wrote down for pt|1 e.g. Eq. (6). In App. C, we analyse how Lce relates to the model loglikelihood and its relation to the Evidence Lower Bound
(ELBO) used to train diffusion models. We stress that Lce does not depend on Rt(xt, j|x1) and so we can postpone the choice of Rt(xt, j|x1) until after training. This enables inference time flexibility in how our discrete data is sampled.

3.2. Choice of Rate Matrix

The missing piece in Eq. (7) is a conditional rate matrix Rt(xt, j|x1) that generates the conditional flow pt|1(xt|x1). There are many choices for Rt(xt, j|x1) that all generate the same pt|1(xt|x1) as we later show in Prop. 3.3. In order to proceed, we start by giving one valid choice of rate matrix and from this, build a set of rate matrices that all generate pt|1. At inference time, we can then pick the rate matrix from this set that performs the best. Our starting choice for a rate matrix that generates pt|1 is defined for xt ̸= j as,

Rt∗(xt, j|x1) := ReLU

∂tpt|1(j|x1) − ∂tpt|1(xt|x1) S · pt|1(xt|x1)

where ReLU(a) = max(a, 0) and ∂tpt|1 can be found by differentiating our explicit form for pt|1. This assumes pt|1(xt|x1) > 0, see App. B.2 for the full form.

We first heuristically justify Rt∗ and then prove it generates pt|1(xt|x1) in Prop. 3.2. Rt∗ can be understood as distributing probability mass to states that require it. If
∂tpt|1(j|x1) > ∂tpt|1(xt|x1) then state j needs to gain more probability mass than the current state xt resulting in a positive rate. If ∂tpt|1(j|x1) ≤ ∂tpt|1(i|x1) then state xt should give no mass to state j hence the ReLU. This rate
should then be normalized by the probability mass in the
current state. The ReLU ensures off-diagonal elements of Rt∗ are positive and is inspired by Zhang et al. (2023).
Proposition 3.2. Assuming zero mass states, pt|1(j|x1) = 0, have ∂tpt|1(j|x1) = 0, then Rt∗ generates pt|1(xt|x1).

The proof is easy to derive by substituting Rt∗ along with pt|1(xt|x1) into the Kolmogorov equation Eq. (4). The forms for Rt∗(xt, j|x1) under put|n1if or pmt|1ask are simple

Rt∗unif

=

δ{x1

,j}(1−δ{x1 1−t

,xt

})

,

Rt∗mask

=

δ{x1,j}δ{xt,M } 1−t

as we derive in App. F. Using Rt∗ as a starting point, we now build out a set of rate matrices that all generate pt|1. We can accomplish this by adding on a second rate matrix
that is in detailed balance with pt|1.
Proposition 3.3. Let RtDB be a rate matrix that satisfies the detailed balance condition for pt|1,

pt|1(i|x1)RtDB(i, j|x1) = pt|1(j|x1)RtDB(j, i|x1), . (9)

Let Rtη be defined by Rt∗, RtDB and parameter η ∈ R≥0, Rtη := Rt∗ + ηRtDB.



=== Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (Lou, Aaron; Ermon, Stefano; Meng, Chenlin) ===
Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution

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

One can simulate this process by taking small ∆t Euler steps and randomly sampling the resulting transitions. In particular, the samples are defined by transition densities which come from the columns of Qt:
p(xt+∆t = y|xt = x) = δxy + Qt(y, x)∆t + O(∆t2) (2)

Finally, this process has a well known reversal (Kelly, 1980; Sun et al., 2023) given by another diffusion matrix Qt:

dpT −t dt

= QT −tpT −t

Qt(y, x)

=

pt(y) pt(x)

Qt

(x,

y)

Qt(x, x) = − Qt(y, x) (3)
y̸=x

This reverse process is analogous to the time reversal for typ-

ical

diffusion

processes

on

Rn,

with

the

ratios

pt (y ) pt (x)

(which

are collectively known as the concrete score (Meng et al.,

2022)) generalizing the typical score function ∇x log pt (Song & Ermon, 2019) 1

1The gradient operator for discrete structures is (up to some

2.2. Discrete Diffusion Models

The goal of a discrete diffusion model is to construct the

aforementioned

reverse

process

by

learning

the

ratios

pt pt

(y) (x)

.

Unlike the continuous diffusion case, which has settled

around (up to minor scaling variations) the theoretical frame-

work given by score matching (Hyva¨rinen, 2005), there cur-

rently exist many competing methods for learning discrete

diffusion models. In particular, these tend to produce mixed

empirical results, which spurs the need for a reexamination.

Mean Prediction. Instead of directly parameterizing the

ratios

pt pt

(y) (x)

,

Austin

et

al.

(2021);

Campbell

et

al.

(2022)

instead follow a strategy of Ho et al. (2020) to learn the re-

verse

density

p0|t.

This

actually

recovers

the

ratios

pt (y ) pt (x)

in

a

roundabout way (as shown in our Theorem 4.2), but comes

with several drawbacks. First, learning p0|t is inherently

harder since it is a density (as opposed to a general value).

Furthermore, the objective breaks down in continuous time

and must be approximated (Campbell et al., 2022). As a

result, this framework largely underperforms empirically.

Ratio Matching. Originally introduced in Hyva¨rinen (2007) and augmented in Sun et al. (2023), ratio matching learns the marginal probabilities of each dimension with maximum likelihood training. However, the resulting setup departs from standard score matching and requires specialized and expensive network architectures (Chen & Duvenaud, 2019). As such, this tends to perform worse than mean prediction.

Concrete Score Matching. Meng et al. (2022) generalizes

the standard Fisher divergence in score matching, learning

sθ(x, t) ≈

pt (y ) pt (x)

with concrete score matching:
y̸=x





1 LCSM = 2 Ex∼pt 
y̸=x

sθ (xt ,

t)y

−

pt(y) pt(x)

2


(4)

Unfortunately, the ℓ2 loss is incompatible with the fact that

pt (y ) pt (x)

must be positive.

In particular, this does not suffi-

ciently penalize negative or zero values, leading to divergent

behavior. Although theoretically promising, Concrete Score

Matching struggles (as seen in Appendix D).

3. Score Entropy Discrete Diffusion Models

In this section, we introduce score entropy. Similar to con-

crete score matching, we learn the collected concrete score

sθ(x, t) ≈

pt (y ) pt (x)

(sθ : X × R → R|X |).
y̸=x

We design

the score entropy loss to incorporate the fact that these ratios

are positive and evolve under a discrete diffusion.

scaling) defined for pairs x ̸= y by ∇f (xy) := f (y) − f (x).

The score function would generalize to the normalized gradients

∇p(xy) p(x)

=

p(y) p(x)

− 1.

2

Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution

Definition 3.1. The score entropy LSE for a distribution p, weights wxy ≥ 0 and a score network sθ(x)y is

dimensions, this is intractable, which means we have to sample y uniformly, but this introduces additional variance analogous to that introduced by the Hutchinson trace esti-


Ex∼p  wxy
y̸=x

p(y) sθ(x)y − p(x) log sθ(x)y + K

p(y) p(x)

mator (Hutchinson, 1989) for sliced score matching (Song  et al., 2019). As a result, implicit score entropy is impracti cal for large-scale tasks. Instead, we work a denoising score
matching loss (Vincent, 2011) variant of score entropy:

(5)

where K(a) = a(log a − 1) is a normalizing constant func-

tion that ensures that LSE ≥ 0.

Remark. Instead of building off of Fisher divergences,

score entropy builds off of the Bregman divergence

DF

s(x)y ,

p(y) p(x)

when F = − log is the convex function.

As such, score entropy is non-negative, symmetric, and con-

vex. It also generalizes standard cross entropy to general

positive values (instead of simplex-valued probabilities), in-

spiring the name. The weights wxy are used primarily when combining score entropy with diffusion models.

While this expression is more complex than the standard score matching variants, it satisfies several desiderata for a discrete diffusion training objective:

3.1. Score Entropy Properties First, score entropy is a suitable loss function that recovers

Theorem 3.4 (Denoising Score Entropy). Suppose p is a
perturbation of a base density p0 by a transition kernel p(·|·), ie p(x) = x0 p(x|x0)p0(x0). The score entropy LSE is equivalent (up to a constant independent of θ) to the denoising score entropy LDSE is



x0E∼p0 

wxy

x∼p(·|x0) y̸=x



sθ (x)y

−

p(y|x0) p(x|x0)

log

sθ (x)y



(7)

LDSE is scalable since Monte Carlo sampling only requires the evaluation of one sθ(x), which gives us all sθ(x)y, and the variance introduced by x0 is manageable. Additionally, it is particularly appealing for discrete diffusion since the intermediate pt are all perturbations of the base density p0 (resulting from Equations 1, 2), enabling us to train with LDSE using the diffusion transition densities pt|0(·|x0) (which we can make tractable).

the ground truth concrete score.

Proposition 3.2 (Consistency of Score Entropy). Suppose

p is fully supported and wxy > 0. As the number of samples and model capacity approaches ∞, the optimal θ∗ that

minimizes

Equation

5

satisfies

sθ∗ (x)y

=

p(y) p(x)

for

all

pairs

x, y Furthermore, LSE will be 0 at θ∗.

Second, score entropy directly improves upon concrete

score matching by rescaling problematic gradients. For

the

weights

wxy

=

1,

∇sθ(x)y LSE

=

sθ

1 (x)y

∇sθ

(x)y

LCSM

,

so the gradient signals for each pair (x, y) are scaled by a

factor of sθ(x)y as a normalization component. As such, this forms a natural log-barrier which keeps our sθ ≥ 0.

Third, similar to concrete score matching, score entropy

can be made computationally tractable by removing the

unknown

p(y) p(x)

term.

There

are

two

alternative

forms,

the

first of which is analogous to the implicit score matching

loss (Hyva¨rinen, 2005):

3.2. Likelihood Bound For Score Entropy Discrete Diffusion

Fourth, the score entropy can be used to define an ELBO for likelihood-based training and evaluation.

Definition 3.5. For our time dependent score network

sθ(·, t), the parameterized reverse matrix is Qθt (y, x) =

sθ(x, t)yQt(x, y) − z̸=x Qθt (z, y)

x ̸= y found by replacing the ground
x=y

truth scores in Equation 3. Our parameterized densities pθt

thus satisfy the following differential equation:

dpθT −t dt

= QθT −tpθT −t

pθT = pbase ≈ pT

(8)

The log likelihood of data points can be bounded using an ELBO based off of Dynkin’s formula (Hanson, 2007), which was derived for discrete diffusion models in Campbell et al.

Proposition 3.3 (Implicit Score Entropy). LSE is equal up (2022). Interestingly, this takes the form of our denoising to a constant independent of θ to the implicit score entropy score entropy loss weighted by the forward diffusion:





LISE = Ex∼p  wxysθ(x)y − wyx log sθ(y)x (6)
y̸=x

Theorem 3.6 (Likelihood Training and Evaluation). For the diffusion and forward probabilities defined above,
− log pθ0(x0) ≤ LDWDSE(x0) + DKL(pT |0(·|x0) ∥ pbase) (9)

Unfortunately, a Monte Carlo estimate would require sam- where LDWDSE(x0) is the diffusion weighted denoising pling an x and evaluating sθ(y)x for all other y. For high score entropy for data point x0

3

Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution

T

Ext ∼pt|0 (·|x0 )

Qt(xt, y) sθ(xt, t)y−

0

y̸=xt

pt|0(y|x0) pt|0(xt|x0)

log

sθ (xt ,

t)y

+

K

pt|0(y|x0) pt|0(xt|x0)

dt (10)

Crucially, this result allows us to directly models based on their likelihood values (and the related perplexity scores), the core metric for language modeling tasks. In particular, we can train and evaluate an upper bound.
Remark. The DWDSE (and the implicit version) can be derived from the general framework of Benton et al. (2022) assuming a concrete score parameterization. In particular, the implicit version coincides with the likelihood loss introduced in Campbell et al. (2022).

3.3. Practical Implementation
Fifth, score entropy can be scaled to high dimensional tasks.
In practice, our state factorizes into sequences X = {1, . . . , n}d to form sequences x = x1 . . . xd (e.g. sequences of tokens or image pixel values). As a general Qt would be of exponential size, we instead choose a sparse structured matrix that perturbs tokens independently with a matrix Qttok. In particular, the nonzero entries of Qt are given by

Qt(x1 . . . xi . . . xd, x1 . . . xi . . . xd) = Qttok(xi, xi) (11)

Since LDWDSE weights the loss by Qt(x, y), this token level transition Qt renders most ratios irrelevant. In particular, we only need to model all ratios between sequences with
Hamming distnace 1, so we can build our score network sθ(·, t) : {1, . . . , n}d → Rd×n as a seq-to-seq map:

(sθ (x1

. . . xi

. . . xd, t))i,xi

≈

pt(x1 . . . xi . . . xd) pt(x1 . . . xi . . . xd)

(12)

To fully compute LDWDSE, we just need to calculate the forward transition pst|e0q(·|·). Luckily, this decomposes as each token is perturbed independently:

d

pst|e0q(x|x) = ptt|o0k(xi|xi)

(13)

i=1

For each ptt|o0k(·|·), we employ the previously discussed strat-

egy and set Qttok = σ(t)Qtok for a noise level σ and a fixed transition Qtok. This avoids numerical integration as, if we

define σ(t) as the cumulative noise

t 0

σ(s)ds,

we

have:

ptt|o0k(·|x) = x-th column of exp σ(t)Qtok

(14)

There are some practical consequences that render most Qtok unusable for large scale experiments (e.g. for GPT-2 tasks, n = 50257). In particular, one is not able to store all edge weights Qtok(i, j) since this takes around 20 GB of GPU memory and is extremely slow to access. Furthermore, one must be able to compute the columns exp(σ(t) · Qtok) to get the transition ratios, but this must avoid matrix-matrix multiplication again can’t be stored in memory.
To sidestep these issues, we follow prior work (Austin et al., 2021; Campbell et al., 2022) and use two standard matrices with special structures. They arise, respectively, from considering a fully connected graph structure and from introducing a MASK absorbing state (similar to the BERT language modeling paradigm (Devlin et al., 2019)):

1 − N 1 · · · 1 

1

Quniform =   

...

1−N ···

...

...

1

...

  

(15)

1

1 ··· 1−N

−1 0 · · · 0 0

 0 −1 · · · 0 0

 Qabsorb = 


...

...

...

...

...

  

(16)

 

0

0 · · · −1 0

1 1 ··· 1 0

With such a structured Q, one can quickly and cheaply compute all values in LDWDSE. As such, our training iteration is about as fast and uses a similar amount of memory as standard autoregressive training. In particular, our training algorithm is given in Algorithm 1.

4. Simulating Reverse Diffusion with Concrete Scores
Given our scores sθ, we now derive various strategies for simulating a path xt = x1t x2t . . . xdt ∼ pt of the reverse diffusion process. Notably, the additional information that we gain from sθ being an approximate ratio of pt can be used to enhance the sampling process.
4.1. Time-Reversal Strategies
To simulate the diffusion in Definition 3.5, one may be tempted to use the Euler strategy from Equation 2. However, as noted in Campbell et al. (2022), this is inefficient because the structure of Qsteq only allows one position to be modified per step.

