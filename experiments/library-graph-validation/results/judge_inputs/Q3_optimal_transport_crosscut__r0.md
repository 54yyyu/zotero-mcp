You are a meticulous expert reviewer comparing four answers to the SAME research
question, each generated from a different (undisclosed) evidence-retrieval method over
the same personal paper library. Judge ONLY answer quality. Do not try to guess the method.

Score each answer 1-10 on each dimension:
- coverage: how many of the genuinely relevant papers/points are represented.
- synthesis: quality of CROSS-PAPER structure (consensus, disagreement, lineage/subsumption,
  taxonomy) -- NOT just a list of per-paper summaries.
- correctness: are the relational and technical claims ACCURATE? Heavily penalize hallucinated
  or wrong subsumption/contradiction claims (e.g. "X generalizes Y" when false).
- faithfulness: claims attributed to specific papers; no unsupported assertions; no outside facts.
- directness: answers the actual question precisely without padding (length is NOT a virtue).

Then give an overall ranking of the four (best to worst). Ties are NOT allowed in the ranking.

The relevant papers for this question (ground truth, for assessing coverage) are listed below.
Be skeptical: a confident, well-written answer can still be wrong or shallow.

Output STRICT JSON only, no prose:
{"scores": {"Option 1": {"coverage":N,"synthesis":N,"correctness":N,"faithfulness":N,"directness":N},
            "Option 2": {...}, "Option 3": {...}, "Option 4": {...}},
 "ranking": ["Option X","Option Y","Option Z","Option W"],
 "notes": "1-3 sentences on the key differentiators"}


=== QUESTION ===
Where in my library is optimal transport used, and for what distinct purposes -- generative modeling vs data alignment vs trajectory inference vs Schrodinger bridges?

=== RELEVANT PAPERS (ground truth) ===
- Diffusion Schrödinger Bridge Matching (Campbell, Andrew; Shi, Yuyang; Bortoli, Valentin De; Doucet,)
- Optimal transport mapping via input convex neural networks (Makkuva, Ashok Vardhan; Taghvaei, Amirhossein; Oh, Sewoong; )
- SCOT: Single-Cell Multi-Omics Alignment with Optimal Transport (Demetci, Pinar; Santorella, Rebecca; Sandstede, Björn; Noble)
- Learning single-cell perturbation responses using neural optimal transport (Bunne, Charlotte; Stark, Stefan G.; Gut, Gabriele; del Casti)
- Wasserstein Flow Matching: Generative modeling over families of distributions (Haviv, Doron; Pe'er, Dana; Pooladian, Aram-Alexandre; Amos, )
- Stochastic Interpolants: A Unifying Framework for Flows and Diffusions (Albergo, Michael S.; Boffi, Nicholas M.; Vanden-Eijnden, Eri)
- Gene trajectory inference for single-cell data by optimal transport metrics (Qu, Rihao; Cheng, Xiuyuan; Sefik, Esen; Stanley III, Jay S.;)
- A unified computational framework for single-cell data integration with optimal transport (Cao, Kai; Gong, Qiyu; Hong, Yiguang; Wan, Lin)
- NovoSpaRc: flexible spatial reconstruction of single-cell gene expression with optimal transport (Friedman, Nir; Moriel, Noa; Senel, Enes; Rajewsky, Nikolaus;)

=== ANSWERS TO JUDGE ===
===== Option 1 =====
Optimal transport (OT) appears across your library in two broad camps -- machine-learning methodology and single-cell/spatial biology -- and within them serves four distinct purposes. The unifying mathematical core is the Monge/Kantorovich problem and its entropic regularization via Sinkhorn, which several papers cite as shared scaffolding ("Optimal transport mapping via input convex neural networks"; "SCOT"; "A unified computational framework for single-cell data integration with optimal transport").

Generative modeling. This is the densest cluster. "Optimal transport mapping via input convex neural networks" learns the Brenier OT map as the gradient of a convex function (ICNN) via minimax optimization, and frames it explicitly as a generative model from a latent to a target distribution. "Wasserstein Flow Matching" lifts flow matching onto families of distributions, treating each sample as itself a measure and using (entropic) OT maps as geodesics in Wasserstein space to generate Gaussians and point-clouds. "Stochastic Interpolants" builds flow/diffusion generative models bridging arbitrary densities; OT enters at the limit (Benamou-Brenier) and as a special optimized case. "Diffusion Schrödinger Bridge Matching" situates generative modeling (DDMs, flow/bridge matching) as transport problems that are *not* generally close to the OT map -- a deliberate contrast.

Schrödinger bridges. Two papers treat the SB as dynamic, entropy-regularized OT. "Diffusion Schrödinger Bridge Matching" is the dedicated treatment: it introduces Iterative Markovian Fitting to solve SBs, recovering classical OT as the zero-noise limit and the Benamou-Brenier formula. "Stochastic Interpolants" independently shows its interpolants recover the Schrödinger bridge when one optimizes over the interpolant. These two converge on the same object (SB = EOT in path space) from different algorithmic routes; this is a lineage/subsumption relationship -- DSBM frames diffusion/flow/bridge matching as special or limiting cases of its IMF machinery (Figure 1, Table 1), and stochastic interpolants likewise position themselves as a unifying framework subsuming score-based diffusion and rectified flows.

Data alignment / integration. In biology, OT is used to align unpaired or cross-modal data. "SCOT" uses Gromov-Wasserstein OT to align single-cell multi-omics across different metric spaces without correspondences, preserving local geometry via k-NN graph distances. "A unified computational framework for single-cell data integration with optimal transport" (uniPort) combines a coupled VAE with minibatch *unbalanced* OT to integrate multi-omics and spatial data and to transfer labels via the OT plan. "NovoSpaRc" applies OT (Gromov-Wasserstein-style, with a structural-correspondence prior) to probabilistically map cells to spatial locations -- spatial reconstruction, a form of cross-space alignment. "Learning single-cell perturbation responses using neural optimal transport" (CellOT) aligns unpaired control and perturbed cell populations using neural OT, and notably reuses the ICNN dual-potential construction from "Optimal transport mapping via input convex neural networks" -- a direct methodological lineage between the ML and biology camps.

Trajectory inference. "Gene trajectory inference for single-cell data by optimal transport metrics" (GeneTrajectory) uses graph-based Wasserstein distances between *gene* distributions over a cell graph to order genes, distinguishing itself from cell-trajectory methods. "Learning single-cell perturbation responses using neural optimal transport" also claims trajectory uses (hematopoietic developmental trajectories, reprogramming trajectories), so it straddles alignment and trajectory inference.

A genuine tension: the generative/SB papers prize the OT map for its optimality properties, whereas "Diffusion Schrödinger Bridge Matching" stresses that popular generative transports (DDMs, flow matching) do *not* recover the OT map -- motivating SBs. The biology papers, by contrast, treat OT primarily as an alignment/coupling tool rather than a generator. The context does not include head-to-head benchmarks across these purposes, so relative performance claims cannot be adjudicated here.

===== Option 2 =====
Optimal transport (OT) appears across your library in two broad clusters -- a machine-learning / generative-modeling cluster and a single-cell biology cluster -- and within them serves four distinguishable purposes.

**1. Generative modeling (and the closely tied domain-transfer task).** This is the dominant use. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow* frames both generative modeling and domain transfer as one "transport mapping problem" -- finding a map T pushing π0 to π1 -- and explicitly proposes OT as the unifying lens, seeking couplings that minimize a transport cost E[c(Z1−Z0)] (typically quadratic), while noting existing OT solvers are too slow for high-dimensional data (B7, B15, B27). *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions* uses OT chiefly to *regularize* flow-based generation -- via path-length penalties or parameterization structure -- and discusses unbiased vs. biased (rectification-based) routes to lowering transport cost (B1, B8). *Diffusion Schrödinger Bridge Matching* (DSBM) motivates its method by the observation that diffusion and flow-matching models do not generally yield transports close to the deterministic OT map, which is desirable for its Wasserstein-2 properties (B3, B20, B26, B31). *Wasserstein Flow Matching* is catalogued here too, performing generative modeling over families of distributions using OT/Wasserstein tools (B18).

**2. Data alignment / coupling across domains.** This is the distinctive biology use. *SCOT: Single-Cell Multi-Omics Alignment with Optimal Transport* uses (entropy-regularized, Gromov–Wasserstein) OT to find a coupling matrix aligning cells across two omics domains with no assumed 1-1 correspondence, relying only on shared underlying biology (B2, B6, B12). *Learning single-cell perturbation responses using neural optimal transport* uses OT's "dual roles" -- as a distance between distributions and as a geometry-based coupling between them -- to map cells from unperturbed to perturbed states (B13).

**3. Trajectory / dynamics inference.** OT is used to model temporal evolution of cell populations. SCOT cites Schiebinger et al. (2019), who applied regularized unbalanced OT to study temporal gene-expression change between time points (B2). The Stochastic Interpolants references include TrajectoryNet, "a dynamic optimal transport network for modeling cellular dynamics," and Bunne et al.'s proximal-OT modeling of population dynamics is cited in DSBM's references (B21, B5). Note: these trajectory-inference uses appear mainly as cited prior work rather than as papers fully present in the context, so the evidence here is thinner than for the other categories.

**4. Schrödinger bridges (entropy-regularized, stochastic OT).** Multiple papers treat the SB problem as the diffusive/stochastic counterpart of deterministic OT -- formally, an entropy-regularized or dynamic version of OT (B1, B20, B31). DSBM is the central work, introducing Iterative Markovian Fitting and recovering OT-CFM, Rectified Flow, and flow matching as special/limiting cases (B17, B22, B30). Stochastic Interpolants also offers an SB solution via the Benamou–Brenier formulation (B8). Rectified Flow positions SB as an alternative but computationally challenging route to diffusion generative models (B25, B35).

**Cross-cutting structure.** The ML papers largely agree that generative modeling, domain transfer, and SBs are facets of one transport-mapping problem, with OT supplying the "ideal" map; DSBM and Stochastic Interpolants explicitly unify SB and OT (B17, B20, B30, B8). A genuine methodological tension is deterministic ODE vs. stochastic SDE transport: Rectified Flow argues diffusion noise is unnecessary and straight (deterministic) paths suffice (B34, B35), whereas DSBM insists the SDE approach is theoretically crucial for the SB characterization and empirically better (B17, B30). The biology papers, by contrast, use OT not to generate but to *align* and to *infer dynamics*.

===== Option 3 =====
Optimal transport (OT) recurs across your library in four functionally distinct roles. They share a mathematical core (Kantorovich potentials, Wasserstein geometry, entropic regularization) but pursue separate goals, and the papers occasionally disagree about whether OT optimality is even desirable.

**1. Generative modeling.** Here OT mostly serves as a *geometry for transport paths* rather than as an end in itself. "Flow Matching for Generative Modeling" uses OT displacement interpolation to define conditional probability paths with straight-line trajectories, arguing these beat curved diffusion paths on training/generation speed and sample quality. "Wasserstein Flow Matching" goes further, making distributions themselves the samples: it instantiates Riemannian flow matching with target velocities equal to OT maps (geodesics in Wasserstein space via Benamou-Brenier/McCann interpolation), using entropic OT to approximate maps between point clouds. "Stochastic Interpolants" and "Step-by-Step Diffusion" situate these flow methods within a unified flow/diffusion framework, and "Mean Flows for One-step Generative Modeling" notes that marginal velocity fields induce curved trajectories *even when conditional flows are rectified*. A notable internal disagreement: "Flow Straight and Fast" (Rectified Flow) explicitly does *not* target a c-optimal coupling (except in 1D), positioning recursive rectification as a transport-cost-reducing alternative rather than true OT. This contrasts with the OT-mapping papers below, which insist on recovering the *actual* optimal map.

**2. Data alignment.** "SCOT" uses Gromov-Wasserstein OT to align single-cell multi-omics datasets, exploiting that GW compares intra-domain pairwise distances rather than points directly -- enabling alignment when features across modalities are incomparable. It needs neither sample- nor feature-wise correspondences and uses GW distance as a self-tuning heuristic. This is OT as an unsupervised correspondence-finding tool, distinct from generation.

**3. Trajectory inference / perturbation response.** "Learning single-cell perturbation responses using neural optimal transport" (CellOT) recovers an OT map between unpaired control and perturbed cell distributions (one map per perturbation), capturing higher moments of the perturbed population rather than only the mean (as autoencoder baselines do). This is OT used to infer the mapping between biological states. Note: the context describes CellOT in terms of perturbation response, not explicit time-series trajectory inference; I see no paper here doing developmental trajectory inference per se, so that sub-purpose is only partially covered.

**4. Schrodinger bridges.** "Diffusion Schrodinger Bridge Matching" (DSBM) is the most explicit OT-theoretic paper: it argues DDMs and Flow Matching are *not* guaranteed to produce transports close to the OT map, whereas Schrodinger bridges compute stochastic dynamic mappings recovering entropy-regularized OT. DSBM unifies the generative-modeling thread, recovering Flow Matching (as sigma->0), Bridge Matching, and DDMs as special/limiting cases.

**Cross-cutting structure.** A foundational methodological thread links categories 1, 3, and 4: "Optimal transport mapping via input convex neural networks" learns the true OT map as the gradient of an ICNN-parametrized convex potential, criticizing WGANs for using OT only as a distance and learning arbitrary, initialization-sensitive generators. CellOT directly builds on this ICNN-dual-potential idea, and SCOT/WFM connect via the shared OT machinery. The deepest tension in your library is precisely *OT-as-distance vs. OT-as-map-recovery*: WGAN-style and Rectified-Flow approaches treat optimality loosely, while ICNN-based mapping, CellOT, and Schrodinger-bridge methods treat recovering the genuine (or entropy-regularized) optimal map as essential.

===== Option 4 =====
Optimal transport (OT) appears across your library as a shared mathematical backbone, but the papers deploy it for several distinct purposes. Below is a taxonomy by purpose, noting where methods overlap, subsume one another, or differ in their treatment of OT.

## 1. Generative modeling (the dominant use)
Most of the OT in your library is in service of generative modeling, where the goal is to transport a reference/noise distribution to a data distribution.

- **"Flow Matching for Generative Modeling"** (Lipman et al.) uses OT instrumentally: it shows that Continuous Normalizing Flows can be trained simulation-free by regressing onto conditional probability paths, and that using *OT displacement interpolation* (McCann) to define those paths yields straight-line trajectories that train and sample faster and generalize better than curved diffusion paths. Here OT is a design choice for the conditional path, not the object being solved.
- **"Flow Straight and Fast: ... Rectified Flow"** (Liu et al.) frames generative modeling and domain transfer jointly as a transport-mapping problem between two empirical distributions, and learns straight-path ODEs. Critically, it proves that "rectification" yields couplings with provably *non-increasing convex transport costs* — i.e., it moves toward (but does not claim to exactly recover) the OT coupling. This is a partial, iterative approach to optimality rather than exact OT.
- **"Optimal transport mapping via input convex neural networks"** (Makkuva et al.) is the most directly OT-native generative approach: it learns the exact Monge/Kantorovich OT map (under quadratic cost / 2-Wasserstein) as the gradient of an input-convex neural network via minimax optimization, and then repurposes that map as a deep generative model. Its claimed advantages over GAN-style training — initialization-independence and the ability to represent discontinuous maps — stem precisely from seeking the *true* OT map rather than an arbitrary transport.
- **"Wasserstein Flow Matching"** (Haviv et al.) lifts flow matching to *families of distributions*, using (entropic) OT and Wasserstein geometry so that each data sample is itself a distribution (Gaussians or point-clouds). OT here defines the geometry (geodesics in Wasserstein space) along which generation occurs.

## 2. Schrödinger bridges (stochastic, entropy-regularized OT)
- **"Diffusion Schrödinger Bridge Matching"** (Shi, De Bortoli, Campbell, Doucet) is the clearest exemplar of the SB purpose. It explicitly distinguishes its target — Schrödinger bridges recover *entropy-regularized, dynamic* OT — from diffusion/flow matching, which it argues are **not** guaranteed to be close to the OT map. Its Iterative Markovian Fitting (IMF)/DSBM algorithm subsumes several recent transport methods as special/limiting cases, positioning SB as a more principled (if numerically harder) route to OT than the generative-modeling methods above.

This is a genuine point of contrast in your library: A1 (DSBM) frames the flow/diffusion family (A4, A5) as approximate or non-OT, whereas those papers treat OT as a useful but optional ingredient. A2 (ICNN) instead pursues exact deterministic OT directly.

## 3. Data alignment / domain transfer
This appears as a secondary, shared purpose rather than a dedicated paper. Both **Rectified Flow** (image-to-image translation, domain adaptation) and the **ICNN** paper (which cites domain adaptation, color/shape transfer, data assimilation as OT applications) treat alignment between two empirically observed distributions as the same transport problem used for generation.

## 4. Trajectory inference — not covered
None of the provided papers addresses trajectory inference (e.g., single-cell developmental trajectories via OT). Although A3 and A1 mention single-cell genomics and biology as OT application areas, the context is insufficient to characterize trajectory inference as a distinct purpose in your library.
