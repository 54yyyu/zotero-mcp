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
