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
Do my protein-ensemble-generation methods (AlphaFlow, P2DFlow, Distributional Graphormer, BioEmu, ESMAdam, force-guided SE(3) diffusion) actually claim to recover the true Boltzmann/equilibrium distribution, or just generate diverse conformations? Where do they disagree?

=== RELEVANT PAPERS (ground truth) ===
- AlphaFold Meets Flow Matching for Generating Protein Ensembles (Jing, Bowen; Berger, Bonnie; Jaakkola, Tommi)
- P2DFlow: A Protein Ensemble Generative Model with SE(3) Flow Matching (Jin, Yaowei; Huang, Qi; Song, Ziyang; Zheng, Mingyue; Teng, )
- Ab initio characterization of protein molecular dynamics with AI2BMD (Wang, Tong; He, Xinheng; Li, Mingyu; Li, Yatao; Bi, Ran; Wan)
- Structure Language Models for Protein Conformation Generation (Lu, Jiarui; Chen, Xiaoyin; Lu, Stephen Zhewen; Shi, Chence; )
- Protein Conformation Generation via Force-Guided SE(3) Diffusion Models (Gu, Quanquan; Wang, Yan; Wang, Lihao; Shen, Yuning; Wang, Yi)
- ESMAdam: a plug-and-play all-purpose protein ensemble generator (Yu, Zongxin; Liu, Yikai; Lin, Guang; Jiang, Wen; Chen, Ming)
- Predicting equilibrium distributions for molecular systems with deep learning (Zhang, He; Liu, Haiguang; Liu, Tie-Yan; Shi, Yu; Zheng, Shux)
- AFsample2: Predicting multiple conformations and ensembles with AlphaFold2 (Wallner, Björn; Kalakoti, Yogesh)
- Scalable emulation of protein equilibrium ensembles with generative deep learning (Yim, Jason; Campbell, Andrew; Lewis, Sarah; Hempel, Tim; Lun)

=== ANSWERS TO JUDGE ===
===== Option 1 =====
The six methods fall into a spectrum of how strongly they claim to recover the true Boltzmann/equilibrium distribution versus merely producing diverse conformations. They genuinely disagree, both in stated ambition and in how they justify it.

**Explicit Boltzmann-targeting claims.**

- *Distributional Graphormer (DiG)* is the most direct: it "aims to predict the equilibrium distribution of molecular systems," transforming a simple distribution into the system's equilibrium distribution via a Graphormer diffusion process ("Predicting equilibrium distributions for molecular systems with deep learning"). It positions itself against Boltzmann generators (which sample equilibrium but don't generalize across molecules) and against AlphaFold (single high-probability structure). Validation is distributional — ~70% of MD-sampled SARS-CoV-2 RBD conformations covered by 10,000 DiG structures.
- *BioEmu* makes the strongest quantitative equilibrium claim, asserting it aims to be "the first to quantitatively match equilibrium ensembles and predict experimental observables," explicitly contrasting prior ML systems that only "qualitatively sample distinct conformational states" ("Scalable emulation of protein equilibrium ensembles with generative deep learning"). It backs this with relative free-energy errors around 1 kcal/mol against millisecond MD.
- *AlphaFlow/ESMFLOW* claims Boltzmann-targeting conditionally: only "when trained on MD ensembles" does it belong to the "new generation of Boltzmann-targeting generative models," explicitly trading the exact likelihoods of normalizing-flow Boltzmann generators for scalability ("AlphaFold Meets Flow Matching for Generating Protein Ensembles"). It reports faster wall-clock convergence to equilibrium properties than MD. So its Boltzmann claim is contingent on training data, not intrinsic.
- *Force-guided SE(3) diffusion (CONFDIFF)* claims to align conformations "more truthful to the Boltzmann distribution" via energy/force guidance, but frames this as a correction: existing score-based diffusion "cannot properly incorporate physical prior knowledge... causing large deviations" from equilibrium ("Protein Conformation Generation via Force-Guided SE(3) Diffusion Models"). Notably it does not rely on MD data during training (per its edge to DiG), so its equilibrium fidelity comes from force guidance rather than fitting MD ensembles.

**Thermodynamics-via-proxy, weaker claims.**

- *ESMAdam* reports reproducing "equilibrium free energy surfaces matching ground-truth MD" and "thermodynamic properties," but only under explicit guidance (radius of gyration, secondary-structure distributions). Its core assumption is that ensembles lie "latently embedded near the native structures" in ESMFold embedding space ("ESMAdam: a plug-and-play all-purpose protein ensemble generator") — i.e., it optimizes embeddings to match a supplied constraint rather than learning the Boltzmann measure itself. Equilibrium recovery is therefore conditional on the guidance constraint.
- *P2DFlow* is the most equivocal. It does not claim to recover the Boltzmann distribution; instead it injects an "approximate energy" dimension derived via the Boltzmann equation from MD density to *condition* generation toward distinct-energy states, explicitly to *avoid* generating "non-existent intermediate states" ("P2DFlow: A Protein Ensemble Generative Model with SE(3) Flow Matching"). This is a substantive disagreement with AlphaFlow, which P2DFlow criticizes for generating spurious intermediates between energy minima.

**Disagreements.** (1) Whether MD training data is required: BioEmu/AlphaFlow/DiG/P2DFlow lean on MD; CONFDIFF deliberately does not. (2) Whether intermediate states are signal or artifact: AlphaFlow's intermediates are, per P2DFlow, "non-existent" and to be suppressed. (3) Honesty of scope: BioEmu and DiG claim genuine distributional matching; AlphaFlow qualifies its claim by training regime; ESMAdam and P2DFlow recover thermodynamics only via explicit energy/observable guidance rather than as an emergent property.

The context does not include any explicit Boltzmann claim from the structure-language-model or AFsample2 lines for the six named methods, so no further attribution is possible.

===== Option 2 =====
Your six methods do not make a uniform claim. They fall on a spectrum from "generate diverse conformations" to "approximate the true Boltzmann/equilibrium distribution," and they disagree explicitly about whether the others succeed.

**Methods that explicitly target the Boltzmann/equilibrium distribution.** Distributional Graphormer (DiG) makes the strongest framing: its stated goal is to "predict the equilibrium distribution of molecular systems," providing not just diverse conformations but state-density estimates and thermodynamic properties, validated by treating millisecond MD as a proxy for equilibrium ("Predicting equilibrium distributions for molecular systems with deep learning"). To do this it adds physics-informed diffusion pre-training (PIDP) that supervises the model with energy functions/force fields where data are scarce. BioEmu shares this ambition and is the most quantitative: it claims to "approximately sample from the equilibrium distribution," reporting relative free-energy errors around 1 kcal/mol validated against millisecond MD and experimental stabilities ("Scalable emulation of protein equilibrium ensembles with generative deep learning"). Notably, BioEmu uses an architecture similar to DiG but argues that qualitative diversity is insufficient and that "a demonstration that generative ML can quantitatively match equilibrium ensembles... is critical" — an implicit critique that prior diversity-focused work, including DiG-style sampling, had not delivered quantitative equilibrium fidelity. Force-guided SE(3) diffusion (ConfDiff) also explicitly targets the Boltzmann distribution, using an MD-energy reward and an intermediate force-guidance network to "reweight the generated conformations to ensure they adhere better to the equilibrium distribution" ("Protein Conformation Generation via Force-Guided SE(3) Diffusion Models").

**Methods that target diversity/fidelity to MD ensembles without strong equilibrium claims.** AlphaFlow is positioned as providing a "superior combination of precision and diversity"; it claims faster wall-clock convergence to "certain equilibrium properties" than MD and situates itself within "Boltzmann-targeting generative models" only when trained on MD, but does not claim to recover the true Boltzmann distribution ("AlphaFold Meets Flow Matching"). P2DFlow aims to capture "observable dynamic fluctuations" and "more accurately recover protein ensemble distributions," using an 'approximate energy' dimension derived from the Boltzmann equation, but frames success in terms of fidelity/dynamics metrics against MD ground truth rather than equilibrium recovery ("P2DFlow"). ESMAdam is the weakest claimant: it notes that generating Boltzmann-distributed ensembles is "one fundamental objective" but its own method is constraint-driven optimization in ESMFold's embedding space toward user-defined targets (radius of gyration, secondary structure), making no claim of unbiased equilibrium sampling ("ESMAdam").

**Where they disagree.** The sharpest disagreement is over MSA-perturbation and energy-free diffusion methods. ConfDiff argues such heuristics "cannot guarantee the predicted structure to be a low energy state" and questions whether sequence-/energy-free models like Str2Str produce outputs that "adhere to the Boltzmann distribution." P2DFlow criticizes AlphaFlow for failing to generate non-existent intermediate states between minima. ConfDiff and BioEmu also differ from DiG on data: ConfDiff deliberately avoids MD training data (unlike DiG), while DiG and BioEmu rely heavily on MD/energy supervision to justify equilibrium claims.

**Insufficiency flag.** The provided context contains no AFsample2 material in the question's six but does include it as background; AFsample2 explicitly only claims to "widen the conformational landscape" and improve alternate states, not recover equilibrium. The context does not give head-to-head quantitative free-energy comparisons across all six, so relative equilibrium accuracy cannot be ranked from these excerpts.

===== Option 3 =====
Your methods split into two camps on the central question, and the split is explicit in how each paper frames its own goal.

**Camp 1 — explicitly target the Boltzmann/equilibrium distribution.**

- *BioEmu* ("Scalable emulation of protein equilibrium ensembles with generative deep learning") makes the strongest equilibrium claim and backs it quantitatively: it states its ensembles "represent equilibrium in a range of challenging and practically relevant metrics," reporting relative free-energy errors around 1 kcal/mol validated against millisecond-timescale MD and experimentally measured protein stabilities. It explicitly aims to recover conformational *and binding state probabilities*, not just structures.
- *Force-guided SE(3) diffusion* ("Protein Conformation Generation via Force-Guided SE(3) Diffusion Models," ConfDiff) is built precisely on the premise that recovering the Boltzmann distribution is the goal and that prior diffusion methods fail at it: it argues "existing score-based diffusion methods... caus[e] large deviations in the sampled protein conformations from the equilibrium distribution," and adds MD-force guidance so conformations "better adhere to the Boltzmann distribution." So its claim is aspirational-but-targeted: better fidelity to Boltzmann, not a proof of recovery.

**Camp 2 — target diverse / accurate ensembles, with equilibrium as a partial or downstream property.**

- *AlphaFlow* ("AlphaFold Meets Flow Matching for Generating Protein Ensembles") is deliberately hedged. It claims a superior "precision and diversity" combination and, when trained on MD, "accurately captures conformational flexibility, positional distributions, and higher-order ensemble observables." Its equilibrium claim is explicitly partial: it converges faster than replicate MD "to *certain* equilibrium properties," positioning itself as a proxy rather than a true Boltzmann sampler.
- *P2DFlow* ("P2DFlow: A Protein Ensemble Generative Model with SE(3) Flow Matching") frames its goal as predicting structural ensembles that capture "observable dynamic fluctuations" matching crystal/MD data. It invokes physical laws via a designed prior "which can reflect the physical laws governing the distribution of ensembles," but its stated validation is fluctuation/observable matching on ATLAS MD, not free-energy or Boltzmann recovery. It is a "potential proxy" for MD.
- *ESMAdam* ("ESMAdam: a plug-and-play all-purpose protein ensemble generator") is the most pluralistic: it names Boltzmann-distributed generation as only *one* of many tasks ("One fundamental objective is to generate... ensembles that follow the Boltzmann distribution"), alongside constraint-, ligand-, topology-, backmapping-, and cryo-EM-driven generation. As a plug-and-play optimizer over ESMFold embeddings, it makes no claim to recover equilibrium by default; equilibrium is just one possible restraint condition.

**Where they disagree.**

The sharpest disagreement is about whether learned generative models recover equilibrium *at all*. ConfDiff explicitly accuses score-based diffusion of large deviations from equilibrium and treats added force guidance as necessary; BioEmu instead claims near-equilibrium accuracy (~1 kcal/mol) from training alone (MD + structures + stabilities) without inference-time force guidance — a methodological disagreement about whether physics must be injected at sampling time or can be learned from data. A second contested point, raised by P2DFlow against AlphaFlow, is interpolation: P2DFlow notes AlphaFlow "encounters challenges in generating non-existent intermediate states between two or more minima," i.e. it may reproduce known metastable basins (diversity) without correctly populating transition regions (true distribution). P2DFlow and ConfDiff also fault STR2STR-style diffusion for over-diverse, low-energy-unfocused output — diversity without distributional correctness.

**Caveats / insufficiency.** No paper titled "Distributional Graphormer" is present in the provided context, so I cannot characterize its claim. The context also includes a Structure Language Model / ESMDiff paper not in your list, which notes that 3D-diffusion methods "center around the vicinity of metastable states" — reinforcing the diversity-vs-equilibrium tension but not directly answering for your six named methods.

Bottom line: only BioEmu (with evidence) and ConfDiff (by design intent) claim equilibrium recovery; AlphaFlow, P2DFlow, and ESMAdam primarily claim accurate/diverse ensembles with equilibrium as partial, observable-level, or optional.

===== Option 4 =====
These six methods do not make uniform claims. They split into three camps along a spectrum from "diverse-but-not-distributionally-calibrated conformations" to "explicit recovery of the Boltzmann/equilibrium distribution," and the explicitly Boltzmann-targeting methods openly criticize the others for failing on exactly this point.

**Camp 1 — Diversity / distributional fidelity to MD, but no claim of true Boltzmann recovery.**
*AlphaFold Meets Flow Matching for Generating Protein Ensembles* (AlphaFlow) frames its goal as learning and sampling "conformational landscapes," and its claims are empirical rather than thermodynamic: a superior precision–diversity Pareto frontier versus MSA subsampling, accurate capture of conformational flexibility, positional distributions, and higher-order observables, and faster wall-clock convergence to "certain equilibrium properties" than replicate MD — positioning it as a proxy for MD, not a Boltzmann sampler [AlphaFold Meets Flow Matching for Generating Protein Ensembles]. Notably, that same paper draws an explicit distinction between its own line of work and "a related but separate line of work… learning generative models of Boltzmann distributions," placing Distributional Graphormer (Zheng et al.) and EigenFold in the structure-generating camp rather than the Boltzmann-sampling camp, and remarking that such ensemble models "have yet to show convincing validations" against MSA baselines [AlphaFold Meets Flow Matching for Generating Protein Ensembles]. P2DFlow is similar in spirit: it targets recovery of MD/observable distributions on ATLAS and "more accurately recovers protein ensembles distributions" than AlphaFlow and STR2STR, using an "approximate energy" (a Gaussian-KDE-over-RG/RMSD pseudo-energy, not a force field) to bias structures [P2DFlow: A Protein Ensemble Generative Model with SE(3) Flow Matching]. P2DFlow concedes that learning the *correct* distribution would require "advanced force fields" it does not yet incorporate — implicitly admitting it does not recover the true Boltzmann distribution [P2DFlow: A Protein Ensemble Generative Model with SE(3) Flow Matching].

**Camp 2 — Explicit Boltzmann/equilibrium targeting via physics.**
The force-guided SE(3) diffusion model (ConfDiff) is the most explicit: it states plainly that prior score-based diffusion methods "cannot properly incorporate important physical prior knowledge," causing "large deviations… from the equilibrium distribution," and its stated aim is to sample from p₀(x₀) ∝ q₀(x₀)e^(−kE₀(x₀)) using the OpenMM energy as guidance, with formal propositions for the guided marginals [Protein Conformation Generation via Force-Guided SE(3) Diffusion Models]. It explicitly criticizes the diversity-only methods — noting that training data "does not comply with the equilibrium distribution" and questioning "whether the diverse generated outcomes adhere to the Boltzmann distribution" [Protein Conformation Generation via Force-Guided SE(3) Diffusion Models]. BioEmu makes the strongest *quantitative* equilibrium claim: relative free-energy errors "around 1 kcal/mol" validated against millisecond MD and experimental stabilities, emulating both structural ensembles and thermodynamic properties [Scalable emulation of protein equilibrium ensembles with generative deep learning].

**Camp 3 — Boltzmann as one task among many (plug-and-play).**
ESMAdam frames Boltzmann-following ensembles as just "one fundamental objective" alongside constraint-satisfaction tasks; it claims to generate ensembles that "align with the Boltzmann distribution" benchmarked against long MD, but hedges that distributions may be effectively "few-mode" at room temperature [ESMAdam: a plug-and-play all-purpose protein ensemble generator].

**Where they disagree.** The sharpest disagreement is methodological: ConfDiff and BioEmu argue true equilibrium recovery requires explicit physics/free-energy validation, and ConfDiff directly impugns sequence-conditioned diffusion models for deviating from Boltzmann [Protein Conformation Generation via Force-Guided SE(3) Diffusion Models]; AlphaFlow and P2DFlow accept distributional/observable fidelity to MD as their bar without claiming exact Boltzmann recovery.

**Insufficiency.** The context contains no chunk directly from Distributional Graphormer itself; its claim is characterized only secondhand by AlphaFlow and via citations [AlphaFold Meets Flow Matching for Generating Protein Ensembles; Scalable emulation of protein equilibrium ensembles with generative deep learning], so its self-stated goal cannot be fully verified here.
