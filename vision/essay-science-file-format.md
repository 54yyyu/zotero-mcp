# Science's File Format Problem

*Draft — blog-style essay, Paul Graham register. June 2026.*

---

Here is a strange thing we do. A scientist runs an experiment. The results exist as numbers in a database, code in a repo, structure everywhere. Then she writes a story about them, typesets the story, and flattens it into a PDF. Then thousands of other people — and now millions of AI agents — spend enormous effort trying to extract the numbers and the structure back out of the story. We pay to destroy the structure, and then we pay again, over and over, to rebuild it.

OCR on scientific papers is reverse-engineering our own compression artifact. The information was machine-readable the moment it was created. We made it unreadable on purpose, because the publication format demanded it.

The PDF is a printout. It was a perfectly good format when the reader was a human with a printer. But look at who reads papers now. Increasingly it's machines: agents doing literature review, extraction pipelines, models being trained. Within a few years the majority of paper-reads will be by machines, and every single one of those reads pays an extraction tax — separately, redundantly, lossily. A format optimized for the minority reader.

That's the first problem, and it's really just an efficiency problem. The second problem is worse, because it's an information problem: papers only record wins.

A paper is a highlight reel. It shows the path that worked, cleaned up to look like the author knew it all along. The dead ends — the four approaches that failed before the fifth one worked — are edited out, because the format is a story and stories need to be clean. But those dead ends are often the most valuable information the project produced. They're what would save the next lab six months. In any other learning system we'd call failure data the most important training data. In science we systematically delete it.

The result is that the literature radically understates what humanity has actually tried. Somewhere right now, several labs are running an experiment that several other labs have already run and abandoned. None of them know about the others, because there's no place a failure could have been recorded. We don't have a record of science. We have a record of the parts of science that made good stories.

Why does it work this way? Incentives, of course. The format follows the reward. You get credit for publications, publications need to be narratives of success, so success narratives are what get written. Nobody writes the failure paper because the failure paper counts for nothing. The format problem and the incentive problem are the same problem wearing two coats.

So the fix has two parts, and they have to come in the right order.

The first part is the format. The unit of scientific communication should not be a 20-page story. It should be a structured object: claims, the evidence for each claim, the methods that produced the evidence, and — this is the part nobody ships — the trajectory. Not just "we found X" but "we tried A, B, C; A failed because of this; B was promising but didn't scale; C worked." A graph, not a narrative. Negative branches welcome, because a negative branch is a signpost, and signposts are cheap to store and expensive to rediscover.

This is no longer hypothetical. In the last year, people have shown you can decompose papers into atomic claims at journal scale — nearly two million claims from sixteen thousand papers — and that a machine evaluating those claims agrees with human peer reviewers about as often as reviewers agree with each other, while covering far more of the paper than humans bother to review. Others have shown that agents given structured research artifacts instead of PDFs extract knowledge dramatically more accurately and reproduce results more reliably, with the gap widest exactly where papers omit the most: configurations, failure modes, the tacit stuff. The format works. That's settled.

What's not settled is adoption, and here's where every previous attempt died. Nanopublications have existed since 2009. They're elegant and correct and nobody uses them. Micropublication platforms welcome negative results and stay niche. The pattern is always the same: a new format that asks the author to do extra work, for the benefit of the community, paid out in a currency the author's career doesn't accept. Asking the whole world to switch formats at once is asking for a miracle.

The way around a chicken-and-egg problem is to find someone who benefits even when they're the only user. So: start selfish.

A researcher converts her own library — the few hundred PDFs she already has — into the structured form, because it makes *her* faster this week. Her agent can suddenly answer questions her PDF folder never could: where do my papers disagree, which methods keep failing under which conditions, what has actually been tried in this sub-sub-field. No one else needs to participate. The format pays for itself at n=1.

Then it spreads the way useful things spread. Her lab shares converted libraries, because conversion is expensive and a labmate's conversion is as good as yours — convert once, reuse everywhere. Labs federate into communities. At some point the corpus of converted papers is large enough that the obvious question arises: why are we converting at all? Why not publish in this form directly? That's when new work starts being born structured, trajectory and all, and conversion becomes something you only do to the past.

And only then — on top of a graph that people already live in — does the incentive layer make sense. When knowledge is a graph of claims and trajectories, credit can finally follow contribution instead of narrative. Your negative result is a node; when it saves someone a detour, that's visible, attributable, creditable. Evaluation itself becomes a first-class contribution: assessing claims builds reputation the way good answers build reputation on Stack Overflow, so review stops being unpaid invisible labor. You don't need a token or a chain for this; you need signatures, provenance, and a community that already gets daily value from the substrate. The credit system is the roof, not the foundation. Every project that built the roof first is a ghost town.

The deepest version of this is that science becomes cumulative in a way it currently only pretends to be. Today each paper is an island with citation bridges. In a claim graph, knowledge composes: an agent planning an experiment can traverse everything tried before, including the failures, and start from the actual frontier instead of the advertised one. The frontier of science is further out than the literature says — we just can't see it, because we wrote down the highlights and threw the map away.

PDFs were the right format for a world where humans were the only readers and paper was the medium. That world ended. The format just hasn't noticed yet.
