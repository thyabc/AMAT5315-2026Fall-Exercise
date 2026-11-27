# Independent DeepSeek review

Date: 2026-11-27

The primary assistant opened an actual DeepSeek web conversation and submitted
the assignment implementation plan, then the actual Part 1 generator and MILP
construction snippets. This file summarizes the final replies, excluding
intermediate reasoning. DeepSeek did not run code or experiments.

## Scope and conclusion

The plan review covered objective offsets, XOR variables, exactness, branching
trial state, dataset splits and numerical parity. The Part 1 snippet review
reported no definite error in the displayed objective signs, constant term,
four edge inequalities, continuous bond variables or rounding threshold.
This was a scoped static review, not a review of every source file or a
confirmation of measured benchmark values.

## Review questions and disposition by the primary assistant

1. Match every RNG call and edge ordering against the original generator
   specification. The original Week 12 PDF was read directly; the code follows
   its per-instance seeding, minimum-degree stage, random additions, sorted
   edge couplings and final field draws. DeepSeek was not given the complete
   original generator text, so its reply did not certify this match.
2. Confirm the expected mean LP bound and rounding excess. Both -52 and 34.25
   are stated on page 7 of the supplied sheet. They are assertions only.
   Actual JSON is preserved on mismatch with validation_status=failed_reference;
   the script stops and the plotting script refuses to plot a failed run.
3. Confirm configure() actually sets all options and columns are binary.
   configure() sets presolve=off, threads=1, mip_rel_gap=0, mip_abs_gap=0.
   Only x columns use add_integer_column with bounds [0,1]. The highs 2.4.0
   API was checked against its official Rust documentation.
4. Use tolerance for XOR checks. The code uses absolute error <=1e-6 for
   integrality, XOR and direct physical-energy consistency checks.
5. Ensure exactly m distinct edges and the correct gap direction. BTreeSet
   guarantees uniqueness; construction rejects an excessive initial edge set
   and the fill loop stops at exactly m. Gap is MILP energy minus LP energy.

## What remains to be validated on the user's computer

- Both pred examples, in particular the 100-spin reference energy -138.
- All Part 1 physical-energy and reference-statistic assertions.
- Local measured times and the resulting gap.png.
- Parts 2-4 and all learning/generalization claims, which are not implemented
  or validated by this Part 1 delivery.

<!-- WEEK12_PART2_BEGIN -->
## Part 2 independent review (2026-11-27)

An actual follow-up request was submitted to the same DeepSeek conversation.
The submitted excerpts covered continuous node LPs, the DFS search loop,
incumbent updates, Strong Branching trial isolation, gain direction, child
ordering, solve counts, timing scope and the workshop trace contract.
DeepSeek performed static review only and reported no definite error in the
shown minimization/pruning, incumbent or trial-accounting logic.

Review dispositions by the primary assistant:

1. The 1e-6 pruning/integrality tolerance is the sheet's requirement. The
   provided generator uses integer couplings and fields; no claim of certified
   exactness for arbitrary real input at differences below that tolerance is made.
2. The gain sentinel 1e6 is explicitly required by the sheet and changes only
   variable selection. Trial LPs use cloned bound vectors and independent models;
   an immutable incumbent snapshot is used and trials cannot update it.
3. Integral solutions are rounded only within tolerance and checked by direct
   Hamiltonian recomputation; the final search is independently compared with
   HiGHS. Non-optimal/non-infeasible LP statuses abort the run and cannot be
   reported as a completed successful benchmark.
4. Counts include infeasible visited/trial nodes. No trial solve is cached or
   reused as a visited-node LP. The program checks the total-count identity.
5. DeepSeek's final prose reversed the required workshop PUSH order in its
   last point. The delivered code correctly pushes x2>=3 FIRST and x2<=2 LAST;
   LIFO therefore VISITS x2<=2 first. This was checked by the actual three-node
   unit test and CLI integration trace; that erroneous prose was not adopted.

Validation performed before delivery: Rust compilation passed; both policies
matched exhaustive enumeration on eight small n=3/4/6/8 graphs; the actual
workshop trace visited three nodes; a separate 2-instance, 6-spin CLI/plot
integration check passed with zero energy errors and correct solve accounting.
Those are development checks, not the assignment's formal benchmark results.
The two official 48-instance runs and their reported times remain to be
computed on the user's own computer by run_part2.sh.
<!-- WEEK12_PART2_END -->

<!-- WEEK12_PART3_BEGIN -->
## Part 3: DeepSeek review of learning and inference excerpts

On 2026-11-27 ChatGPT sent the displayed feature formulas, label-export contract,
JAX network/loss/Adam code, checkpoint rule, model/probe schema and Rust inference
contract to DeepSeek in the same review conversation:
https://chat.deepseek.com/a/chat/s/fc47b89b-c851-4ed2-a26b-c7409e906862

Scope: independent static review of the supplied excerpts and interface
contracts. DeepSeek did not run the programs or review the entire source tree.
Its response reported no definite error in those excerpts and identified five
risks. The following are paraphrases and checked dispositions, not test claims
made by DeepSeek:

1. **Bond-gap meaning.** The expression involving 0.5*(1-s_i*s_j) is not an LP
   constraint-violation measure at fractional x. We retain it because it is the
   feature explicitly defined by the learning sheet; the formula is documented
   exactly and independently recomputed in Python.
2. **Dual scale and bond bias.** The excerpt omitted the dual-scale assignment.
   Full Rust code uses max(1e-9,max|row_dual|), and bond bias counts the signs of
   incident J, as specified. The feature check and hand-computed unit example
   cover these definitions.
3. **Policy-state distribution shift.** Training/validation labels come from
   Strong states, whereas deployment states and incumbents come from neural
   search. This is a limitation of imitation learning, not train/test leakage.
   The script preserves the required instance split and train-only statistics,
   then evaluates the fixed checkpoint on separately seeded unseen graphs.
4. **Masked padding and Adam.** Padding is finite zero after standardization,
   masks are boolean, each recorded node has at least two real candidates, and
   padded targets are zero. A finite -1e30 masked logit avoids zero times
   negative infinity. Adam moments start at zero; its bias-correction step
   increments once per batch from one. Training stops on non-finite CE and
   checks zero padded probability.
5. **Matrix layout and exactness.** Rust computes w1[input][hidden] and the same
   normalization, biases and tanh. Six probes compare actual probabilities and
   record logits. Neural only chooses an eligible spin and makes no trial LPs;
   the existing exact partition, incumbent and pruning logic are retained.
   Non-finite model coefficients/features/scores produce an error instead of
   allowing an unchecked argmax. A fallback is not needed to report failure
   safely. Both branch policies remain checked against actual exact energies.

ChatGPT's pre-delivery development checks passed four Rust tests and a separate
small n=6 label/train/evaluate/plot pipeline. That pipeline ran 120 JAX epochs
and checked six Rust/JAX probes, exact energies, no graph overlap, LP accounting
and the figure from raw JSON. It is not a substitute for the formal n=22 local
experiments. The bootstrap runs the formal dataset and unseen evaluation on the
user's computer and saves the actual results, including any failed criterion.
<!-- WEEK12_PART3_END -->

<!-- WEEK12_PART4_BEGIN -->
## Part 4: DeepSeek review of root caching and timing excerpts

On 2026-11-27, ChatGPT sent the actual new root-cache, population-normalization
and branch-selection snippets plus the timing/call-count contract to DeepSeek:
https://chat.deepseek.com/a/chat/s/fc47b89b-c851-4ed2-a26b-c7409e906862

Scope: independent static review of the displayed excerpts, not full-source
inspection or program execution. DeepSeek reported no definite error, with four
points to verify. Their dispositions were checked against the full source:

1. **Finite root features without an incumbent.** feature_rows already assigns
   gap_ratio=1 when the incumbent is not finite, avoiding infinity/infinity.
   Network inputs/outputs and standardized root scores are checked for finite
   values. The feature example explicitly uses an infinite initial incumbent.
2. **Cache indices.** all_spins is exactly 0..n, feature_rows maps the supplied
   indices in order, and scores preserves that row order. The cache has length
   n. The independent Python audit recomputes every root-policy branch choice
   from the saved cache and current LP fractions; the small Rust test does so
   independently through actual search traces as well.
3. **Raw-score evidence.** Full code stores both root_raw_scores and root_scores
   in the search result. Python recomputes population normalization from raw
   logits. Creating these records is inside the search; writing final JSON is
   outside its timer.
4. **Per-search counters, feasibility and model guards.** Caches, counts and
   timers are local variables initialized by every search invocation. Model
   presence is checked before entering either neural policy. For this spin-glass
   model the initial box is always feasible (for example all x=0,z=0), so every
   root-policy course instance executes exactly one feature and forward call.
   This is not a claim for arbitrary infeasible integer programs. Counters and
   solve accounting are checked from actual evidence.

Pre-delivery development validation passed five Rust tests and a separate
two-size n=6/n=8 inference/evaluation/plot pipeline using the prior small-graph
development model. Both learned policies retained exactness, zero trial LPs,
valid timing components and correct root-cache behavior. No course n=22/n=30
experiment was substituted for the user's local run. The final bootstrap uses
the user's unchanged trained model, preserves raw results before performance
checks, and generates its result tables only from those local files.
<!-- WEEK12_PART4_END -->
