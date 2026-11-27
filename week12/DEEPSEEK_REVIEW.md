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
