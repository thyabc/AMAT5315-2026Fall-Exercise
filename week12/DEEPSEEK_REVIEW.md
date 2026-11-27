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
