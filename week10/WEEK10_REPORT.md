# Week 10 Report --- Approximate Tensor Contraction for the 2D Ising Model

## 1. Objective

This exercise studies how matrix and tensor compression make large Ising
partition-function calculations tractable. The model is the zero-field
square-lattice Ising model with (s_i=`\pm1`{=tex}) and (K=1/T). The
partition function is represented as a tensor network, and the quantity
\[ g(K)=`\frac{\log Z(K)}{L^2}`{=tex} \] is used to obtain the specific
heat per spin from its curvature: \[
c_V(K;h)`\approx `{=tex}K\^2`\frac{g(K+h)-2g(K)+g(K-h)}{h^2}`{=tex}. \]
The reference critical temperature is
(T_c=2/`\log`{=tex}(1+`\sqrt2`{=tex})`\approx2.269185314213`{=tex}).

## 2. Part 1 --- SVD compression

A (512`\times512`{=tex}) matrix of independent standard-normal entries
was generated with `ChaCha8Rng`, using seed 2026. The thin SVD was
computed, and reconstructions were saved at retained ranks
(`\chi=16`{=tex},64,128,256,512).

For a truncated SVD, the relative Frobenius error is predicted by the
discarded singular-value tail: \[ e\_`\chi`{=tex}=
`\sqrt{\frac{\sum_{\alpha>\chi}s_\alpha^2}
{\sum_\alpha s_\alpha^2}}`{=tex}. \] The program reloads the saved
original matrix and each reconstruction, independently measures the
reconstruction error, and compares it with this prediction. The
implementation enforces a difference below (10\^{-8}), and a full-rank
reconstruction error below (10\^{-8}).

For a square (512`\times512`{=tex}) matrix, the original storage is
(512\^2) entries. Two rank-(`\chi`{=tex}) factors require
(512`\chi`{=tex}+`\chi512`{=tex}=1024`\chi`{=tex}) entries, so the
storage break-even point is (`\chi=256`{=tex}). The plotted error
reaches 10% only at a rank above this point (approximately
(`\chi=395`{=tex})). Therefore, the tested matrix does not admit a rank
that simultaneously reaches 10% relative error and reduces storage. The
singular spectrum decays gradually, so a relatively large number of
singular directions must be retained.

## 3. Part 2 --- Boundary MPS on open squares

The bond matrix is factorized into (W) factors, which are combined
around each spin to form a four-leg site tensor. At an open edge, the
missing bond is represented by the boundary vector. Rows are absorbed
successively into a boundary MPS. Canonical SVD sweeps compress the MPS
bonds back to the maximum retained dimension (`\chi`{=tex}).

The recorded peak study used open squares with (L=16,32,64). The coarse
grid used (K=0.40) to (0.50) with spacing 0.0025. Nine-point local
refinements used spacing 0.00125 at (`\chi=16`{=tex}) and
(`\chi=32`{=tex}). The finite-size peak temperatures were extrapolated
with (T\_{`\mathrm{peak}`{=tex}}(L)=T_c+a/L).

  Result                                   Value
  ----------------------------- ----------------
  Exact reference (T_c)           2.269185314213
  Fitted (T_c), all sizes         2.267506445602
  Fitted (T_c), (L=32,64)         2.268763780635
  Relative error, all sizes          0.07398552%
  Relative error, largest two        0.01857643%

The largest-two fit is closer to the reference value. This is consistent
with the smallest system having the strongest finite-size shift. The
peak temperature moves toward the bulk transition as the lattice size
increases.

An independent small-system check compared the finite-difference
specific heat on the open (4`\times4`{=tex}) lattice with the exact
energy-variance result:

  Check                                                        Value
  ------------------------- ----------------------------------------
  Finite-difference (c_V)                             0.439393072554
  Exact-variance (c_V)                                0.439396912160
  Absolute difference         (3.839606051159`\times10`{=tex}\^{-6})

The check passed. The recorded analysis-script time was 89.042 s.

## 4. Part 3 --- Periodic TRG

The periodic TRG method repeatedly coarse-grains a translationally
invariant four-leg tensor. Each step normalizes the tensor by its
maximum absolute entry, records the associated scale contribution to
(`\log `{=tex}Z), performs two diagonal SVD splits, and regroups four
half-tensors into a new effective tensor. The number of tensors is
halved at each step. Two steps halve the linear side length, so a
power-of-two lattice of side (L) requires (2`\log`{=tex}\_2L) steps.

The final periodic closure joins opposite legs in pairs: \[
z=`\sum`{=tex}*{i,j}A*{ijij}. \] The two sums are independent; setting
all four indices equal would represent a different contraction.

A completed (L=1024), (K=0.44), (`\chi=16`{=tex}) run performed 20 TRG
steps and ended with final tensor dimension 16. Its recorded runtime was
approximately 145.869 s, with
(`\log `{=tex}Z`\approx9.738376577832154`{=tex}`\times10`{=tex}\^5) and
(g`\approx0.9287239625770716`{=tex}).

### 4.1 Coarse peak results

The available coarse peak table at (`\chi=16`{=tex}) and spacing
(h=0.0025) is:

  ----------------------------------------------------------------------------------------------------------------
              \(L\)   (K\_{`\mathrm{peak}`{=tex}})   (T\_{`\mathrm{peak}`{=tex}})   (c\_{V,`\mathrm{peak}`{=tex}})
  ----------------- ------------------------------ ------------------------------ --------------------------------
                 64                         0.4375                       2.285714                         2.251706

                256                         0.4400                       2.272727                         2.824852

               1024                         0.4400                       2.272727                         2.948088
  ----------------------------------------------------------------------------------------------------------------

The peak height increases with lattice size in this coarse data. The
all-size fit against (1/L) gives (T_c`\approx2.2705627706`{=tex}),
approximately 0.0607% from the Onsager value. The fit using only
(L=256,1024) gives (T_c`\approx2.2727272727`{=tex}), approximately
0.1561% from the reference.

These estimates are preliminary. The two largest lattices have the same
coarse-grid peak (K=0.44), so the two-size extrapolation cannot resolve
their small peak shift. A finer local grid and a larger retained rank
would be needed to assess this reliably.

### 4.2 Scope and limitations

The requested TRG refinement---nine points around each coarse peak at
spacing 0.00125 for both (`\chi=16`{=tex}) and (`\chi=32`{=tex})---was
intentionally omitted to avoid the long runtime. Therefore:

-   the available periodic peak estimates are coarse-grid,
    (`\chi=16`{=tex}) results;
-   rank dependence ((`\chi=16`{=tex}) versus (`\chi=32`{=tex})) has not
    been measured;
-   step-size dependence ((h=0.0025) versus (h=0.00125)) has not been
    measured;
-   the final refined (`\chi=32`{=tex}) critical-temperature
    extrapolation is not available.

The coarse result is useful as a preliminary trend and as a check that
the peak moves toward the known transition, but it is not equivalent to
the full refined TRG study requested in the learning sheet.

## 5. Software verification

After the TRG module was added, the local terminal reported 15 tests
passed and 0 failed for `cargo test --bin thermo`. The tests cover local
tensor construction, boundary MPS operations, periodic TRG contraction,
periodic exact enumeration at small sizes, and the (L=2) specific-heat
check.

## 6. Conclusion

The SVD experiment shows that low-rank approximation is controlled by
the singular-value tail; for the tested Gaussian matrix, the rank needed
for 10% error is above the storage break-even rank. The boundary-MPS
calculation gives a critical-temperature estimate within 0.02% when the
smallest size is omitted, and its (4`\times4`{=tex}) specific-heat check
agrees closely with exact enumeration. The periodic TRG code
successfully contracts a (1024`\times1024`{=tex}) system in 20
coarse-graining steps and produces coarse peak estimates close to the
Onsager transition. However, because the TRG fine scan was not run, its
rank and step-size stability remain unverified; the periodic (T_c)
result should therefore be reported as preliminary.
