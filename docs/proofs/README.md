# Machine-Checked Formal Proofs in Lean 4

This directory contains interactive theorem proofs in **Lean 4**, formally certifying four core algebraic and geometric properties of TrustFedGNN's aggregation mechanism and orthogonal projection.

## Files & Verified Theorems

| File | Corresponding Math Result | Description |
|---|---|---|
| [`OrthogonalProjection.lean`](OrthogonalProjection.lean) | Theorem 4 (Fairness-Subspace Non-Leakage) | Proves that for the FTGD update $g_{\text{task}}^\perp$, the inner product $\langle g_{\text{task}}^\perp, g_{\text{fair}}\rangle$ vanishes exactly ($\varepsilon = 0$), eliminating the linear channel of sensitive attribute leakage. |
| [`SimplexProperties.lean`](SimplexProperties.lean) | Proposition 2 (Simplex Validity) | Proves that whenever at least one active client has positive score $\bar{\varphi}_k > 0$, the FU-Shapley weights satisfy non-negativity $w_k \ge 0$ and sum-to-one $\sum_{k=1}^K w_k = 1$. |
| [`NullPlayer.lean`](NullPlayer.lean) | Proposition 3 (Null Player Axiom) | Proves that a non-contributing client with zero update ($g_k = \mathbf{0}$) receives exactly zero aggregation weight ($w_k = 0$) across all execution paths, including fallback branches. |
| [`LinearDecomposition.lean`](LinearDecomposition.lean) | Proposition 4 (Additive Decomposition) | Formally proves the exact bilinearity split of client contribution into utility and fairness components: $\varphi_k = \varphi_k^{\text{util}} + \alpha \varphi_k^{\text{fair}}$. |

## Verification Command
To verify with Lean 4:
```bash
lean docs/proofs/OrthogonalProjection.lean
lean docs/proofs/SimplexProperties.lean
lean docs/proofs/NullPlayer.lean
lean docs/proofs/LinearDecomposition.lean
```
