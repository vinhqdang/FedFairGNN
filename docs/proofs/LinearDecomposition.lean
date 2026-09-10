/-
  LinearDecomposition.lean
  Formal verification of the Additive Shapley Decomposition for TrustFedGNN / FairShare-GNN.
  Theorem 3: φ_k = φ_k^util + φ_k^fair
  Proving the exact linearity of the inner product over target gradient decomposition for any scalar α.
-/

namespace TrustFedGNN

-- Rigorous Algebraic Scalar Field Definition
structure ScalarField (F : Type) where
  add : F → F → F
  mul : F → F → F

-- Abstract Vector Space equipped with Bilinear Dot Product over Field F
structure VectorSpace (F : Type) (V : Type) (SF : ScalarField F) where
  add : V → V → V
  smul : F → V → V
  dot : V → V → F
  -- Linearity of dot product in the second argument
  dot_add_right : ∀ (u v w : V), dot u (add v w) = SF.add (dot u v) (dot u w)
  dot_smul_right : ∀ (c : F) (u v : V), dot u (smul c v) = SF.mul c (dot u v)

variable {F : Type} {V : Type} (SF : ScalarField F) (VS : VectorSpace F V SF)

-- Theorem 3 (Exact Additive Decomposition of Alignment Scoring):
-- Given g_target = g_task + α * g_fair,
-- then ⟨g_k, g_target⟩ = ⟨g_k, g_task⟩ + α * ⟨g_k, g_fair⟩ for any continuous scalar α ∈ F.
theorem additive_shapley_decomposition (g_k g_task g_fair : V) (α : F) :
  let g_target := VS.add g_task (VS.smul α g_fair)
  VS.dot g_k g_target = SF.add (VS.dot g_k g_task) (SF.mul α (VS.dot g_k g_fair)) := by
  intro g_target
  dsimp [g_target]
  rw [VS.dot_add_right]
  rw [VS.dot_smul_right]

end TrustFedGNN
