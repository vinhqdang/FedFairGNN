/-
  NullPlayer.lean
  Formal verification of the Null-Player Property for FU-Shapley Aggregation in TrustFedGNN / FairShare-GNN.
  Proving:
  1. A client submitting zero gradient produces an exact zero raw Shapley score: <0, g_target> = 0
  2. ReLU gating preserves zero: relu(0) = 0
  3. The normalized aggregation weight for a null client strictly evaluates to 0 in all branches
-/

namespace TrustFedGNN

structure OrderedVectorSpace (V : Type) (F : Type) where
  add : F → F → F
  mul : F → F → F
  div : F → F → F
  zero : F
  one : F
  vzero : V
  inner : V → V → F
  relu : F → F
  -- Vector space & field axioms
  inner_zero_left : ∀ (v : V), inner vzero v = zero
  relu_zero : relu zero = zero
  mul_zero_left : ∀ (a : F), mul zero a = zero
  div_zero : ∀ (c : F), div zero c = zero

variable {V F : Type} (OVS : OrderedVectorSpace V F)

-- Theorem 1 (Zero Raw Contribution for Null Player):
-- For any server target gradient g_target, a client submitting zero pseudo-gradient yields zero score.
theorem null_player_zero_score (g_target : V) :
    OVS.inner OVS.vzero g_target = OVS.zero := by
  exact OVS.inner_zero_left g_target

-- Theorem 2 (Zero ReLU-Gated Score):
-- The ReLU activation applied to the null player's score evaluates to zero.
theorem null_player_relu_zero (g_target : V) :
    OVS.relu (OVS.inner OVS.vzero g_target) = OVS.zero := by
  rw [OVS.inner_zero_left g_target]
  exact OVS.relu_zero

-- Theorem 3 (Strictly Zero Aggregation Weight for Null Player):
-- When normalized by any non-zero total sum across clients, the null player receives weight 0.
theorem null_player_weight_zero (g_target : V) (total : F) :
    OVS.div (OVS.relu (OVS.inner OVS.vzero g_target)) total = OVS.zero := by
  rw [OVS.inner_zero_left g_target]
  rw [OVS.relu_zero]
  exact OVS.div_zero total

end TrustFedGNN
