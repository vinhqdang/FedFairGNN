/-
  SimplexProperties.lean
  Formal verification of the Simplex Normalization and Non-Negativity properties for TrustFedGNN / FU-Shapley / BFWA.
  Proving:
  1. Generalized non-negativity of weights for an arbitrary list of K non-negative components
  2. Generalized exact sum-to-one simplex normalization condition for arbitrary K clients
  3. Closed-form 2-client specialization
-/

namespace TrustFedGNN

-- Rigorous Ordered Field Structure for Continuous Probability Simplex Weights
structure OrderedField (F : Type) where
  add : F → F → F
  div : F → F → F
  zero : F
  one : F
  le : F → F → Prop
  lt : F → F → Prop
  -- Field axioms
  add_div : ∀ (a b c : F), div (add a b) c = add (div a c) (div b c)
  div_zero : ∀ (c : F), div zero c = zero
  div_self : ∀ (a : F), a ≠ zero → div a a = one
  div_nonneg : ∀ (a b : F), le zero a → lt zero b → le zero (div a b)
  le_refl : ∀ (a : F), le a a
  lt_imp_ne_zero : ∀ (a : F), lt zero a → a ≠ zero

variable {F : Type} (OF : OrderedField F)

-- =========================================================================
-- Part 1: Generalized K-Client Simplex Formalization
-- =========================================================================

-- Recursive sum of a list of weights
def sumList : List F → F
  | [] => OF.zero
  | x :: xs => OF.add x (sumList xs)

-- Component-wise normalization by total sum
def normalizeList (tot : F) : List F → List F
  | [] => []
  | x :: xs => OF.div x tot :: normalizeList tot xs

-- Predicate: all components are non-negative
def allNonneg : List F → Prop
  | [] => True
  | x :: xs => OF.le OF.zero x ∧ allNonneg xs

-- Theorem 1 (Generalized Non-Negativity):
-- For any list of non-negative weights and positive total, all normalized weights are non-negative.
theorem generalized_normalize_nonneg (ws : List F) (tot : F)
    (hws : allNonneg OF ws) (htot : OF.lt OF.zero tot) :
    allNonneg OF (normalizeList OF tot ws) := by
  induction ws with
  | nil =>
    dsimp [normalizeList, allNonneg]
  | cons x xs ih =>
    dsimp [normalizeList, allNonneg] at *
    exact ⟨OF.div_nonneg x tot hws.1 htot, ih hws.2⟩

-- Lemma: Normalization distributes over summation
theorem sum_normalize_distrib (ws : List F) (tot : F) :
    sumList OF (normalizeList OF tot ws) = OF.div (sumList OF ws) tot := by
  induction ws with
  | nil =>
    dsimp [sumList, normalizeList]
    rw [OF.div_zero]
  | cons x xs ih =>
    dsimp [sumList, normalizeList]
    rw [ih]
    rw [← OF.add_div]

-- Theorem 2 (Generalized Exact Sum-to-One Normalization):
-- For any list of weights with positive sum, the normalized vector strictly sums to 1.
theorem generalized_normalize_sum_to_one (ws : List F) (h_pos : OF.lt OF.zero (sumList OF ws)) :
    sumList OF (normalizeList OF (sumList OF ws) ws) = OF.one := by
  have h_ne : sumList OF ws ≠ OF.zero := OF.lt_imp_ne_zero (sumList OF ws) h_pos
  rw [sum_normalize_distrib OF ws (sumList OF ws)]
  exact OF.div_self (sumList OF ws) h_ne

-- =========================================================================
-- Part 2: Two-Client Specialization (Closed Form)
-- =========================================================================

structure SimplexWeights2 (F : Type) where
  p1 : F
  p2 : F

def normalize2 (w1 w2 : F) : SimplexWeights2 F :=
  let total := OF.add w1 w2
  ⟨OF.div w1 total, OF.div w2 total⟩

theorem normalized_weights2_nonneg
    (w1 w2 : F)
    (hw1 : OF.le OF.zero w1)
    (hw2 : OF.le OF.zero w2)
    (h_tot_pos : OF.lt OF.zero (OF.add w1 w2)) :
    let p := normalize2 OF w1 w2
    OF.le OF.zero p.p1 ∧ OF.le OF.zero p.p2 := by
  intro p
  dsimp [p, normalize2]
  have hp1 : OF.le OF.zero (OF.div w1 (OF.add w1 w2)) :=
    OF.div_nonneg w1 (OF.add w1 w2) hw1 h_tot_pos
  have hp2 : OF.le OF.zero (OF.div w2 (OF.add w1 w2)) :=
    OF.div_nonneg w2 (OF.add w1 w2) hw2 h_tot_pos
  exact ⟨hp1, hp2⟩

theorem normalized_weights2_sum_to_one
    (w1 w2 : F)
    (h_tot_pos : OF.lt OF.zero (OF.add w1 w2)) :
    let p := normalize2 OF w1 w2
    OF.add p.p1 p.p2 = OF.one := by
  intro p
  dsimp [p, normalize2]
  have h_ne_zero : OF.add w1 w2 ≠ OF.zero := OF.lt_imp_ne_zero (OF.add w1 w2) h_tot_pos
  rw [← OF.add_div w1 w2 (OF.add w1 w2)]
  rw [OF.div_self (OF.add w1 w2) h_ne_zero]

end TrustFedGNN
