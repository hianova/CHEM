/-
  ChemCert — Formal Verification Layer for CHEM Anti-Hallucination Certificates
  ============================================================================
  Imports `Flow.MolecularBundle` and `Flow.Interval` (zero `sorryAx`) to certify:
    1. Mode A spin-adapted block trace invariants (`1*D0 + 3*D1`, `Q`, `G`);
    2. Unconditional anytime dual weak-duality + Weyl/roundoff lower bound;
    3. Anytime Early-Veto soundness (`E_lb_cert > E_ai - tol -> E_ai < E_exact + tol`);
    4. Certified spin-state ordering & gap enclosure via `Flow.Interval.Interval1D`;
    5. Mode B exact spin-squared trace identity (`4 * swap_ab`);
    6. Instantiation of `Flow.MolecularBundle.certified_energy_bracket_from_sos`.
-/
import Flow.MolecularBundle
import Flow.Interval

namespace ChemCert

open Flow.Interval
open Flow.MolecularBundle
open Flow.GaugeBundle

/-- Theorem C1 (Mode A Spin-Adapted Trace Invariants):
    With `N = 2 * ne` electrons in `K` spatial orbitals, weighting singlet (`m=1`)
    and triplet (`m=3`) blocks reproduces the exact spin-orbital pair traces. -/
theorem mode_a_spin_trace_invariants (K ne : Int) :
    let N := 2 * ne
    let trD0_2 := ne * (ne + 1)
    let trD1_2 := ne * (ne - 1)
    let trQ0_2 := (K - ne) * (K - ne + 1)
    let trQ1_2 := (K - ne) * (K - ne - 1)
    let trG0 := ne * (K - ne + 2)
    let trG1 := ne * (K - ne)
    1 * trD0_2 + 3 * trD1_2 = N * (N - 1) ∧
    1 * trQ0_2 + 3 * trQ1_2 = (2 * K - N) * (2 * K - N - 1) ∧
    1 * trG0 + 3 * trG1 = N * (2 * K - N + 1) := by
  refine ⟨?_, ?_, ?_⟩ <;> grind

/-- Theorem C2 (Unconditional Anytime Dual Weyl Lower Bound):
    For an arbitrary block count (covering both 6-block Mode A and 9-block Mode B),
    given exact energy decomposition `E_state = E_dual + ∑_b slack_b`, non-negative
    block traces `0 ≤ tr_b`, certified eigenvalue floors `tr_b * lam_cert_b ≤ slack_b`,
    and conservative roundoff bound `0 ≤ delta_round`, shifting negative parts yields
    a valid anytime lower bound `E_lb_cert ≤ E_state`. -/
theorem anytime_dual_weyl_lower_bound {B : Nat}
    (E_state E_dual delta_round : Int)
    (slack tr lam_cert : Fin B → Int)
    (hDecomp : E_state = E_dual + Flow.GaugeBundle.sumFin B slack)
    (hTr : ∀ b, 0 ≤ tr b)
    (hFloor : ∀ b, tr b * lam_cert b ≤ slack b)
    (hDelta : 0 ≤ delta_round) :
    E_dual + Flow.GaugeBundle.sumFin B (fun b => tr b * negPart (lam_cert b)) - delta_round ≤
      E_state := by
  have hsum_le :
      Flow.GaugeBundle.sumFin B (fun b => tr b * negPart (lam_cert b)) ≤
        Flow.GaugeBundle.sumFin B slack := by
    have hdiff :
        0 ≤ Flow.GaugeBundle.sumFin B (fun b => slack b - tr b * negPart (lam_cert b)) := by
      apply Flow.GaugeBundle.sumFin_nonneg
      intro b
      have hn : negPart (lam_cert b) ≤ lam_cert b := by
        unfold negPart
        split <;> omega
      have hmul : tr b * negPart (lam_cert b) ≤ tr b * lam_cert b :=
        Int.mul_le_mul_of_nonneg_left hn (hTr b)
      have hb := hFloor b
      omega
    have hfn :
        (fun b => slack b - tr b * negPart (lam_cert b)) =
          (fun b => slack b - tr b * negPart (lam_cert b)) := rfl
    rw [Flow.GaugeBundle.sumFin_sub] at hdiff
    omega
  omega

/-- Theorem C3 (Anytime Early-Veto Soundness):
    If `E_lb_cert ≤ E_exact` and the veto trigger `E_ai - tol < E_lb_cert` fires,
    then the candidate AI/single-reference prediction strictly underestimates reality
    (`E_ai < E_exact + tol`). -/
theorem anytime_early_veto_soundness (E_lb_cert E_exact E_ai tol : Int)
    (hLB : E_lb_cert ≤ E_exact) (hVeto : E_ai - tol < E_lb_cert) :
    E_ai < E_exact + tol := by
  omega

/-- Theorem C4 (Certified Spin-State Ordering & Gap Enclosure via `Flow.Interval`):
    Given disjoint certified brackets `ivA = [lbA, ubA]` and `ivB = [lbB, ubB]`
    with `ivA.hi < ivB.lo`, `E_A < E_B` holds strictly and `E_B - E_A` is enclosed
    within `[ivB.lo - ivA.hi, ivB.hi - ivA.lo]`. -/
theorem certified_spin_gap_ordering (ivA ivB : Interval1D) (E_A E_B : Int)
    (hA : ivA.contains E_A) (hB : ivB.contains E_B) (hSep : ivA.hi < ivB.lo) :
    E_A < E_B ∧
    ({ lo := ivB.lo - ivA.hi, hi := ivB.hi - ivA.lo } : Interval1D).contains (E_B - E_A) := by
  unfold Interval1D.contains at *
  dsimp
  omega

/-- Theorem C5 (Mode B Spin-Squared Exact Identity):
    Integer form of `4 * swap_ab = 2*(na + nb) + (na - nb)^2 - s2*(s2 + 2)`,
    reducing cleanly to `swap_ab = nb` when `s2 = na - nb` (e.g. `S = Sz`). -/
theorem mode_b_spin_squared_identity (na nb s2 swap_ab : Int)
    (hEq : 4 * swap_ab = 2 * (na + nb) + (na - nb) * (na - nb) - s2 * (s2 + 2)) :
    (s2 = na - nb → swap_ab = nb) ∧
    ((na = nb ∧ s2 = 0) → swap_ab = nb) := by
  constructor
  · intro hs
    subst hs
    have h4 : 4 * swap_ab = 4 * nb := by
      calc
        4 * swap_ab = 2 * (na + nb) + (na - nb) * (na - nb) - (na - nb) * (na - nb + 2) := hEq
        _ = 4 * nb := by grind
    omega
  · intro ⟨hnb, hs0⟩
    subst hnb
    subst hs0
    have h4 : 4 * swap_ab = 4 * na := by
      calc
        4 * swap_ab = 2 * (na + na) + (na - na) * (na - na) - 0 * (0 + 2) := hEq
        _ = 4 * na := by grind
    omega

/-- Theorem C6 (CHEM SOS Energy Bracket Instantiation from `Flow.MolecularBundle`):
    Directly instantiates `certified_energy_bracket_from_sos` for discrete-scaled
    `CHEM` lower/upper bounds `[E_lb, E_ub]`. -/
theorem chem_bracket_from_flow_molecular_bundle {m : Nat} (w lam : Fin m → Int)
    (E_lb E_exact E_ub : Int)
    (hw : ∀ i, 0 ≤ w i) (hwpos : 0 < specW0 m w)
    (hlam : ∀ i, E_exact ≤ lam i)
    (hlb : E_lb ≤ E_exact) (hub : specW1 m w lam ≤ E_ub * specW0 m w) :
    let iv : Interval1D := { lo := E_lb, hi := E_ub }
    iv.contains E_exact ∧ 0 ≤ iv.width :=
  certified_energy_bracket_from_sos w lam E_lb E_exact E_ub hw hwpos hlam hlb hub

/-- Theorem C7 (Certified Observable & Force Envelope Bounds):
    Given an energy upper bound `E_state ≤ E_ub` and non-negative Lagrange multipliers
    `0 ≤ lam_lo, lam_hi` solving `L_lo ≤ O + lam_lo * H` and `L_hi ≤ -O + lam_hi * H`,
    the certified interval `[L_lo - lam_lo * E_ub, lam_hi * E_ub - L_hi]` encloses `O_state`. -/
theorem certified_observable_envelope_bounds
    (O_state E_state E_ub lam_lo lam_hi L_lo L_hi : Int)
    (hLamLo : 0 ≤ lam_lo) (hLamHi : 0 ≤ lam_hi)
    (hEub : E_state ≤ E_ub)
    (hDualLo : L_lo ≤ O_state + lam_lo * E_state)
    (hDualHi : L_hi ≤ -O_state + lam_hi * E_state) :
    ({ lo := L_lo - lam_lo * E_ub, hi := lam_hi * E_ub - L_hi } : Interval1D).contains O_state := by
  have hmulLo : lam_lo * E_state ≤ lam_lo * E_ub := Int.mul_le_mul_of_nonneg_left hEub hLamLo
  have hmulHi : lam_hi * E_state ≤ lam_hi * E_ub := Int.mul_le_mul_of_nonneg_left hEub hLamHi
  unfold Interval1D.contains
  dsimp
  omega

/-- Theorem C8 (Subspace T2 Constant Trace Upper Bound for Weyl Shift):
    For any projected partial 3-RDM subspace block whose true trace satisfies
    `0 ≤ tr_sub ≤ tr_full` and `tr_sub ≤ dim_sub`, replacing `tr_sub` by
    `min dim_sub tr_full` when `lam_cert ≤ 0` yields an unconditionally valid lower bound. -/
theorem subspace_t2_trace_upper_bound
    (tr_sub tr_full dim_sub lam_cert : Int)
    (hSubLeFull : tr_sub ≤ tr_full)
    (hSubLeDim : tr_sub ≤ dim_sub)
    (hLam : lam_cert ≤ 0) :
    min dim_sub tr_full * lam_cert ≤ tr_sub * lam_cert := by
  have hmin : tr_sub ≤ min dim_sub tr_full := by
    rw [Int.min_def]
    split <;> omega
  exact Int.mul_le_mul_of_nonpos_right hmin hLam

/-! ## v0.3.0 Formal Foundations: Zero-FP-Trust Cholesky+SDD, GPC Borland–Dennis, Phase Transitions & Eckart Fidelity -/

private theorem int_sq_nonneg_c (z : Int) : 0 ≤ z * z := by
  by_cases hz : 0 ≤ z
  · exact Int.mul_nonneg hz hz
  · have hneg : 0 ≤ -z := by omega
    have hsq : (-z) * (-z) = z * z := Int.neg_mul_neg z z
    rw [← hsq]
    exact Int.mul_nonneg hneg hneg

private theorem sumFin_congr_c {N : Nat} (f g : Fin N → Int) (h : ∀ i, f i = g i) :
    sumFin N f = sumFin N g := by
  have hfn : f = g := funext h
  rw [hfn]

private theorem sumFin_mul_right_c {N : Nat} (f : Fin N → Int) (c : Int) :
    sumFin N (fun i => f i * c) = sumFin N f * c := by
  calc
    sumFin N (fun i => f i * c)
        = sumFin N (fun i => c * f i) := sumFin_congr_c _ _ (fun i => Int.mul_comm _ _)
    _ = c * sumFin N f := Flow.GaugeBundle.sumFin_mul_left N c f
    _ = sumFin N f * c := Int.mul_comm _ _

private theorem sumPrefix_swap_c {N M : Nat} (f : Fin N → Fin M → Int) :
    ∀ k : Nat,
      sumPrefix (fun i : Fin N => sumPrefix (fun j : Fin M => f i j) M) k =
        sumPrefix (fun j : Fin M => sumPrefix (fun i : Fin N => f i j) k) M
  | 0 => by
    simp only [sumPrefix]
    symm
    exact Flow.GaugeBundle.sumPrefix_zero M
  | k + 1 => by
    have ih := sumPrefix_swap_c f k
    have hstep :
        sumPrefix (fun j : Fin M => sumPrefix (fun i : Fin N => f i j) (k + 1)) M =
          sumPrefix (fun j : Fin M =>
            sumPrefix (fun i : Fin N => f i j) k + (if h : k < N then f ⟨k, h⟩ j else 0)) M := rfl
    rw [hstep, Flow.GaugeBundle.sumPrefix_add, ← ih]
    simp only [sumPrefix]
    by_cases hk : k < N
    · simp [hk]
    · simp only [hk, ↓reduceDIte, Int.add_zero]
      have hz := Flow.GaugeBundle.sumPrefix_zero (N := M) M
      simp only [hz, Int.add_zero]

private theorem sumFin_swap_c {N M : Nat} (f : Fin N → Fin M → Int) :
    sumFin N (fun i => sumFin M (fun j => f i j)) =
      sumFin M (fun j => sumFin N (fun i => f i j)) :=
  sumPrefix_swap_c f N

def vecNormSq (n : Nat) (v : Fin n → Int) : Int :=
  sumFin n (fun i => v i * v i)

def matQuad (n : Nat) (M : Fin n → Fin n → Int) (v : Fin n → Int) : Int :=
  sumFin n (fun i => sumFin n (fun j => v i * M i j * v j))

def gramLLt (n : Nat) (L : Fin n → Fin n → Int) (i j : Fin n) : Int :=
  sumFin n (fun k => L i k * L j k)

def absInt (z : Int) : Int :=
  if 0 ≤ z then z else -z

def IsSDD (n : Nat) (E : Fin n → Fin n → Int) : Prop :=
  (∀ i j, E i j = E j i) ∧
  (∀ i, sumFin n (fun j => if j = i then 0 else absInt (E i j)) ≤ E i i)

private theorem sdd_pair_nonneg (a x y : Int) :
    0 ≤ absInt a * (x * x) + 2 * (a * (x * y)) + absInt a * (y * y) := by
  by_cases ha : 0 ≤ a
  · have hfa : absInt a = a := by simp [absInt, ha]
    rw [hfa]
    have hEq : a * (x * x) + 2 * (a * (x * y)) + a * (y * y) = a * ((x + y) * (x + y)) := by grind
    rw [hEq]
    exact Int.mul_nonneg ha (int_sq_nonneg_c (x + y))
  · have haNeg : 0 ≤ -a := by omega
    have hfa : absInt a = -a := by simp [absInt, ha]
    rw [hfa]
    have hEq : (-a) * (x * x) + 2 * (a * (x * y)) + (-a) * (y * y) = (-a) * ((x - y) * (x - y)) := by grind
    rw [hEq]
    exact Int.mul_nonneg haNeg (int_sq_nonneg_c (x - y))

/-- Theorem C9 (Symmetric Diagonally Dominant Quadratic Form Nonnegativity):
    Any symmetric diagonally dominant `n × n` integer matrix defines a PSD quadratic form. -/
theorem sdd_quadratic_form_nonneg (n : Nat) (E : Fin n → Fin n → Int)
    (hSDD : IsSDD n E) : ∀ v : Fin n → Int, 0 ≤ matQuad n E v := by
  intro v
  rcases hSDD with ⟨hSymm, hDom⟩
  let offDiag : Fin n → Int := fun i => sumFin n (fun j => if j = i then 0 else absInt (E i j))
  let diagSum : Int := sumFin n (fun i => (E i i - offDiag i) * (v i * v i))
  let offSqI : Int := sumFin n (fun i => offDiag i * (v i * v i))
  let pairSum : Int :=
    sumFin n (fun i => sumFin n (fun j =>
      if j = i then 0 else absInt (E i j) * (v i * v i) + 2 * (E i j * (v i * v j)) + absInt (E i j) * (v j * v j)))
  have h1 : matQuad n E v = diagSum + offSqI + sumFin n (fun i => sumFin n (fun j => if j = i then 0 else E i j * (v i * v j))) := by
    have hA : matQuad n E v =
        sumFin n (fun i => E i i * (v i * v i) + sumFin n (fun j => if j = i then 0 else E i j * (v i * v j))) := by
      unfold matQuad
      apply sumFin_congr_c
      intro i
      have hSplit :
          (fun j : Fin n => v i * E i j * v j) =
            (fun j : Fin n => (if j = i then E i i * (v i * v i) else 0) + (if j = i then 0 else E i j * (v i * v j))) := by
        funext j
        by_cases hj : j = i
        · subst hj; grind
        · simp [hj]; grind
      rw [hSplit, Flow.GaugeBundle.sumFin_add, Flow.GaugeBundle.sumFin_single_hit]
    rw [hA, Flow.GaugeBundle.sumFin_add]
    have hD : sumFin n (fun i => E i i * (v i * v i)) = diagSum + offSqI := by
      have hSplit : (fun i => E i i * (v i * v i)) = (fun i => (E i i - offDiag i) * (v i * v i) + offDiag i * (v i * v i)) := by
        funext i; grind
      rw [hSplit, Flow.GaugeBundle.sumFin_add]
    rw [hD]
  have hOff1 : offSqI = sumFin n (fun i => sumFin n (fun j => if j = i then 0 else absInt (E i j) * (v i * v i))) := by
    unfold offSqI
    apply sumFin_congr_c
    intro i
    change sumFin n (fun j => if j = i then 0 else absInt (E i j)) * (v i * v i) = _
    rw [← sumFin_mul_right_c]
    apply sumFin_congr_c
    intro j
    by_cases hj : j = i <;> simp [hj]
  have hOff2 : offSqI = sumFin n (fun i => sumFin n (fun j => if j = i then 0 else absInt (E i j) * (v j * v j))) := by
    rw [hOff1]
    have hSw := sumFin_swap_c (fun i j : Fin n => if j = i then 0 else absInt (E i j) * (v i * v i))
    rw [hSw]
    apply sumFin_congr_c
    intro i
    apply sumFin_congr_c
    intro j
    by_cases hji : j = i
    · subst hji; rfl
    · have hij' : i ≠ j := fun h => hji h.symm
      simp only [hji, hij', ↓reduceIte]
      rw [hSymm j i]
  have hPairEq :
      2 * offSqI + 2 * sumFin n (fun i => sumFin n (fun j => if j = i then 0 else E i j * (v i * v j))) = pairSum := by
    have hCross :
        2 * sumFin n (fun i => sumFin n (fun j => if j = i then 0 else E i j * (v i * v j))) =
          sumFin n (fun i => sumFin n (fun j => if j = i then 0 else 2 * (E i j * (v i * v j)))) := by
      rw [← Flow.GaugeBundle.sumFin_mul_left]
      apply sumFin_congr_c
      intro i
      rw [← Flow.GaugeBundle.sumFin_mul_left]
      apply sumFin_congr_c
      intro j
      by_cases hj : j = i <;> simp [hj]
    have hSumOff :
        offSqI + offSqI =
          sumFin n (fun i => sumFin n (fun j =>
            if j = i then 0 else absInt (E i j) * (v i * v i) + absInt (E i j) * (v j * v j))) := by
      calc offSqI + offSqI
          = sumFin n (fun i => sumFin n (fun j => if j = i then 0 else absInt (E i j) * (v i * v i))) +
              sumFin n (fun i => sumFin n (fun j => if j = i then 0 else absInt (E i j) * (v j * v j))) := by
                rw [← hOff1, ← hOff2]
        _ = sumFin n (fun i =>
              sumFin n (fun j => if j = i then 0 else absInt (E i j) * (v i * v i)) +
              sumFin n (fun j => if j = i then 0 else absInt (E i j) * (v j * v j))) := by
                rw [Flow.GaugeBundle.sumFin_add]
        _ = sumFin n (fun i => sumFin n (fun j =>
              if j = i then 0 else absInt (E i j) * (v i * v i) + absInt (E i j) * (v j * v j))) := by
                apply sumFin_congr_c; intro i
                rw [← Flow.GaugeBundle.sumFin_add]
                apply sumFin_congr_c; intro j
                by_cases hj : j = i <;> simp [hj]
    have h2 : 2 * offSqI = offSqI + offSqI := by omega
    rw [h2, hSumOff, hCross]
    unfold pairSum
    rw [← Flow.GaugeBundle.sumFin_add]
    apply sumFin_congr_c
    intro i
    rw [← Flow.GaugeBundle.sumFin_add]
    apply sumFin_congr_c
    intro j
    by_cases hj : j = i <;> simp [hj] <;> omega
  have h2Quad : 2 * matQuad n E v = 2 * diagSum + pairSum := by
    omega
  have hDiagNonneg : 0 ≤ diagSum := by
    apply Flow.GaugeBundle.sumFin_nonneg
    intro i
    have hd : 0 ≤ E i i - offDiag i := by have := hDom i; grind
    exact Int.mul_nonneg hd (int_sq_nonneg_c (v i))
  have hPairNonneg : 0 ≤ pairSum := by
    apply Flow.GaugeBundle.sumFin_nonneg
    intro i
    apply Flow.GaugeBundle.sumFin_nonneg
    intro j
    by_cases hj : j = i
    · simp [hj]
    · simp only [hj, ↓reduceIte]
      exact sdd_pair_nonneg (E i j) (v i) (v j)
  omega

private theorem gramLLt_psd_c (n : Nat) (L : Fin n → Fin n → Int) (v : Fin n → Int) :
    0 ≤ matQuad n (gramLLt n L) v := by
  have hEq : matQuad n (gramLLt n L) v =
      sumFin n (fun k => (sumFin n (fun i => v i * L i k)) * (sumFin n (fun j => v j * L j k))) := by
    unfold matQuad gramLLt
    calc
      sumFin n (fun i => sumFin n (fun j => v i * sumFin n (fun k => L i k * L j k) * v j))
          = sumFin n (fun i => sumFin n (fun j => sumFin n (fun k => (v i * L i k) * (v j * L j k)))) := by
            apply sumFin_congr_c; intro i
            apply sumFin_congr_c; intro j
            rw [← Flow.GaugeBundle.sumFin_mul_left, ← sumFin_mul_right_c]
            apply sumFin_congr_c; intro k; grind
      _ = sumFin n (fun i => sumFin n (fun k => sumFin n (fun j => (v i * L i k) * (v j * L j k)))) := by
            apply sumFin_congr_c; intro i; exact sumFin_swap_c _
      _ = sumFin n (fun k => sumFin n (fun i => sumFin n (fun j => (v i * L i k) * (v j * L j k)))) :=
            sumFin_swap_c _
      _ = sumFin n (fun k => sumFin n (fun i => (v i * L i k) * sumFin n (fun j => v j * L j k))) := by
            apply sumFin_congr_c; intro k
            apply sumFin_congr_c; intro i
            exact Flow.GaugeBundle.sumFin_mul_left n (v i * L i k) (fun j => v j * L j k)
      _ = sumFin n (fun k => sumFin n (fun i => v i * L i k) * sumFin n (fun j => v j * L j k)) := by
            apply sumFin_congr_c; intro k
            exact sumFin_mul_right_c (fun i => v i * L i k) _
  rw [hEq]
  apply Flow.GaugeBundle.sumFin_nonneg
  intro k
  exact int_sq_nonneg_c _

/-- Theorem C10 (Integer Cholesky + SDD Certified Eigenvalue Lower Bound):
    If `E = A - mu*I - L*L^T` is symmetric diagonally dominant, then `mu` is an exact PSD floor of `A`. -/
theorem cholesky_sdd_certified_psd (n : Nat) (A L : Fin n → Fin n → Int) (mu : Int)
    (hSDD : IsSDD n (fun i j => A i j - (if i = j then mu else 0) - gramLLt n L i j)) :
    ∀ v : Fin n → Int, mu * vecNormSq n v ≤ matQuad n A v := by
  intro v
  let E : Fin n → Fin n → Int := fun i j => A i j - (if i = j then mu else 0) - gramLLt n L i j
  have hNonneg : 0 ≤ matQuad n E v := sdd_quadratic_form_nonneg n E hSDD v
  have hGram : 0 ≤ matQuad n (gramLLt n L) v := gramLLt_psd_c n L v
  have hDecomp : matQuad n E v = matQuad n A v - mu * vecNormSq n v - matQuad n (gramLLt n L) v := by
    have hD : sumFin n (fun i => sumFin n (fun j => v i * (if i = j then mu else 0) * v j)) = mu * vecNormSq n v := by
      unfold vecNormSq
      rw [← Flow.GaugeBundle.sumFin_mul_left]
      apply sumFin_congr_c
      intro i
      have hSub : (fun j : Fin n => v i * (if i = j then mu else 0) * v j) =
          (fun j : Fin n => if j = i then mu * (v i * v i) else 0) := by
        funext j
        by_cases hj : j = i
        · subst hj; simp; grind
        · have hij : i ≠ j := fun h => hj h.symm
          simp [hj, hij]
      rw [hSub, Flow.GaugeBundle.sumFin_single_hit]
    unfold matQuad E
    calc
      sumFin n (fun i => sumFin n (fun j => v i * (A i j - (if i = j then mu else 0) - gramLLt n L i j) * v j))
          = sumFin n (fun i => sumFin n (fun j =>
              (v i * A i j * v j - v i * (if i = j then mu else 0) * v j) - v i * gramLLt n L i j * v j)) := by
            apply sumFin_congr_c; intro i
            apply sumFin_congr_c; intro j; grind
      _ = sumFin n (fun i => sumFin n (fun j => v i * A i j * v j) -
              sumFin n (fun j => v i * (if i = j then mu else 0) * v j) -
              sumFin n (fun j => v i * gramLLt n L i j * v j)) := by
            apply sumFin_congr_c; intro i
            rw [Flow.GaugeBundle.sumFin_sub, Flow.GaugeBundle.sumFin_sub]
      _ = matQuad n A v -
            sumFin n (fun i => sumFin n (fun j => v i * (if i = j then mu else 0) * v j)) -
            matQuad n (gramLLt n L) v := by
            rw [Flow.GaugeBundle.sumFin_sub, Flow.GaugeBundle.sumFin_sub]; rfl
      _ = matQuad n A v - mu * vecNormSq n v - matQuad n (gramLLt n L) v := by
            rw [hD]
  omega

def matAt {n : Nat} (rows : List (List Int)) (i j : Fin n) : Int :=
  (rows.getD i.val []).getD j.val 0

def triAt {n : Nat} (rows : List (List Int)) (i k : Fin n) : Int :=
  if k.val ≤ i.val then matAt rows i k else 0

private def allFinAux (n : Nat) (p : Fin n → Bool) : (k : Nat) → k ≤ n → Bool
  | 0, _ => true
  | k + 1, hk => allFinAux n p k (Nat.le_of_lt hk) && p ⟨k, hk⟩

def allFinB (n : Nat) (p : Fin n → Bool) : Bool :=
  allFinAux n p n (Nat.le_refl n)

private theorem allFinAux_true (n : Nat) (p : Fin n → Bool) :
    ∀ k (hk : k ≤ n), allFinAux n p k hk = true → ∀ j (hj : j < k), p ⟨j, Nat.lt_of_lt_of_le hj hk⟩ = true
  | 0, _, _, _, hj => by omega
  | k + 1, hk, hall, j, hj => by
    simp only [allFinAux, Bool.and_eq_true] at hall
    by_cases hjk : j = k
    · subst hjk; exact hall.2
    · have hjlt : j < k := by omega
      exact allFinAux_true n p k (Nat.le_of_lt hk) hall.1 j hjlt

theorem allFinB_true (n : Nat) (p : Fin n → Bool) (h : allFinB n p = true) :
    ∀ i : Fin n, p i = true := by
  intro i
  exact allFinAux_true n p n (Nat.le_refl n) h i.val i.isLt

def checkCholeskySDD (n : Nat) (A L : List (List Int)) (mu : Int) : Bool :=
  let E : Fin n → Fin n → Int := fun i j =>
    matAt A i j - (if i = j then mu else 0) - gramLLt n (triAt L) i j
  let symOk := allFinB n (fun i => allFinB n (fun j => decide (matAt A i j = matAt A j i)))
  let domOk := allFinB n (fun i =>
    decide (sumFin n (fun j => if j = i then 0 else absInt (E i j)) ≤ E i i))
  symOk && domOk

/-- Kernel-verifiable soundness for arbitrary block size `n` using row-major `List (List Int)`. -/
theorem checkCholeskySDD_sound (n : Nat) (A L : List (List Int)) (mu : Int)
    (hCheck : checkCholeskySDD n A L mu = true) :
    ∀ v : Fin n → Int, mu * vecNormSq n v ≤ matQuad n (matAt A) v := by
  unfold checkCholeskySDD at hCheck
  simp only [Bool.and_eq_true] at hCheck
  have hSymmMat : ∀ i j : Fin n, matAt A i j = matAt A j i := by
    intro i j
    have hi := allFinB_true n _ hCheck.1 i
    have hj := allFinB_true n _ hi j
    exact of_decide_eq_true hj
  have hDomE : ∀ i : Fin n,
      sumFin n (fun j => if j = i then 0 else absInt (matAt A i j - (if i = j then mu else 0) - gramLLt n (triAt L) i j)) ≤
        matAt A i i - (if i = i then mu else 0) - gramLLt n (triAt L) i i := by
    intro i
    simpa using of_decide_eq_true (allFinB_true n _ hCheck.2 i)
  refine cholesky_sdd_certified_psd n (matAt A) (triAt L) mu ⟨?_, hDomE⟩
  intro i j
  have hg : gramLLt n (triAt L) i j = gramLLt n (triAt L) j i := by
    unfold gramLLt
    apply sumFin_congr_c; intro k; exact Int.mul_comm _ _
  by_cases hij : i = j
  · subst hij; rfl
  · have hji : j ≠ i := fun h => hij h.symm
    simp only [hij, hji, ↓reduceIte, hSymmMat i j, hg]

/-- Theorem C11 (Exact Borland–Dennis Generalized Pauli Constraint for `(3e,3o)`):
    Pure `(2α, 1β)` states in 6 spin-orbitals with SVD weights `p1 ≥ p2 ≥ p3 ≥ 0`,
    `p1 + p2 + p3 = W` always satisfy pairing + `0 ≤ l5 + l6 - l4`, with exact pinning iff `2 * p1 ≥ W`. -/
theorem borland_dennis_3e3o_exact (W p1 p2 p3 : Int)
    (hp3 : 0 ≤ p3) (hp32 : p3 ≤ p2) (hp21 : p2 ≤ p1) (hsum : p1 + p2 + p3 = W) :
    let l1 := W - p3
    let l2 := W - p2
    let l3 := if 2 * p1 ≤ W then W - p1 else p1
    let l4 := if 2 * p1 ≤ W then p1 else W - p1
    let l5 := p2
    let l6 := p3
    (l6 ≤ l5 ∧ l5 ≤ l4 ∧ l4 ≤ l3 ∧ l3 ≤ l2 ∧ l2 ≤ l1) ∧
    (l1 + l6 = W ∧ l2 + l5 = W ∧ l3 + l4 = W) ∧
    0 ≤ l5 + l6 - l4 ∧
    (W ≤ 2 * p1 ↔ l5 + l6 - l4 = 0) := by
  by_cases hcmp : 2 * p1 ≤ W
  · simp only [hcmp, ↓reduceIte]
    refine ⟨⟨hp32, ?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_⟩, ?_, ?_⟩ <;> omega
  · simp only [hcmp, ↓reduceIte]
    refine ⟨⟨hp32, ?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_⟩, ?_, ?_⟩ <;> omega

/-- Theorem C12 (Exact Borland–Dennis / Löwdin Pairing for `(2e,2o)` Singlet):
    2 electrons in 4 spin-orbitals satisfy exact occupation pairing `l1 + l4 = W` and `l2 + l3 = W`. -/
theorem borland_dennis_2e2o_exact (W p1 : Int) (_hp0 : 0 ≤ p1) (_hup : p1 ≤ W) (hmaj : W ≤ 2 * p1) :
    let l1 := p1
    let l2 := p1
    let l3 := W - p1
    let l4 := W - p1
    l4 ≤ l3 ∧ l3 ≤ l2 ∧ l2 ≤ l1 ∧ l1 + l4 = W ∧ l2 + l3 = W := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> omega

/-- Theorem C13 (Certified Quantum Phase Transition Order-Parameter Separation):
    Disjoint certified intervals `[loA, hiA]` and `[loB, hiB]` prove `O_A < O_B`, exact difference
    enclosure, and strict sign reversal when `hiA < 0 < loB`. -/
theorem certified_phase_transition_separation (ivA ivB : Interval1D) (O_A O_B : Int)
    (hA : ivA.contains O_A) (hB : ivB.contains O_B) (hSep : ivA.hi < ivB.lo) :
    O_A < O_B ∧
    ({ lo := ivB.lo - ivA.hi, hi := ivB.hi - ivA.lo } : Interval1D).contains (O_B - O_A) ∧
    ((ivA.hi < 0 ∧ 0 < ivB.lo) → (O_A < 0 ∧ 0 < O_B)) := by
  unfold Interval1D.contains at *
  dsimp
  refine ⟨by omega, ⟨by omega, by omega⟩, fun ⟨h1, h2⟩ => ⟨by omega, by omega⟩⟩

/-- Theorem C14 (Eckart Certified Ground-State Wavefunction Fidelity Lower Bound):
    Given `0 ≤ f ≤ W`, `E_lb ≤ E_0 ≤ E_ub < E1_lb`, and orthogonal decomposition
    `W * E_ub = f * E_0 + (W - f) * E_perp` with `E1_lb ≤ E_perp`,
    the overlap deficiency satisfies `(W - f) * (E1_lb - E_lb) ≤ W * (E_ub - E_lb)`. -/
theorem eckart_certified_fidelity_bound
    (W f E_lb E_0 E_ub E1_lb E_perp : Int)
    (hf0 : 0 ≤ f) (hfW : f ≤ W)
    (hE0 : E_lb ≤ E_0) (hEperp : E1_lb ≤ E_perp)
    (hDecomp : W * E_ub = f * E_0 + (W - f) * E_perp) :
    (W - f) * (E1_lb - E_lb) ≤ W * (E_ub - E_lb) := by
  have h1 : f * E_lb ≤ f * E_0 := Int.mul_le_mul_of_nonneg_left hE0 hf0
  have hwm : 0 ≤ W - f := by omega
  have h2 : (W - f) * E1_lb ≤ (W - f) * E_perp := Int.mul_le_mul_of_nonneg_left hEperp hwm
  have hExp : (W - f) * (E1_lb - E_lb) = (W - f) * E1_lb + f * E_lb - W * E_lb := by grind
  have hRhs : W * (E_ub - E_lb) = W * E_ub - W * E_lb := by grind
  rw [hExp, hRhs]
  omega

/-! ## Frontier 1 & Frontier 2 Theorems: Energy–Observable–Gap Decoupling & T1 = D3 + Q3 -/

/-- Theorem C15a (Energy–Observable–Gap Decoupling via 2-Level PSD Density Superpositions):
    In a two-level eigenstate subspace `{|ψ0⟩, |ψ1⟩}` with ground energy `E0 ≤ E1` and
    observable off-diagonal coupling `m = ⟨ψ0|O|ψ1⟩`, any scaled 2×2 density matrices
    `N * ρ_± = [[f, ±c], [±c, e]]` (`N = f + e`, `0 ≤ f`, `0 ≤ e`, PSD `c * c ≤ f * e`)
    fitting inside the energy ceiling `e * (E1 - E0) ≤ (f + e) * (E_ub - E0)` satisfy:
    (1) `f * E0 + e * E1 ≤ (f + e) * E_ub` (exact variational energy feasibility), and
    (2) `4 * c * m ≤ (f + e) * (O_ub - O_lb)` for any certified outer envelope `[O_lb, O_ub]`
        covering both `ρ_+` and `ρ_-`. -/
theorem observable_gap_decoupling_feasibility
    (f e c E0 E1 E_ub O00 O11 m O_lb O_ub : Int)
    (hBudget : e * (E1 - E0) ≤ (f + e) * (E_ub - E0))
    (hLo : (f + e) * O_lb ≤ f * O00 - 2 * c * m + e * O11)
    (hHi : f * O00 + 2 * c * m + e * O11 ≤ (f + e) * O_ub) :
    f * E0 + e * E1 ≤ (f + e) * E_ub ∧
    4 * c * m ≤ (f + e) * (O_ub - O_lb) := by
  constructor
  · have hE : f * E0 + e * E1 - (f + e) * E_ub = e * (E1 - E0) - (f + e) * (E_ub - E0) := by
      grind
    omega
  · have hW : (f + e) * (O_ub - O_lb) = (f + e) * O_ub - (f + e) * O_lb := by grind
    have h4 : 4 * c * m = 2 * (2 * c * m) := by grind
    omega

/-- Theorem C15b (Coherence–Gap Δ^{-1/2} Scaling & Observable Width Lower Bound):
    Given a valid 2-level PSD density matrix (`0 ≤ f`, `0 ≤ e`, `c * c ≤ f * e`) within
    spectral-gap energy budget `e * Δ ≤ (f + e) * δE` (`0 ≤ Δ`) and non-negative envelope
    width bound `4 * c * m ≤ (f + e) * W` (`0 ≤ c`, `0 ≤ m`, `0 ≤ W`, `0 ≤ f + e`), both:
    (1) `c * c * Δ ≤ (f + e) * (f + e) * δE` (coherence scales as `√(δE / Δ) ∝ Δ^{-1/2}`), and
    (2) `16 * (c * c) * (m * m) ≤ ((f + e) * W) * ((f + e) * W)` (forcing observable width
        blowup whenever transition coupling `m ≠ 0` at avoided crossings). -/
theorem certified_observable_gap_lower_bound
    (f e c Δ δE m W : Int)
    (hf : 0 ≤ f) (he : 0 ≤ e) (hc : 0 ≤ c) (hΔ : 0 ≤ Δ) (hm : 0 ≤ m) (hW : 0 ≤ W)
    (hPSD : c * c ≤ f * e)
    (hBudget : e * Δ ≤ (f + e) * δE)
    (hSep : 4 * c * m ≤ (f + e) * W) :
    c * c * Δ ≤ (f + e) * (f + e) * δE ∧
    16 * (c * c) * (m * m) ≤ ((f + e) * W) * ((f + e) * W) := by
  constructor
  · have h1 : c * c * Δ ≤ f * e * Δ := Int.mul_le_mul_of_nonneg_right hPSD hΔ
    have hfeΔ : f * e * Δ = f * (e * Δ) := by grind
    have h2 : f * (e * Δ) ≤ f * ((f + e) * δE) := Int.mul_le_mul_of_nonneg_left hBudget hf
    have hfN : f ≤ f + e := by omega
    have hδE_nonneg : 0 ≤ (f + e) * δE := by
      have heΔ : 0 ≤ e * Δ := Int.mul_nonneg he hΔ
      omega
    have h3 : f * ((f + e) * δE) ≤ (f + e) * ((f + e) * δE) :=
      Int.mul_le_mul_of_nonneg_right hfN hδE_nonneg
    have h4 : (f + e) * ((f + e) * δE) = (f + e) * (f + e) * δE := by grind
    omega
  · have hcm : 0 ≤ 4 * c * m := by
      have h4c : 0 ≤ 4 * c := by omega
      exact Int.mul_nonneg h4c hm
    have hsq1 : (4 * c * m) * (4 * c * m) ≤ (4 * c * m) * ((f + e) * W) :=
      Int.mul_le_mul_of_nonneg_left hSep hcm
    have hNW : 0 ≤ (f + e) * W := Int.mul_nonneg (by omega) hW
    have hsq2 : (4 * c * m) * ((f + e) * W) ≤ ((f + e) * W) * ((f + e) * W) :=
      Int.mul_le_mul_of_nonneg_right hSep hNW
    have hlhs : (4 * c * m) * (4 * c * m) = 16 * (c * c) * (m * m) := by grind
    omega

/-- Theorem C16 (Subspace `T1 = D3 + Q3` Composite Trace Upper Bound for Weyl Shift):
    In any projected orbital subspace `S` of triplet basis size `dim_sub`, the composite
    3-positivity matrix `T1 = D3 + Q3` has nonnegative principal-submatrix traces bounded by
    full-space `tr_d3_full + tr_q3_full` and unit-occupancy diagonal floor `dim_sub`.
    Therefore when certified eigenvalue floor `lam_cert ≤ 0`, substituting
    `min dim_sub (tr_d3_full + tr_q3_full)` produces an unconditionally valid lower bound. -/
theorem subspace_t1_trace_upper_bound
    (tr_d3_sub tr_q3_sub tr_d3_full tr_q3_full dim_sub lam_cert : Int)
    (hD3 : tr_d3_sub ≤ tr_d3_full)
    (hQ3 : tr_q3_sub ≤ tr_q3_full)
    (hSubLeDim : tr_d3_sub + tr_q3_sub ≤ dim_sub)
    (hLam : lam_cert ≤ 0) :
    min dim_sub (tr_d3_full + tr_q3_full) * lam_cert ≤
      (tr_d3_sub + tr_q3_sub) * lam_cert := by
  have hmin : tr_d3_sub + tr_q3_sub ≤ min dim_sub (tr_d3_full + tr_q3_full) := by
    rw [Int.min_def]
    split <;> omega
  exact Int.mul_le_mul_of_nonpos_right hmin hLam

/-- Theorem C17 (Physical Coulomb PSD Manifold Gram/Cholesky Positivity):
    For any density-fitting / Cholesky ERI factor `B : Fin R → Fin M → Int` (`V = B^T B`),
    the physical 2-electron Coulomb quadratic form equals a sum of squares over auxiliary rank `R`
    and is unconditionally nonnegative (`0 ≤ x^T V x`). -/
theorem coulomb_psd_cholesky_nonneg
    {R M : Nat} (B : Fin R → Fin M → Int) (x : Fin M → Int) :
    0 ≤ sumFin M (fun a => sumFin M (fun b =>
          (sumFin R (fun mu => B mu a * B mu b)) * (x a * x b))) ∧
    sumFin M (fun a => sumFin M (fun b =>
          (sumFin R (fun mu => B mu a * B mu b)) * (x a * x b))) =
      sumFin R (fun mu => (sumFin M (fun a => B mu a * x a)) * (sumFin M (fun b => B mu b * x b))) := by
  have hEq :
      sumFin M (fun a => sumFin M (fun b => (sumFin R (fun mu => B mu a * B mu b)) * (x a * x b))) =
        sumFin R (fun mu =>
          (sumFin M (fun a => B mu a * x a)) * (sumFin M (fun b => B mu b * x b))) := by
    calc
      sumFin M (fun a => sumFin M (fun b => (sumFin R (fun mu => B mu a * B mu b)) * (x a * x b)))
          = sumFin M (fun a => sumFin M (fun b => sumFin R (fun mu => B mu a * B mu b * (x a * x b)))) := by
              apply sumFin_congr_c; intro a
              apply sumFin_congr_c; intro b
              exact (sumFin_mul_right_c (fun mu => B mu a * B mu b) (x a * x b)).symm
      _ = sumFin M (fun a => sumFin R (fun mu => sumFin M (fun b => B mu a * B mu b * (x a * x b)))) := by
            apply sumFin_congr_c; intro a
            exact sumFin_swap_c (fun b mu => B mu a * B mu b * (x a * x b))
      _ = sumFin R (fun mu => sumFin M (fun a => sumFin M (fun b => B mu a * B mu b * (x a * x b)))) := by
            exact sumFin_swap_c (fun a mu => sumFin M (fun b => B mu a * B mu b * (x a * x b)))
      _ = sumFin R (fun mu => sumFin M (fun a => sumFin M (fun b => (B mu a * x a) * (B mu b * x b)))) := by
            apply sumFin_congr_c; intro mu
            apply sumFin_congr_c; intro a
            apply sumFin_congr_c; intro b
            grind
      _ = sumFin R (fun mu => (sumFin M (fun a => B mu a * x a)) * (sumFin M (fun b => B mu b * x b))) := by
            apply sumFin_congr_c; intro mu
            have hin :
                sumFin M (fun a => sumFin M (fun b => (B mu a * x a) * (B mu b * x b))) =
                  sumFin M (fun a => (B mu a * x a) * sumFin M (fun b => B mu b * x b)) := by
              apply sumFin_congr_c; intro a
              exact Flow.GaugeBundle.sumFin_mul_left M (B mu a * x a) (fun b => B mu b * x b)
            rw [hin]
            exact sumFin_mul_right_c (fun a => B mu a * x a) (sumFin M (fun b => B mu b * x b))
  have hPos :
      0 ≤ sumFin R (fun mu =>
            (sumFin M (fun a => B mu a * x a)) * (sumFin M (fun b => B mu b * x b))) := by
    apply Flow.GaugeBundle.sumFin_nonneg
    intro mu
    exact int_sq_nonneg_c (sumFin M (fun a => B mu a * x a))
  rw [hEq]
  exact ⟨hPos, rfl⟩

#print axioms mode_a_spin_trace_invariants
#print axioms anytime_dual_weyl_lower_bound
#print axioms anytime_early_veto_soundness
#print axioms certified_spin_gap_ordering
#print axioms mode_b_spin_squared_identity
#print axioms chem_bracket_from_flow_molecular_bundle
#print axioms certified_observable_envelope_bounds
#print axioms subspace_t2_trace_upper_bound
#print axioms sdd_quadratic_form_nonneg
#print axioms cholesky_sdd_certified_psd
#print axioms checkCholeskySDD_sound
#print axioms borland_dennis_3e3o_exact
#print axioms borland_dennis_2e2o_exact
#print axioms certified_phase_transition_separation
#print axioms eckart_certified_fidelity_bound
#print axioms observable_gap_decoupling_feasibility
#print axioms certified_observable_gap_lower_bound
#print axioms subspace_t1_trace_upper_bound
#print axioms coulomb_psd_cholesky_nonneg

end ChemCert
