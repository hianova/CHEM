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

#print axioms mode_a_spin_trace_invariants
#print axioms anytime_dual_weyl_lower_bound
#print axioms anytime_early_veto_soundness
#print axioms certified_spin_gap_ordering
#print axioms mode_b_spin_squared_identity
#print axioms chem_bracket_from_flow_molecular_bundle
#print axioms certified_observable_envelope_bounds
#print axioms subspace_t2_trace_upper_bound

end ChemCert
