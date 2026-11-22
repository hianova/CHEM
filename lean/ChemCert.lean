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
import Flow.CholeskySDD
import Flow.MolecularBundle
import Flow.Interval

namespace ChemCert

open Flow.Interval
open Flow.MolecularBundle
open Flow.GaugeBundle

export Flow.FiniteSum (absInt absInt_nonneg int_sq_nonneg sumFin sumFin_congr sumFin_zero
  sumFin_nonneg sumFin_add sumFin_sub sumFin_neg sumFin_mul_left sumFin_mul_right
  sumFin_swap sumFin_single_hit sumFin_le_sumFin)
export Flow.CholeskySDD (vecNormSq vecNormSq_nonneg matQuad gramLLt IsSDD
  sdd_quadratic_form_nonneg gramLLt_psd cholesky_sdd_certified_psd matAt triAt
  allFinB allFinB_true checkCholeskySDD checkCholeskySDD_sound)

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

/-! ## v0.3.0 Formal Foundations: Zero-FP-Trust Cholesky+SDD (`Flow.CholeskySDD`), GPC Borland–Dennis, Phase Transitions & Eckart Fidelity -/

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
              apply Flow.FiniteSum.sumFin_congr; intro a
              apply Flow.FiniteSum.sumFin_congr; intro b
              exact (Flow.FiniteSum.sumFin_mul_right R (fun mu => B mu a * B mu b) (x a * x b)).symm
      _ = sumFin M (fun a => sumFin R (fun mu => sumFin M (fun b => B mu a * B mu b * (x a * x b)))) := by
            apply Flow.FiniteSum.sumFin_congr; intro a
            exact Flow.FiniteSum.sumFin_swap (fun b mu => B mu a * B mu b * (x a * x b))
      _ = sumFin R (fun mu => sumFin M (fun a => sumFin M (fun b => B mu a * B mu b * (x a * x b)))) := by
            exact Flow.FiniteSum.sumFin_swap (fun a mu => sumFin M (fun b => B mu a * B mu b * (x a * x b)))
      _ = sumFin R (fun mu => sumFin M (fun a => sumFin M (fun b => (B mu a * x a) * (B mu b * x b)))) := by
            apply Flow.FiniteSum.sumFin_congr; intro mu
            apply Flow.FiniteSum.sumFin_congr; intro a
            apply Flow.FiniteSum.sumFin_congr; intro b
            grind
      _ = sumFin R (fun mu => (sumFin M (fun a => B mu a * x a)) * (sumFin M (fun b => B mu b * x b))) := by
            apply Flow.FiniteSum.sumFin_congr; intro mu
            have hin :
                sumFin M (fun a => sumFin M (fun b => (B mu a * x a) * (B mu b * x b))) =
                  sumFin M (fun a => (B mu a * x a) * sumFin M (fun b => B mu b * x b)) := by
              apply Flow.FiniteSum.sumFin_congr; intro a
              exact Flow.FiniteSum.sumFin_mul_left M (B mu a * x a) (fun b => B mu b * x b)
            rw [hin]
            exact Flow.FiniteSum.sumFin_mul_right M (fun a => B mu a * x a) (sumFin M (fun b => B mu b * x b))
  have hPos :
      0 ≤ sumFin R (fun mu =>
            (sumFin M (fun a => B mu a * x a)) * (sumFin M (fun b => B mu b * x b))) := by
    apply Flow.FiniteSum.sumFin_nonneg
    intro mu
    exact Flow.FiniteSum.int_sq_nonneg (sumFin M (fun a => B mu a * x a))
  rw [hEq]
  exact ⟨hPos, rfl⟩

/-- **Theorem C18 (Rank-1 Coulomb Integrality & Pair-Block Energy Identity).**
For the zero-$h_1$ rank-1 Coulomb model ($\beta = (-1, -1, 2, 2)$, $K=4, N=4$),
every $(n_A, n_B)$ charge partition with $n_A + n_B = 4$ satisfies the exact
quadratic energy law $8 H(n_A) = 9(2 n_A - 5)^2 - 33$. Because $2 n_A - 5$ is
odd for every integer $n_A$, $(2 n_A - 5)^2 \ge 1$ forces $8 H(n_A) \ge -24$
(attained at $n_A \in \{2, 3\}$), whereas the unquantized relaxation at
$n_A = 5/2$ attains $-33$, creating an exact $9$-unit ($112.5\text{ mHa/unit-Frob}$)
DQG/T1/T2 integrality gap. -/
theorem rank1_coulomb_integrality_gap_exact :
    (∀ nA nB : Int, nA + nB = 4 →
      4 * ((-nA + 2 * nB) * (-nA + 2 * nB) - (nA + 4 * nB)) =
        9 * (2 * nA - 5) * (2 * nA - 5) - 33) ∧
    (∀ nA : Int, -24 ≤ 9 * (2 * nA - 5) * (2 * nA - 5) - 33) ∧
    (9 * (2 * (2 : Int) - 5) * (2 * (2 : Int) - 5) - 33 = -24 ∧
     (-24 : Int) - (-33) = 9) := by
  refine ⟨?_, ?_, ?_⟩
  · intro nA nB hsum
    have hnB : nB = 4 - nA := by omega
    subst hnB
    have hs1 : -nA + 2 * (4 - nA) = 8 - 3 * nA := by omega
    rw [hs1, Int.mul_assoc 9 (2 * nA - 5) (2 * nA - 5)]
    have hna3 : nA * (3 * nA) = 3 * (nA * nA) := by
      rw [← Int.mul_assoc, Int.mul_comm nA 3, Int.mul_assoc]
    have hna2 : nA * (2 * nA) = 2 * (nA * nA) := by
      rw [← Int.mul_assoc, Int.mul_comm nA 2, Int.mul_assoc]
    have hsq1 : (8 - 3 * nA) * (8 - 3 * nA) = 64 - 48 * nA + 9 * (nA * nA) := by
      rw [Int.sub_mul, Int.mul_sub, Int.mul_sub, Int.mul_assoc 3 nA (3 * nA), hna3, ← Int.mul_assoc 3 3 (nA * nA)]
      omega
    have hsq2 : (2 * nA - 5) * (2 * nA - 5) = 4 * (nA * nA) - 20 * nA + 25 := by
      rw [Int.sub_mul, Int.mul_sub, Int.mul_sub, Int.mul_assoc 2 nA (2 * nA), hna2, ← Int.mul_assoc 2 2 (nA * nA)]
      omega
    rw [hsq1, hsq2]
    omega
  · intro nA
    rw [Int.mul_assoc 9 (2 * nA - 5) (2 * nA - 5)]
    let d : Int := 2 * nA - 5
    have hsq : 1 ≤ d * d := by
      by_cases hpos : 1 ≤ d
      · exact Int.mul_le_mul hpos hpos (by decide) (by omega)
      · have hneg : 1 ≤ -d := by omega
        have h1 := Int.mul_le_mul hneg hneg (by decide) (by omega)
        have hsqneg : (-d) * (-d) = d * d := Int.neg_mul_neg d d
        rw [hsqneg] at h1
        exact h1
    change -24 ≤ 9 * (d * d) - 33
    omega
  · decide

/-- **Theorem C19 (Full 3-POS Strictly Separates DQG and FCI on Rank-1 Coulomb).**
Scaling moments by $10^3$ ($m_1 = 5/2$, $m_{2,s} = 1000\, m_2$, $m_{3,s} = 1000\, m_3$):
1. *(DQG < 3-POS)* Any moment triple satisfying the 3-body localizing PSD inequality
   $F_3 : 2500\, m_{3,s} \ge m_{2,s}^2$ and the 3-hole inequality
   $Q_3 : m_{3,s} \le 9\, m_{2,s} - 41000$ forces $m_{2,s} \ge 6345$
   (near the exact algebraic bound $1000(45 - \sqrt{385})/4 \approx 6344.65$),
   strictly ruling out the DQG minimum $m_{2,s} = 6250$.
2. *(3-POS < FCI)* The explicit rational point $(m_{2,s}, m_{3,s}) = (6350, 16130)$
   satisfies all six 3-POS positivity conditions ($D_3, Q_3, \det F_3, \det E_3,
   E_3^{\text{mix}}, F_3^{\text{mix}} \ge 0$) while still strictly undershooting
   the exact FCI second moment $m_{2,\text{FCI},s} = 6500$ ($8 E_{3\text{-POS}} = -29.4 < -24 = 8 E_{\text{FCI}}$). -/
theorem rank1_coulomb_3pos_strict_separation :
    (∀ m2_s m3_s : Int,
      m2_s * m2_s ≤ 2500 * m3_s →
      m3_s ≤ 9 * m2_s - 41000 →
      6345 ≤ m2_s) ∧
    (∃ m2_s m3_s : Int,
      6250 < m2_s ∧ m2_s < 6500 ∧
      m2_s * m2_s ≤ 2500 * m3_s ∧
      m3_s ≤ 9 * m2_s - 41000 ∧
      0 ≤ m3_s - 3 * m2_s + 5000 ∧
      0 ≤ 24000 - 65000 + 9 * m2_s - m3_s ∧
      (10000 - m2_s) * (10000 - m2_s) ≤ 1500 * (4 * m2_s - m3_s) ∧
      0 ≤ 30000 - 7 * m2_s + m3_s ∧
      0 ≤ 5 * m2_s - 10000 - m3_s) := by
  constructor
  · intro m2_s m3_s hF3 hQ3
    have hquad : m2_s * m2_s ≤ 22500 * m2_s - 102500000 := by omega
    by_cases hge : 6345 ≤ m2_s
    · exact hge
    · have hmle : m2_s ≤ 6344 := by omega
      let u : Int := 11250 - m2_s
      have hu_ge : 4906 ≤ u := by omega
      have husq_ge : 4906 * 4906 ≤ u * u :=
        Int.mul_le_mul hu_ge hu_ge (by decide) (by omega)
      have husq_id : u * u = m2_s * m2_s - 22500 * m2_s + 126562500 := by
        change (11250 - m2_s) * (11250 - m2_s) = _
        rw [Int.sub_mul, Int.mul_sub, Int.mul_sub, Int.mul_comm m2_s 11250]
        omega
      omega
  · exact ⟨6350, 16130, by decide⟩

/-- Scaled falling factorial $\prod_{j=0}^{k-1} (x - 2j)$ representing $2^k \prod_{j=0}^{k-1}(x/2 - j)$. -/
def fall2 (x : Int) : Nat → Int
  | 0 => 1
  | k + 1 => fall2 x k * (x - 2 * (k : Int))

theorem fall2_pos (x : Int) (k : Nat) (h : 2 * (k : Int) ≤ x + 1) :
    0 < fall2 x k := by
  induction k with
  | zero => simp [fall2]
  | succ k ih =>
    have hk : 2 * (k : Int) ≤ x + 1 := by omega
    have hlast : 0 < x - 2 * (k : Int) := by omega
    exact Int.mul_pos (ih hk) hlast

/-- **Theorem C20 (Universal Rank-1 Coulomb PSD Obstruction of Arbitrary $p$-POS).**
On $K = 2p$ spatial orbitals ($M = 2p$ spin-orbitals per cluster, $N = 2p$ electrons)
with pure rank-1 Coulomb ERI $\beta = (-(p-1),\dots,-(p-1), p,\dots,p)$ ($h_1 = 0$):
1. *(Pair-Block Law)* Every integer charge partition $(n_A, n_B)$ with $n_A + n_B = 2p$ satisfies
   $8 H(n_A) = (2p-1)^2 (2n_A - (2p+1))^2 - (2p-1)(4p^2 - 2p - 1)$.
2. *(Exact Integer Gap)* Since $2n_A - (2p+1)$ is odd for all $n_A \in \mathbb{Z}$, the FCI ground
   energy is locked at $n_A \in \{p, p+1\}$, exceeding the DQG relaxed energy at $c = p + 1/2$
   by $(2p-1)^2$ in $8H$-units.
3. *(Universal $p$-POS Witness)* At $M = 2p$, $2c = 2p+1$ (matching `K=4,p=2` at $c=5/2$ and
   `K=6,p=3` at $c=7/2$), the $2^k$-scaled hypergeometric expectations are strictly positive for
   ALL $(m_1, m_0)$ with $m_1, m_0 \le p$, preserve exact 1-orbital marginal consistency, and
   annihilate $\text{Var}(n_A)$ across every $p$-body block (0% tightening from $p$-POS).
4. *(Sharp $(p+1)$-POS Breakdown)* At order $p+1$, $Q_{p+1}$ (`m0 = p+1`) turns strictly negative
   because $M - c = p - 1/2 < p$. -/
theorem rank1_coulomb_p_pos_hierarchy_obstruction :
    (∀ p nA nB : Int, nA + nB = 2 * p →
      4 * (((-(p - 1)) * nA + p * nB) * ((-(p - 1)) * nA + p * nB)
           - (((p - 1) * (p - 1)) * nA + (p * p) * nB)) =
        ((2 * p - 1) * (2 * p - 1)) * ((2 * nA - (2 * p + 1)) * (2 * nA - (2 * p + 1)))
        - (2 * p - 1) * (4 * (p * p) - 2 * p - 1)) ∧
    (∀ p nA : Int,
      ((2 * p - 1) * (2 * p - 1)) - (2 * p - 1) * (4 * (p * p) - 2 * p - 1) ≤
        ((2 * p - 1) * (2 * p - 1)) * ((2 * nA - (2 * p + 1)) * (2 * nA - (2 * p + 1)))
        - (2 * p - 1) * (4 * (p * p) - 2 * p - 1)) ∧
    (∀ (p m1 m0 : Nat), m1 ≤ p → m0 ≤ p →
      0 < fall2 (2 * (p : Int) + 1) m1 ∧
      0 < fall2 (2 * (p : Int) - 1) m0 ∧
      0 < fall2 (4 * (p : Int)) (m1 + m0) ∧
      (2 * (p : Int) + 1 - 2 * (m1 : Int)) + (2 * (p : Int) - 1 - 2 * (m0 : Int)) =
        2 * (2 * (p : Int) - ((m1 : Int) + (m0 : Int))) ∧
      2 * (m1 : Int) * (2 * (p : Int) - ((m1 : Int) + (m0 : Int))) +
        (2 * (p : Int) - ((m1 : Int) + (m0 : Int))) * (2 * (p : Int) + 1 - 2 * (m1 : Int)) =
        (2 * (p : Int) + 1) * (2 * (p : Int) - ((m1 : Int) + (m0 : Int)))) ∧
    (∀ p : Nat, fall2 (2 * (p : Int) - 1) (p + 1) < 0) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro p nA nB hsum
    have hnB : nB = 2 * p - nA := by omega
    subst hnB
    let D : Int := 2 * p - 1
    let X : Int := D * nA
    let P2 : Int := p * p
    let P3 : Int := p * P2
    have hX : X = 2 * (p * nA) - nA := by
      change (2 * p - 1) * nA = _
      rw [Int.sub_mul, Int.one_mul, Int.mul_assoc]
    have hneg1 : -(p - 1) = 1 - p := by omega
    have hlin : (-(p - 1)) * nA + p * (2 * p - nA) = 2 * P2 - X := by
      rw [hneg1, Int.sub_mul, Int.one_mul, Int.mul_sub, Int.mul_comm p (2 * p), Int.mul_assoc 2 p p]
      change nA - p * nA + (2 * P2 - p * nA) = 2 * P2 - X
      omega
    have hpm1sq : (p - 1) * (p - 1) = P2 - 2 * p + 1 := by
      rw [Int.sub_mul, Int.mul_sub, Int.mul_sub, Int.one_mul, Int.mul_one]
      change P2 - p - (p - 1) = P2 - 2 * p + 1
      omega
    have hquad_lin : ((p - 1) * (p - 1)) * nA + P2 * (2 * p - nA) = 2 * P3 - X := by
      rw [hpm1sq, Int.add_mul, Int.sub_mul, Int.one_mul, Int.mul_assoc 2 p nA,
          Int.mul_sub, Int.mul_comm P2 (2 * p), Int.mul_assoc 2 p P2]
      change P2 * nA - 2 * (p * nA) + nA + (2 * P3 - P2 * nA) = 2 * P3 - X
      omega
    have hDp : D * (2 * p + 1) = 4 * P2 - 1 := by
      change (2 * p - 1) * (2 * p + 1) = 4 * P2 - 1
      rw [Int.sub_mul, Int.mul_add, Int.mul_add, Int.one_mul, Int.mul_one,
          Int.mul_assoc 2 p (2 * p), Int.mul_comm p (2 * p), Int.mul_assoc 2 p p, ← Int.mul_assoc 2 2 P2]
      omega
    have hDX : D * (2 * nA - (2 * p + 1)) = 2 * X - (4 * P2 - 1) := by
      rw [Int.mul_sub, ← hDp, Int.mul_comm D (2 * nA), Int.mul_assoc 2 nA D, Int.mul_comm nA D]
    have hDrhs : (D * D) * ((2 * nA - (2 * p + 1)) * (2 * nA - (2 * p + 1))) =
        (2 * X - (4 * P2 - 1)) * (2 * X - (4 * P2 - 1)) := by
      let K : Int := 2 * nA - (2 * p + 1)
      change (D * D) * (K * K) = _
      have hre : (D * D) * (K * K) = (D * K) * (D * K) := by
        rw [Int.mul_assoc D D (K * K), Int.mul_comm D (K * K), Int.mul_assoc K K D,
            ← Int.mul_assoc D K (K * D), Int.mul_comm K D]
      rw [hre]
      exact congrArg (fun z => z * z) hDX
    have hDrem : D * (4 * P2 - 2 * p - 1) = 8 * P3 - 8 * P2 + 1 := by
      change (2 * p - 1) * (4 * P2 - 2 * p - 1) = 8 * P3 - 8 * P2 + 1
      rw [Int.sub_mul, Int.mul_sub, Int.mul_sub, Int.one_mul,
          Int.mul_assoc 2 p (4 * P2), Int.mul_comm p (4 * P2), Int.mul_assoc 4 P2 p,
          Int.mul_comm P2 p, ← Int.mul_assoc 2 4 P3,
          Int.mul_assoc 2 p (2 * p), Int.mul_comm p (2 * p), Int.mul_assoc 2 p p, ← Int.mul_assoc 2 2 P2,
          Int.mul_one]
      omega
    have hLsq : (2 * P2 - X) * (2 * P2 - X) =
        4 * (P2 * P2) - 4 * (P2 * X) + X * X := by
      rw [Int.sub_mul, Int.mul_sub, Int.mul_sub,
          Int.mul_assoc 2 P2 (2 * P2), Int.mul_comm P2 (2 * P2), Int.mul_assoc 2 P2 P2, ← Int.mul_assoc 2 2 (P2 * P2),
          Int.mul_assoc 2 P2 X, Int.mul_comm X (2 * P2), Int.mul_assoc 2 P2 X]
      omega
    have hYsq : (4 * P2 - 1) * (4 * P2 - 1) = 16 * (P2 * P2) - 8 * P2 + 1 := by
      rw [Int.sub_mul, Int.mul_sub, Int.one_mul, Int.mul_one,
          Int.mul_assoc 4 P2 (4 * P2), Int.mul_comm P2 (4 * P2), Int.mul_assoc 4 P2 P2, ← Int.mul_assoc 4 4 (P2 * P2)]
      omega
    have hXY : (2 * X) * (4 * P2 - 1) = 8 * (P2 * X) - 2 * X := by
      rw [Int.mul_sub, Int.mul_one, Int.mul_assoc 2 X (4 * P2),
          Int.mul_comm X (4 * P2), Int.mul_assoc 4 P2 X, ← Int.mul_assoc 2 4 (P2 * X)]
      omega
    have hRsq : (2 * X - (4 * P2 - 1)) * (2 * X - (4 * P2 - 1)) =
        4 * (X * X) - 16 * (P2 * X) + 4 * X + 16 * (P2 * P2) - 8 * P2 + 1 := by
      let Y : Int := 4 * P2 - 1
      change (2 * X - Y) * (2 * X - Y) = _
      rw [Int.sub_mul, Int.mul_sub (2 * X) (2 * X) Y, Int.mul_sub Y (2 * X) Y, Int.mul_comm Y (2 * X),
          Int.mul_assoc 2 X (2 * X), Int.mul_comm X (2 * X), Int.mul_assoc 2 X X, ← Int.mul_assoc 2 2 (X * X)]
      change 2 * 2 * (X * X) - (2 * X) * (4 * P2 - 1) - ((2 * X) * (4 * P2 - 1) - (4 * P2 - 1) * (4 * P2 - 1)) = _
      rw [hXY, hYsq]
      omega
    change 4 * (((-(p - 1)) * nA + p * (2 * p - nA)) * ((-(p - 1)) * nA + p * (2 * p - nA))
             - (((p - 1) * (p - 1)) * nA + P2 * (2 * p - nA))) =
           (D * D) * ((2 * nA - (2 * p + 1)) * (2 * nA - (2 * p + 1))) - D * (4 * P2 - 2 * p - 1)
    rw [hlin, hquad_lin, hDrhs, hDrem, hLsq, hRsq]
    omega
  · intro p nA
    let d : Int := 2 * nA - (2 * p + 1)
    have hsq : 1 ≤ d * d := by
      by_cases hpos : 1 ≤ d
      · exact Int.mul_le_mul hpos hpos (by decide) (by omega)
      · have hneg : 1 ≤ -d := by omega
        have h1 := Int.mul_le_mul hneg hneg (by decide) (by omega)
        have hsqneg : (-d) * (-d) = d * d := Int.neg_mul_neg d d
        rw [hsqneg] at h1
        exact h1
    have hD2 : 0 ≤ (2 * p - 1) * (2 * p - 1) := int_sq_nonneg (2 * p - 1)
    have hmul := Int.mul_le_mul_of_nonneg_left hsq hD2
    rw [Int.mul_one] at hmul
    change (2 * p - 1) * (2 * p - 1) - (2 * p - 1) * (4 * (p * p) - 2 * p - 1) ≤
      (2 * p - 1) * (2 * p - 1) * (d * d) - (2 * p - 1) * (4 * (p * p) - 2 * p - 1)
    omega
  · intro p m1 m0 hm1 hm0
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · exact fall2_pos (2 * (p : Int) + 1) m1 (by omega)
    · exact fall2_pos (2 * (p : Int) - 1) m0 (by omega)
    · exact fall2_pos (4 * (p : Int)) (m1 + m0) (by omega)
    · omega
    · let T : Int := 2 * (p : Int) - ((m1 : Int) + (m0 : Int))
      change 2 * (m1 : Int) * T + T * (2 * (p : Int) + 1 - 2 * (m1 : Int)) =
        (2 * (p : Int) + 1) * T
      rw [Int.mul_comm (2 * (m1 : Int)) T, ← Int.mul_add]
      have hcoef : 2 * (m1 : Int) + (2 * (p : Int) + 1 - 2 * (m1 : Int)) = 2 * (p : Int) + 1 := by omega
      rw [hcoef, Int.mul_comm T (2 * (p : Int) + 1)]
  · intro p
    simp only [fall2]
    have hpos : 0 < fall2 (2 * (p : Int) - 1) p := fall2_pos (2 * (p : Int) - 1) p (by omega)
    have hneg : 2 * (p : Int) - 1 - 2 * (p : Int) = -1 := by omega
    rw [hneg]
    exact Int.mul_neg_of_pos_of_neg hpos (by decide)

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
#print axioms rank1_coulomb_integrality_gap_exact
#print axioms rank1_coulomb_3pos_strict_separation
#print axioms rank1_coulomb_p_pos_hierarchy_obstruction

end ChemCert
