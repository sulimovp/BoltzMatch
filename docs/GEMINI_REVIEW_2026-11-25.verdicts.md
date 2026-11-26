# Verdicts on the Gemini review of BoltzMatch

Each finding in `BoltzMatch.md` was checked against the repository on 2026-11-25 (read-only; no
project code run).

**Short version.** Three findings are new and real. First, the hidden bias `hb` is used in training
and in the Python score that picks the epoch, but it is never written to the exported `.ms2`, so
Crux scores without it (3.1); its gradient is larger than that of `W`, and the effect on the 2020
numbers is unknown. Second, skipped spectra stay as all-zero rows whose hidden activations equal
`hb`, so they dominate the divergence regulariser, and `abs(mean(...))` lets terms cancel (3.3).
Third, `docs/REVIEW.md` cites arXiv 2509.24952 (Winnow, a decoy-free FDR estimator) as an
entrapment evaluation, which it is not. Most other items (int16 overflow, broken stack, batching,
Tailor misalignment, wrong iPRG dataset) are already in `docs/REVIEW.md`, and the review rarely
says so. It labels almost everything Blocker and 100% certain: no direct decoy leak is shown (3.2),
the "handicapped Tide baseline" uses the same flags for both arms, and the MS Amanda setting limits
modification sites, not methionines. Several 2026 citations could not be confirmed.

| Finding | Review severity | Verdict | Evidence | Note |
|---|---|---|---|---|
| 3.1 `rbm_hb` not exported to `.ms2` | Blocker | CONFIRMED (mechanism) | `source/utils.py` L125-134 (`s@W` in `spectrum_array`, `hb` in `peptide.bias`); `dbsearch.py` L685-700 writes only `spectrum_array`; Python score adds the bias (L572); training uses it (`models.py` L63-69); `docs/verify_output.txt` L9 (`hb` gradient 1.73 vs `W` 1.64). | New. Impact unmeasured; Major is fairer. |
| 3.2 In-sample training and decoy-guided epoch selection | Blocker | PARTLY | `rbm_train_main.py` L176; `utils.py` L55, L69-71; Python decoys reverse peptides with fixed termini (`dbsearch.py` L386-390), Crux decoys reverse proteins. | In-sample design known (`docs/REVIEW.md` L142-152, M9). Epoch choice over 5-15 epochs by Python decoys cannot leak Crux decoys. Open risk, not proven. |
| 3.3 `div_reg` dominated by zero rows; `abs(mean)` cancels | Blocker | CONFIRMED (mechanics) | `utils.py` L47, L55-57, L383, L387; `rbm_train_main.py` L123. | New. A quirk of the published regulariser; the 70-95% skipped figure is a guess. |
| 3.4 `int16` overflow in high-resolution bins | Blocker | CONFIRMED | `utils.py` L183-187; `verify_output.txt` L2-6 (bin 33515 stored as -32021). | Already C1 in `docs/REVIEW.md`. |
| 3.5 Stack broken on current libraries | Blocker | CONFIRMED | `rbm_train_main.py` L84 (`DataFrame.append`), `models.py` L24, L40 (`uint8` mask), `parameters.py` L30 (`cuda:0`). | Already in `docs/REVIEW.md`. |
| 3.6 Evaluation scripts | Blocker | PARTLY | `TIDE_SEARCH_HUMVAR.sh` L22-23, L29-30, L59-60, L66-67 (no `--fileroot`, `--overwrite T`); `TIDE_SEARCH_IPRG.sh` L9-10 wrong dataset (C3, known); Yeast 50 ppm in Tide vs 3 Da elsewhere; OMSSA and X!Tandem 0 missed cleavages (8 of 9 configs, not 6 of 7) vs Tide 1. | Not confirmed: the Tide flags are shared by both arms; MS Amanda `MaxNumberModSites` limits modifications, not Met count. Whether the second call overwrites depends on Crux naming. |
| 3.7 Tailor index misalignment; negative-norm inversion | Major | CONFIRMED | `dbsearch.py` L564-566, L573-578; `verify_output.txt` item 5. | Known (M8); sign flip plausible, untested. |
| 3.8 Training vs inference mismatches | Major | CONFIRMED, mostly known | `parameters.py` L19 (fragment charge 1); `rbm_train_main.py` L143-150 (summed NLL); M1-M5 in `docs/REVIEW.md`. | New: omitted +2 fragments (effect not checked), summed vs mean NLL. |
| 3.9 Batching, dedup, decoy generation, `.ms2` size | Major | CONFIRMED, one overstatement | `dbsearch.py` L731-743, L399-402 (M6, M7). | W, gradient and momentum at 40,000² float32 are about 19 GB; ">40 GB" not shown. |
| 3.10 Errors in `docs/REVIEW.md` and README | Major | CONFIRMED | `models.py` has 73 lines but REVIEW cites L226-271; utils line cites off; README L58, L59, L68 `rbm_trian_main.py`; Winnow mis-citation; the entrapment preprint (doi 10.64898/2026.07.10.737772, about 6.3% vs nominal 1%) exists. | "Reversed Yeast diagnosis" overstated. |
| 3.11 Minor bugs | Minor | CONFIRMED where checked | `rbm_train_main.py` L201 (`is 'lowres'`); `dbsearch.py` L618. | Others not opened. |

## Sections 1 and 2

Repo facts are correct. Citations: Wen et al., Nat Methods 22 (2025) 1454-1463 (FDRBench),
pUniFind 2507.00087, SpecEncoder (Bioinformatics 40 Suppl 1, 2024), DDA-BERT, InstaNovo-FM,
Tesorai Search (J Mol Biol 2026), Slider (J Proteome Res 2021) exist and match. Unverified: the
Winkelhardt et al. preprint (DOI given as 10.1101/2026.03.06.709532, while `docs/REVIEW.md` has
10.64898/2026.03.04.709532v1; nothing found by search), MS²Rescore 3.0 (search finds Buur,
Declercq, Strobl et al., J Proteome Res 23 (2024) 3200-3207, not MCP 23(4) 100718), and a 2026
Bioinformatics paper for yHydra. Not checked: CHIMERYS, MSFragger-DDA+, HR-XPV, Tide v4.3,
Oktoberfest, the 2025 quantum-RBM papers.

## Section 4: ideas already in the repo

4.1 and 4.2 (Slider-style convolution, peptide-disjoint K-fold, cross-run training, entrapment
audit) are largely in `docs/REVIEW.md` L151-152, L201, L244; the dense-versus-Toeplitz capacity
framing is new. 4.3 (exchangeability theory), 4.4 (predicted-intensity residual calibration and
chimeric deconvolution) and 4.5 (QUBO) are not in the repo.
