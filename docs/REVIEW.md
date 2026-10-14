# BoltzMatch review

Review of [kfattila/BoltzMatch](https://github.com/kfattila/BoltzMatch) (code for Sulimov, Voronkova,
Kertész-Farkas, *Bioinformatics* 2020, btaa206), its lab context, and where it stands against today's
peptide identification methods. Written 2026-10-11.

## 1. Where things are

| Path | What |
|---|---|
| `docs/REVIEW.md` | This file |
| `tests/verify_bugs.py` | Script that imports the repository's own code and confirms 10 of the bugs below |
| `docs/verify_output.txt` | Its output (2026-10-11 run) |

Remotes: `origin` is this fork (`sulimovp/BoltzMatch`); `upstream` is `kfattila/BoltzMatch`. The lab's
successor, Slider, is at [kfattila/Slider](https://github.com/kfattila/Slider).

### Data (Git LFS, about 1.2 GB in total)

| File | Size |
|---|---|
| `fasta/ups1.zip` | 59 KB |
| `ms_data/UPS1.recalre.zip` | 10.8 MB (the smoke test needs only these two) |
| `ms_data/iPRG_continuous_scan.zip` | 11 MB |
| `ms_data/AurumR151.zip` | 17.5 MB |
| `ms_data/v07548_UofF_malaria_TMT_10.zip` | 42 MB |
| `ms_data/HSPP2A.zip` | 97 MB |
| `ms_data/yeast-01_full.zip` | 301 MB |
| `ms_data/Linfeng_012511_HapMap39_3.zip` (HumVar) | 348 MB |
| FASTA zips (IPI human, iPRG, malaria, yeast; target and protein-reversed versions) | 2–50 MB each |

### Rerun the checks

pyteomics and Biopython are not installed. The script substitutes small stand-ins for them, and no
tested code path depends on them.

```bash
python3 -W ignore::SyntaxWarning tests/verify_bugs.py
```

Environment used: Python 3.12.7, torch 2.12.0 (CPU only, no NVIDIA driver), NumPy 1.26.4, pandas 2.2.2.

## 2. How the code works

* `rbm_train_main.py` builds a `DBSearch` (in-silico digestion, binning, Tailor-calibrated search,
  target-decoy q-values). It then trains `RBMModel` once per epoch:
  1. Run the Python Tailor search on normalised spectra.
  2. Take spectra with q < 0.005 and use their top peptide as the positive.
  3. Use all candidates in the precursor window as negatives.
  4. Minimise `-log_softmax` of the positive, plus a "diversifying" penalty.
* Model: score(s, h) = s·W·h + vb·s + hb·h. W is a dense `max_bin × max_bin` matrix (about 2000² at
  1.0005 Da, 40000² at 0.05 Da) with its diagonal fixed to 1.
* Each epoch it re-searches with the transformed spectra, and keeps the epoch with the largest sum of
  PSMs at 1/5/10% FDR. That epoch's transformed spectra are exported to `.ms2`.
* **The numbers in the paper come from the modified Crux/Tide** (`kfattila/crux-toolkit`) searching
  those `.ms2` files, not from the Python search. So bugs in the Python search affect the training
  labels and the epoch choice, not the final scoring.

## 3. Bugs

"Test" means confirmed by `verify_bugs.py`; "reading" means found by reading the code only.

### Critical

| # | Location | Problem | Evidence |
|---|---|---|---|
| C1 | [utils.py:377-381](../source/utils.py) `generate_unique_theoretical_peaks` | Peak bins are stored as `np.int16`. At 0.05 Da, `max_bin` = 40000 > 32767, so every fragment above m/z ≈ 1638 wraps to a negative index, i.e. a random low bin. The high-res model was trained with those peaks in the wrong places. Slider fixed this with `int32`. | Test: bin 33515 became −32021 |
| C2 | Whole stack | The code does not run on a current setup. | See the list below |
| C3 | [scripts/TIDE_SEARCH_IPRG.sh](../scripts/TIDE_SEARCH_IPRG.sh) | `DATA_FILE_BOLTZMATCH_LR/HR` point to `Linfeng_012511_HapMap39_3_boltzmatch_*.ms2` (HumVar data) but search the iPRG index. All `TIDE_SEARCH_*.sh` set `PATH=./CRUX/`, wiping the shell PATH, and hardcode `/home/attila/...`. | Reading |
| C4 | [dbsearch.py:560-582](../source/dbsearch.py) `boltzmatch_tailor_scoring` | When padding to 20 candidates moves `start_id` back (spectra near the end of the mass-sorted peptide list), `norm_scores[i]` no longer belongs to peptide `start_pept + i`, and the wrong peptide wins. Still present in Slider. | Test: correct P24, reported P22 |

C2 in detail:
* `from Bio.Alphabet import generic_protein` ([dbsearch.py:20](../source/dbsearch.py))
  fails because the module was removed in Biopython 1.78 (2020).
* The `uint8` diagonal mask ([models.py:226,242](../source/models.py)) raises
  `RuntimeError: masked_fill_ only supports boolean masks` on torch 2.x (test confirmed).
* `DataFrame.append` ([rbm_train_main.py:84](../rbm_train_main.py)) no longer exists in
  pandas 2 (test confirmed).
* `np.float_` in every mzML config string in `parameters.py` was removed in NumPy 2.0.
* The device is hardcoded to `cuda:0` ([parameters.py:30](../source/parameters.py)).
* `./results/` is never created, but `print_results` writes into it (reading).

### Method and training

| # | Location | Problem | Evidence |
|---|---|---|---|
| M1 | [models.py:265-271](../source/models.py) | `visible_bias[i]` is added to both the positive and the negative scores, so it cancels in the softmax and `rbm_vb` never changes from 0. The search still adds `spectrum.bias`. | Test: \|grad vb\| ≈ 1e-7, \|grad W\| ≈ 1 |
| M2 | [dbsearch.py:517,572](../source/dbsearch.py) vs [models.py](../source/models.py) | Training scores `s·W·h + hb·h`; the search scores `(s·W·h)·50/200 + vb·s + hb·h + 10`, then calibrates. `hb` counts 4× more at search time. | Reading |
| M3 | [utils.py:258](../source/utils.py) vs [utils.py:312-323](../source/utils.py) | Training keeps only the top 100 peaks (`topN_exp`); `transofrm_spectra` uses every peak. | Reading |
| M4 | [utils.py:263](../source/utils.py) | The positive is `spectrum.peptide` with no `target == 1` check, so decoys with q < 0.005 become positives. The positive is also among the negatives, so the loss can't go below log 2. | Reading |
| M5 | [rbm_train_main.py:118-119](../rbm_train_main.py) | The "baseline" Tailor search that makes the labels each epoch resets the spectra but keeps the peptide biases learned in the previous epoch. | Reading |
| M6 | [dbsearch.py:725-743](../source/dbsearch.py) `spectrum_batch_generator` | `shuffle=True` doesn't shuffle: it interleaves the first and second halves in the same order every epoch. The last partial batch is dropped (the `else:` branch is missing a `yield`), and so is one spectrum when the count is odd. | Test: same order every epoch; 896 of 1001 spectra used |
| M7 | [dbsearch.py:397-402](../source/dbsearch.py) | Targets and decoys share one `peptide_set`. If a decoy is generated before an identical target from another protein, the target is dropped. Fix: generate all targets first. Still present in Slider. | Test: `[('ACDEFGK','decoy')]` |
| M8 | [dbsearch.py:574](../source/dbsearch.py) | The Tailor paper divides by the top 1% quantile; the code uses the top 5% (at least the 5th best). `scores += 10` assumes scores are positive, which transformed spectra with negative values don't guarantee. Slider subtracts instead. | Reading |
| M9 | Model selection | The best epoch is chosen by number of accepted PSMs, using decoys on the same data that FDR is reported for. There is no cross-validation. | Design |

### Python search settings vs Crux evaluation vs paper

| Setting | Python (`parameters.py`) | Crux scripts / paper |
|---|---|---|
| Missed cleavages | humvar/yeast/aurum 0, malaria 2, others 1 | Tide index: 1 for all |
| Yeast tolerance | 3 Da | 50 ppm |
| Malaria tolerance | 20 ppm | 50 ppm |
| HSPP2A tolerance | 20 ppm | 50 ppm |
| Max peptide length | 30 (default, never overridden) | 50 |
| Decoys | Reversed peptides with both termini fixed | Reversed proteins (`target-protrev.fasta`, `--decoy-format none`) |
| Epochs | 5 | 15 in the paper |

### Minor

* [rbm_train_main.py:201](../rbm_train_main.py): `args.resolution is 'lowres'` is False
  whenever `-r` is given on the command line, as in both run scripts, so the weight plot never runs
  (test confirmed).
* `div_reg` and `error` are undefined if no batch is processed; `error` divides by zero if the first
  batch has no usable spectra.
* `load_data` iterates over every mzML spectrum: an MS1 scan crashes the charge `eval`.
* For ups1, aurum and hspp2a, `scan_id` is the enumeration index, not the native scan number.
* Config uses `eval` strings.
* `__pycache__` is committed.
* There is no requirements file and no LICENSE; the file header says both "proprietary" and Apache 2.0.
* `RUN_TEST.sh` calls `python3`, `RUN_ALL.sh` calls `python`.

## 4. Engineering improvements

1. **Sparse candidates.** Replace the dense `theo_peaks` (`n_cand × max_bin` on the device;
   [utils.py:280](../source/utils.py)) with index gathers. At 0.05 Da with 1000 candidates
   that's 160 MB per spectrum, times 128 per batch.
2. **Closed-form `div_reg`.** Σ over pairs of h_p·h_q = (‖Σh‖² − Σ‖h‖²)/2. The test confirmed an
   identical value, and it avoids two 8128 × d matrices (about 1.3 GB each at 0.05 Da).
3. **Diagonal.** Use `W.diagonal().fill_(1)`, or define W = I + an off-diagonal part, instead of the
   1.6 GB dense mask at 0.05 Da.
4. **Real PyTorch module.** Make the parameters `nn.Parameter`; detect the device automatically.
5. **Vectorised search.** Score all candidates with a sparse matrix (CSR peptide × bin) times the
   spectrum, instead of a Python loop per PSM that calls `np.unique` every time.
6. **Reproducibility.** Pin requirements; replace `eval` with a YAML config; add `results/`; add CI
   with the UPS1 smoke test; use the same search settings as Crux; add isotope-error support.

## 5. Research notes

### Is the FDR estimate valid?

* The paper argues the score is unbiased because targets and decoys are treated the same way in the
  negative phase. However, the positive phase uses only confident targets, and W, `hb` and the epoch
  choice are all fitted on the same spectra used to report FDR.
* The evidence given (AUC 0.51 between target and decoy scores in the worst-scoring third of PSMs) has
  little power and looks at the wrong region. Any inflation would show up near the threshold.
* One point in its favour: the Crux evaluation uses protein-reversed decoys, which training never saw.
  The remaining risk is that W memorises patterns of frequently identified target peptides.
* A model's ability to memorise target peptides is a known problem: more complex rescoring models can
  raise entrapment FDR by about 500% ([Winnow](https://arxiv.org/pdf/2509.24952)).
* **To do:** k-fold over spectra (as Percolator does), epoch choice without decoys, entrapment checks
  ([Wen et al., Nat Methods 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12240826/)), and
  training on one run while testing on another.

### Model ideas

* **Structured W.** The learned W shows diagonal bands (isotopes, water/ammonia losses). A banded or
  convolutional W plus a low-rank term would have orders of magnitude fewer parameters, make 0.01 Da
  bins feasible and overfit less. Slider has since shown that 465 convolution parameters roughly match
  4 M dense ones.
* **Peptide-dependent terms**, e.g. losses tied to specific amino acids. The paper's own stated
  limitation: BoltzMatch "is not capable of considering loss peaks conditionally for certain amino acids".
* **Hybrid with predicted intensities:** sᵀ W ĥ, where ĥ comes from Prosit, AlphaPeptDeep or MS²PIP
  instead of a binary vector.
* **Chimeric spectra:** let one spectrum match several peptides (as CHIMERYS and MSFragger-DDA+ do).
* **Calibration:** HR-XPV applies, since the transformed score is a dot product (paper Eq. 10–14).

## 6. The Kertész-Farkas lab's work

| Work | What it does | Reported result |
|---|---|---|
| Cascade search; improved FDR estimation (Kertész-Farkas, Keich, Noble; JPR 2015) | Statistics groundwork from the Noble-lab years | n/a |
| **Tailor** (Sulimov & Kertész-Farkas, [JPR 2020](https://pubs.acs.org/doi/abs/10.1021/acs.jproteome.9b00736)) | Divides each score by the top 1% quantile of that spectrum's candidate scores | More PSMs for XCorr, HyperScore and OMSSA; now reported by default in Tide |
| **BoltzMatch** (Bioinformatics 2020) | Dense W; 4 M parameters at low res, 1.6 B at high res | +33% (low res) / +50% (high res) vs XCorr at 0.1% FDR |
| **Slider** (Kudriavtseva, Kashkinov, Kertész-Farkas; [JPR 2021](https://pubmed.ncbi.nlm.nih.gov/34449232/); [repo](https://github.com/kfattila/Slider)) | 1D convolution, BCE loss predicting which bins are fragment peaks of confident PSMs; 465 / 4105 parameters | Trains 3–10× faster than BoltzMatch; low-res results within 2–4% of high-res |
| Crux toolkit (JPR 2023); Tide indexing; **Tide v4.3** ([JPR 2025](https://www.biorxiv.org/content/10.1101/2025.04.01.646675v1.full)) | Search engine at scale | Over 10 M spectra and 7 B peptides in about 20 GB RAM; 2–7× faster than before; similar speed to MSFragger/Sage, which ran out of memory on the largest databases |
| **HR-XPV** (Bhimani, Peresadina, Voznyuk, Kertész-Farkas; Proteomics 2024) | Exact p-values for XCorr on high-res data, by tracking the leftover mass when peaks are binned | Better calibration |
| **SeVa** (Bhimani et al., [IJMS 2026, 519:117532](https://www.sciencedirect.com/science/article/pii/S1387380625001368)) | Finds single amino acid variants without a huge database | n/a |

Theses: Sulimov (2020); Bhimani (defended 2025-10-21); Acquaye (defended 2025-10-17); Kertész-Farkas
habilitation (2022). No learned-scoring paper from the lab after Slider was found.

### BoltzMatch vs Slider vs Prosit (numbers from the two papers)

| | BoltzMatch | Slider | Andromeda + Prosit + Percolator (2019) |
|---|---|---|---|
| HeLa, 0.1% FDR, no Percolator | 1631 | **1757** | 1426 |
| HeLa, 0.1% FDR, with Percolator | **1888** | 1772 | 1426 |
| HeLa, 1% FDR, with Percolator | 2071 (Prosit +5.1%) | Prosit +8.0% | **2176** |
| High res, 0.1% FDR | Slider has 10.1% more | Res-EV has 2.4% more than Slider | n/a |
| FDR validity check | AUC on worst third of PSMs | Same | n/a |

Both learned scores win only at very strict FDR (≤0.1%) and lose to 2019 Prosit at 1%.

### Slider code compared with BoltzMatch's bugs

| Fixed in Slider | Still present in Slider |
|---|---|
| int32 peak bins (C1) | Tailor misalignment, [dbsearch.py:565-588](https://github.com/kfattila/Slider/blob/master/source/dbsearch.py) (C4) |
| Tailor subtracts instead of dividing, so negative scores are fine | Decoy blocks identical target, [dbsearch.py:407](https://github.com/kfattila/Slider/blob/master/source/dbsearch.py) (M7) |
| No dead visible bias (plain conv net) | `Bio.Alphabet` import, [dbsearch.py:11](https://github.com/kfattila/Slider/blob/master/source/dbsearch.py) |
| | Batch generator drops the tail (M6) |
| | Training and FDR on the same data; same weak AUC check (M9) |

## 7. Current state of the field (as of Oct 2026)

* **Prediction-based rescoring.** Predicted fragment intensities, retention time and ion mobility are
  fed into Percolator or mokapot. Gains of +30–89% over Sage, Comet and MSFragger
  ([MS²Rescore, MCP 2024](https://www.mcponline.org/article/S1535-9476(24)00088-4/fulltext)); see also
  [MSBooster](https://www.nature.com/articles/s41467-023-40129-9). A comparison found inSPIRE best below
  1% FDR and MS²Rescore above it, both ahead of Oktoberfest
  ([Proteomics 2025](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/pmic.202400225)).
* **Chimeric-spectrum search.** CHIMERYS (Nat Methods 2025) reports about 2× Sequest HT and about two
  PSMs per spectrum ([preprint](https://www.biorxiv.org/content/10.1101/2024.05.27.596040.full.pdf));
  also [MSFragger-DDA+](https://www.nature.com/articles/s41467-025-58728-z).
* **Learned embeddings / matching.** [SpecEncoder](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i257/7700866)
  (+6–15% unique peptides over MS-GF+ with Percolator), [SpeCollate](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0259349),
  DeepSearch, WinnowNet.
* **Rescoring evens out search engines.** After MS²Rescore or Oktoberfest, the variation between seven
  engines drops from 67% to about 8–10% ([bioRxiv 2026](https://www.biorxiv.org/content/10.64898/2026.03.04.709532v1.full)).
  This is the central argument against investing in a better first-pass score alone.
* **Stricter FDR checking.** Entrapment ([Wen et al. 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12240826/))
  is becoming the expected test for any learned score.

## 8. Can it beat SOTA? Conclusion and plan

**As a standalone score: no.** Even against 2019 Prosit it lost at 1% FDR, and rescoring has improved
a lot since. A one-peptide-per-spectrum score also can't reach chimeric-search gains.

**Where it could win** (estimates, not measured):
1. **As an extra feature in mokapot/Percolator** next to MS²Rescore features. It adapts to each
   dataset without outside training data, which the pretrained predictors don't. Expect single-digit
   percent gains on standard tryptic HCD data.
2. **Where pretrained predictors do poorly:** unusual fragmentation (EThcD, UVPD), cross-links,
   non-tryptic immunopeptides, rare modifications, non-model organisms, low-res ion-trap data.

**Proposed plan, using what the lab already has:**
1. Fix C1–C4 and M1–M8; pin the environment; make the UPS1 smoke test pass on CPU.
2. Replace the dense W with a Slider-style convolution plus peptide-dependent terms.
3. Pretrain on large public data searched with Tide 4.3; apply to held-out runs. This removes the
   same-data FDR problem.
4. Calibrate with HR-XPV; add the score as a Percolator feature (Crux already includes Percolator).
5. Benchmark against Sage + mokapot + MS²Rescore and MSFragger + MSBooster (+ Percolator) on modern
   data (e.g. the MS²Rescore and entrapment benchmark sets), with cross-validation and entrapment-checked FDR.

## 9. Next steps in this fork

Suggested branches, one per bug group:

| Branch | Covers |
|---|---|
| `fix/modern-stack` | C2 |
| `fix/highres-int16` | C1 |
| `fix/tailor-alignment` | C4, M8 |
| `fix/training-consistency` | M1–M7 |
| `fix/scripts` | C3 |

Once the stack is pinned, `tests/verify_bugs.py` should use the real pyteomics and Biopython instead of stand-ins.

To run end to end, fetch only the smoke-test data (about 11 MB):

```bash
git lfs pull --include="fasta/ups1.zip,ms_data/UPS1.recalre.zip"
```

The full benchmark needs the modified Crux from `kfattila/crux-toolkit`, or Tide 4.3, which now supports Tailor.
