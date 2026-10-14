"""Confirm suspected BoltzMatch bugs against the repository code itself.
pyteomics and Biopython are not installed, so minimal stubs stand in for them;
nothing tested here touches the stubbed functions except where noted."""
import sys, types, itertools, subprocess
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# --- stubs for missing third-party modules ---------------------------------
pt = types.ModuleType('pyteomics')
for name in ('mzml', 'parser', 'mass'):
    setattr(pt, name, types.ModuleType(name))
pt.parser.isoforms = lambda p, variable_mods=None, max_mods=None: [p]
sys.modules.update({'pyteomics': pt, 'pyteomics.mzml': pt.mzml,
                    'pyteomics.parser': pt.parser, 'pyteomics.mass': pt.mass})
bio = types.ModuleType('Bio'); bio.SeqIO = types.ModuleType('SeqIO')
alpha = types.ModuleType('Bio.Alphabet'); alpha.generic_protein = None
sys.modules.update({'Bio': bio, 'Bio.SeqIO': bio.SeqIO, 'Bio.Alphabet': alpha})

import numpy as np, torch, pandas as pd
import source.parameters as params
params.types['torch_device'] = torch.device('cpu')
import source.dbsearch as dbsearch, source.utils as utils, source.models as models

def head(t): print('\n=== ' + t)

# 1. int16 overflow of theoretical peak bins at high resolution ------------
head('1. int16 overflow in generate_unique_theoretical_peaks (highres)')
dbs = dbsearch.DBSearch(bin_width=0.05, bin_offset=0.4, max_theo_pept_peak_charge=1, static_mods={})
print('max_bin at 0.05 Da =', dbs.max_bin, '(int16 max = 32767)')
pep = dbsearch.PeptideObj(0, 'WWWWWWWWWK', 0, 1, 'WWWWWWWWWK', 'full', 0)
pep.aa_mass = [dbs.aa_mass[a] for a in pep.peptide_seq]
dbs.peptide_collection = [pep]
utils.generate_unique_theoretical_peaks(dbs)
dbs.calculate_peptide_fragmentation(0)
true_bins = sorted(set(pep.peaks[0]['b']) | set(pep.peaks[0]['y']))
print('true bins   :', true_bins[-4:])
print('stored bins :', sorted(pep.unique_peaks)[:4], '... (negative = wrapped)')
print('wrapped peaks:', int((np.asarray(pep.unique_peaks) < 0).sum()), 'of', len(true_bins))

# 2. visible bias receives zero gradient ------------------------------------
head('2. rbm_vb gets exactly zero gradient (same term on both phases)')
dbs = dbsearch.DBSearch(bin_width=1.0005079, static_mods={})
m = models.RBMModel(dbs)
d = m.rbm_dim
exp = np.random.rand(4, d); cand = (np.random.rand(4, d) < .01).astype(float)
theo = [torch.tensor((np.random.rand(30, d) < .01), dtype=torch.float32) for _ in range(4)]
loss = sum(-torch.log_softmax(s, 0)[0] for s in m.scoring(exp, theo, cand))
loss.backward()
print('|grad vb| =', m.rbm_vb.grad.abs().max().item(), ' |grad hb| =', m.rbm_hb.grad.abs().max().item(),
      ' |grad W| =', m.rbm_weights.grad.abs().max().item())

# 3. uint8 mask indexing in current torch -----------------------------------
head('3. weight_update with uint8 diagonal mask on torch ' + torch.__version__)
opt = torch.optim.SGD((m.rbm_weights, m.rbm_vb, m.rbm_hb), lr=1e-3, momentum=.9, nesterov=True)
try:
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        m.weight_update(loss=sum(-torch.log_softmax(s, 0)[0] for s in m.scoring(exp, theo, cand)), optimizer=opt)
    diag = torch.diagonal(m.rbm_weights).detach()
    print('ok; diag all 1:', bool((diag == 1).all()), '| warnings:', [str(x.message)[:80] for x in w][:1])
except Exception as e:
    print('FAILS:', type(e).__name__, str(e)[:150])
print('dense mask memory at 0.05 Da: %.1f GB' % (40000**2 / 1e9))

# 4. `args.resolution is 'lowres'` -------------------------------------------
head("4. `args.resolution is 'lowres'` with -r lowres on the command line")
out = subprocess.run([sys.executable, '-W', 'ignore', '-c',
    "import argparse,sys;p=argparse.ArgumentParser();p.add_argument('-r','--resolution',default='lowres');"
    "a=p.parse_args(sys.argv[1:]);print('identity check:', a.resolution is 'lowres')", '-r', 'lowres'],
    capture_output=True, text=True)
print(out.stdout.strip() or out.stderr.strip()[-200:], '-> plot_weights never runs when -r is passed (as in RUN_TEST.sh)')

# 5. Tailor scoring index misalignment near the end of the peptide list ------
head('5. boltzmatch_tailor_scoring: score/peptide misalignment for heaviest spectra')
dbs = dbsearch.DBSearch(bin_width=1.0005079, static_mods={})
peps = []
for k in range(25):
    p = dbsearch.PeptideObj(1000 + k, 'P%d' % k, 0, 1, 'P', 'full', 0); p.bias = 0; peps.append(p)
dbs.peptide_collection = peps
dbs.calculate_xcorr_score = lambda s, p: 100.0 if p == 24 else 0.0  # only P24 matches
s = dbsearch.Spectrum('x', 1, np.arange(10.) * 100 + 100, np.ones(10), 2, 600.0, 2000, False, 1.5)
s.start_pept, s.end_pept, s.n_candidates, s.bias = 22, 25, 3, 0
dbs.spectrum_collection = [s]; dbs.reset_search_results(); dbs.boltzmatch_tailor_scoring()
print('candidates are peptides 22..24; correct winner = P24; reported winner =', s.peptide.peptide_seq)

# 6. spectrum_batch_generator ------------------------------------------------
head('6. spectrum_batch_generator: "shuffle" and the dropped tail')
dbs.spectrum_collection = list(range(1001))
b1 = [list(x) for x in dbs.spectrum_batch_generator(128, shuffle=True)]
b2 = [list(x) for x in dbs.spectrum_batch_generator(128, shuffle=True)]
seen = set(itertools.chain(*b1))
print('identical order across epochs:', b1 == b2, '| spectra used:', len(seen), 'of 1001',
      '| first batch starts', b1[0][:6])

# 7. Decoy generated before the equal target suppresses the target ---------
head('7. target lost when an earlier decoy has the same sequence')
dbs = dbsearch.DBSearch(static_mods={}, min_pept_mass=0)
dbs.add_peptide_collection('ACDEFGK', 0, 0)   # decoy from protein 0
dbs.add_peptide_collection('ACDEFGK', 1, 1)   # identical target from protein 1
print('collection:', [(p.peptide_seq, 'target' if p.target else 'decoy') for p in dbs.peptide_collection])

# 8. Train / inference score mismatch ---------------------------------------
head('8. score scale used in training vs. in the Python search')
print('train: s.W.h + hb.h   |  search: (s.W.h)*50/200 + vb.s + hb.h + 10, then /Tailor quantile')
print('-> learned hb is weighted 4x more at search time than during training;')
print('   training also keeps only top-100 peaks (topN_exp) but transofrm_spectra does not')

# 9. pandas / numpy / Biopython API removals --------------------------------
head('9. APIs removed in the current stack')
print('pandas', pd.__version__, 'DataFrame.append exists:', hasattr(pd.DataFrame, 'append'))
print('numpy', np.__version__, '(np.float_ removed in NumPy 2.0; used in every mzML config string)')
print('Bio.Alphabet removed in Biopython 1.78 (Sept 2020) -> dbsearch.py import fails')

# 10. Diversifying regulariser: O(B^2 d) -> O(B d) closed form --------------
head('10. div_reg closed form matches and avoids the 8128 x d pair matrices')
B = 128; X = np.random.rand(B, d)
i1, i2 = utils.choose_indices(B)
ref = utils.calculate_div_reg(m, X, i1, i2, div_reg_rate=1000).item()
H = torch.from_numpy(X).float() @ m.rbm_weights + m.rbm_hb
s_ = H.sum(0)
fast = abs(1000 * ((s_ @ s_ - (H * H).sum()) / 2) / (len(i1) * d)).item()
print('reference %.6f  closed form %.6f  rel.err %.1e' % (ref, fast, abs(ref - fast) / ref))
