"""Builds docs/index.html (self-contained) from results/results.json."""
import json, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
r = json.load(open(root/'results/results.json'))
data = {}
for m, md in r['models'].items():
    data[m] = {}
    for s, sd in md['seeds'].items():
        e = sd['external']; st = sd['source_test']
        data[m][s] = dict(T=sd['T'], lr=sd['lr'],
            src={k: st['cal'][k] for k in ('accuracy','macro_f1','ece15_width','auroc_error')},
            ext={k: e['cal'][k] for k in ('accuracy','macro_f1','ece15_width','auroc_error')},
            ext_uncal_ece=e['uncal']['ece15_width'],
            ci=e['ci']['accuracy'], sel=e['selective'],
            rc=sd['plots']['external']['rc_curve'] if 'plots' in sd else None,
            rel=sd['plots']['external']['reliability_bins'] if 'plots' in sd else None)
html = (root/'report/explorer_template.html').read_text().replace('/*DATA*/null', json.dumps(data))
(root/'docs/index.html').write_text(html)
print(len(html))
