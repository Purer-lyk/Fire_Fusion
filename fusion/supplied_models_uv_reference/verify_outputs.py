"""Independent serialized-output audit; does not rerun inference or overwrite old outputs."""
import run_supplied_models as r
import json,csv,zipfile
import numpy as np
from collections import Counter
from docx import Document
from pathlib import Path

out=r.OUT
summary=json.loads((out/'execution_summary.json').read_text(encoding='utf-8'))
rows=list(csv.DictReader((out/'fusion_results.csv').open(encoding='utf-8-sig')))
js=json.loads((out/'fusion_results.json').read_text(encoding='utf-8'))['rows']
assert len(rows)==len(js)==1528 and list(map(int,(x['pair_index'] for x in rows)))==list(range(1528))
assert all(set(a)==set(b) for a,b in zip(rows,js))
samples=[r.c.parse_sample_line(l) for l in (r.SOURCE/'paper_sample.txt').read_text(encoding='utf-8-sig').splitlines()]
for mode in ['rgb','thermal']:
    data=json.loads((out/(mode+'_predictions.json')).read_text(encoding='utf-8'))
    r.check_alignment(samples,data['images'],mode)
    for i,item in enumerate(data['images']):
        assert r.sha(item['image_path'])==item['image_sha256']
        assert all(d['score']>.5 and d['class_id']==0 for d in item['detections'])
        assert float(rows[i][mode+'_confidence'])==max((d['score'] for d in item['detections']),default=0.)
metrics=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
for name,expected in metrics['methods'].items():
    y=[int(x['UV_reference_label']) for x in rows]; p=[int(x[name+'_binary05']) for x in rows]
    assert y==[x.uv for x in samples]
    counts=Counter(zip(y,p))
    actual=dict(TP=counts[(1,1)],FP=counts[(0,1)],FN=counts[(1,0)],TN=counts[(0,0)])
    assert all(expected[k]==v for k,v in actual.items()) and sum(actual.values())==1528
    assert expected['accuracy']==(actual['TP']+actual['TN'])/1528
for name in ['rgb_bpa','thermal_bpa','uv_bpa','legacy_fixed_01','classic_ds','candidate_geometry_repeated','recommended_spatial_then_ds']:
    for row in rows:
        masses=[float(row[name+'_'+key]) for key in ['fire','nofire','uncertain']]
        assert all(0<=x<=1 for x in masses) and abs(sum(masses)-1)<1e-12
fresh=np.loadtxt(out/'cpp_reference_fresh_results.txt'); old=np.loadtxt(out/'cpp_reference_results.txt')
assert fresh.shape==old.shape==(1528,15)
err=float(np.max(np.abs(fresh-old))); assert err<1e-10
for p,h in summary['source_algorithm_sha256'].items(): assert r.sha(p)==h
r.make_report(summary)
doc=Document(r.REPORT)
readback='\n'.join([p.text for p in doc.paragraphs]+[cell.text for table in doc.tables for row in table.rows for cell in row.cells])
for phrase in ['1528','YOLO26n','循环评价','cv2.imread','原C++固定0.1','0.085079','0.457490','未定义（分母为0）']: assert phrase in readback,phrase
(out/'report_readback.txt').write_text(readback,encoding='utf-8')
with zipfile.ZipFile(r.REPORT) as z:
    assert z.testzip() is None
    media=[x for x in z.namelist() if x.startswith('word/media/')]
assert len(media)==5
verification=json.loads((out/'deliverable_verification.json').read_text(encoding='utf-8'))
verification.update(docx_sha256=r.sha(r.REPORT),docx_crc_ok=True,docx_readback_ok=True,fresh_compiled_cpp_exit_code=0,fresh_compiled_cpp_rows=1528,fresh_compiled_vs_existing_max_abs_error=err,fresh_reference_exe_sha256=r.sha(out/'cpp_reference_fresh.exe'),all_3056_original_image_hashes_reverified=True,serialized_CSV_JSON_metrics_independently_recomputed=True,docx_embedded_examples=len(media))
r.save('deliverable_verification.json',verification)
r.save('fresh_cpp_build_verification.json',dict(compile_command='g++ -std=c++17 -O2 -static -I C:/lyk/paper/bpa_test/include C:/lyk/fire/qualityDetection/fusion/cpp_reference_harness.cpp C:/lyk/paper/bpa_test/src/bpa.cpp C:/lyk/paper/bpa_test/src/vision_bpa.cpp C:/lyk/paper/bpa_test/src/temp_bpa.cpp C:/lyk/paper/bpa_test/src/uv_bpa.cpp C:/lyk/paper/bpa_test/src/ds_fusion.cpp -o C:/lyk/fire/qualityDetection/fusion/supplied_models_uv_reference/cpp_reference_fresh.exe',exit_code=0,rows=1528,values_per_row=15,fresh_vs_previous_max_abs_error=err,fresh_exe_sha256=r.sha(out/'cpp_reference_fresh.exe'),source_hashes_verified_unchanged=True))
print(json.dumps(verification,ensure_ascii=False,indent=2))
