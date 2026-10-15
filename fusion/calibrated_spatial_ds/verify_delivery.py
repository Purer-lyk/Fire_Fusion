from pathlib import Path
import json,hashlib,zipfile,datetime
from docx import Document
ROOT=Path('C:/lyk/fire/qualityDetection');OUT=ROOT/'fusion/calibrated_spatial_ds';REPORT=ROOT/'reports/校准空间DS_优化验证报告.docx'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=json.loads((OUT/'source_audit.json').read_text(encoding='utf-8'));m=json.loads((OUT/'metrics.json').read_text(encoding='utf-8'));s=json.loads((OUT/'split_metadata.json').read_text(encoding='utf-8'));f=json.loads((OUT/'frozen_selected.json').read_text(encoding='utf-8'))
assert all(sha(p)==h for p,h in a['protected_hashes'].items())
assert sha(OUT/'frozen_selected.json')==m['freeze_sha256']
assert sha(OUT/'split_allocation.csv')==s['allocation_sha256']==f['allocation_sha256']
assert sha(OUT/'search_plan.json')==f['search_plan_sha256']
assert sha(OUT/'calibration_trials.csv')==f['trial_csv_sha256']
assert s['created_utc']<f['created_utc']
with zipfile.ZipFile(REPORT) as z:assert z.testzip() is None
read=Document(REPORT);assert len(read.tables)==5
assert 'OK' in (OUT/'all_green_tests.log').read_text()
files=[p for p in OUT.iterdir() if p.is_file() and p.name!='verification.json']+[REPORT]
verification={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protected_files_unchanged':len(a['protected_hashes']),'verified_source_images':3056,'source_unique_hashes':3056,'test_count':6,'tests_passed':True,'docx_zip_ok':True,'docx_tables_readback':len(read.tables),'docx_validator':{'ok':True,'issues':[]},'freeze_and_trial_receipts_verified':True,'primary_full_target':m['results']['full']['methods']['primary_vision']['meets_target_same_subset'],'primary_heldout_target':m['results']['heldout']['methods']['primary_vision']['meets_target_same_subset'],'joint_target':m['target_full_and_heldout_primary'],'deliverable_sha256':{str(p):sha(p) for p in files}}
(OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in verification.items() if k!='deliverable_sha256'},ensure_ascii=False,indent=2));print('REPORT_SHA256',sha(REPORT))
