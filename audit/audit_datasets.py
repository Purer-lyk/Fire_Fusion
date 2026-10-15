"""Read-only source dataset audit; stdlib only."""
from pathlib import Path
import struct, math, json, hashlib, re, datetime, collections
import xml.etree.ElementTree as ET

def jpeg_size(path):
    with open(path,'rb') as f:
        if f.read(2)!=b'\xff\xd8': raise ValueError('not JPEG')
        while True:
            b=f.read(1)
            if not b: raise ValueError('no SOF')
            if b!=b'\xff': continue
            marker=f.read(1)
            while marker==b'\xff': marker=f.read(1)
            m=marker[0]
            if m in (0,0xD8,0xD9) or 0xD0<=m<=0xD7: continue
            n=struct.unpack('>H',f.read(2))[0]
            if m in (0xC0,0xC1,0xC2,0xC3,0xC5,0xC6,0xC7,0xC9,0xCA,0xCB,0xCD,0xCE,0xCF):
                _,h,w=struct.unpack('>BHH',f.read(5)); return w,h
            f.seek(n-2,1)

def split_records(records,reserved):
    # Union filename bursts AND exact hash duplicates BEFORE exclusion.
    parent={r['group']:r['group'] for r in records}
    def root(g):
        while parent[g]!=g:
            parent[g]=parent[parent[g]]; g=parent[g]
        return g
    hashes={}
    for r in records:
        h=r['sha256']; g=r['group']
        if h in hashes: parent[root(g)]=root(hashes[h])
        else: hashes[h]=g
    blocked={root(r['group']) for r in records if r['sha256'] in reserved}
    groups=collections.defaultdict(list); excluded=[]
    for r in records:
        g=root(r['group']); r['split_group']=g
        if not r['valid'] or g in blocked:
            r['exclusion_reason']='invalid_annotation' if not r['valid'] else ('reserved_test_hash' if r['sha256'] in reserved else 'reserved_test_temporal_or_duplicate_group')
            excluded.append(r)
        else: groups[g].append(r)
    ordered=sorted(groups,key=lambda g:hashlib.sha256(('qualityDetection-v1:'+g).encode()).hexdigest())
    target=round(sum(map(len,groups.values()))*.2); val=[]; train=[]
    for g in ordered:
        if abs(len(val)+len(groups[g])-target)<abs(len(val)-target): val.extend(groups[g])
        else: train.extend(groups[g])
    return train,val,excluded

def read_annotation(path,size):
    issues=[]; boxes=[]; classes=[]; declared=None
    try:
        xml=ET.parse(path).getroot()
        declared=[int(xml.findtext('size/width')),int(xml.findtext('size/height'))]
        if declared!=list(size): issues.append('xml_size_mismatch')
        for obj in xml.findall('object'):
            name=obj.findtext('name','').strip(); classes.append(name)
            if name not in ('fire','Fire'): issues.append('unexpected_class:'+name)
            b=[float(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax')]
            boxes.append(b)
            x,y,X,Y=b
            if not all(math.isfinite(v) for v in b) or not (0<=x<X<=size[0] and 0<=y<Y<=size[1]): issues.append('invalid_box')
    except Exception as e: issues.append('xml_error:'+type(e).__name__)
    return {'valid':not issues,'issues':issues,'boxes':boxes,'classes':classes,'xml_size':declared,'xml_path':str(path).replace('\\','/')}

def yolo_agreement(path,size,boxes):
    converted=[]; ids=[]; issues=[]
    try:
        for line in Path(path).read_text().splitlines():
            if not line.strip(): continue
            fields=line.split()
            if len(fields)!=5: raise ValueError('not five columns')
            c=int(fields[0]); ids.append(c); x,y,w,h=map(float,fields[1:]); W,H=size
            b=[(x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H]
            if c!=0: issues.append('unexpected_yolo_class:'+str(c))
            if not all(math.isfinite(v) for v in b) or not (-.01<=b[0]<b[2]<=W+.01 and -.01<=b[1]<b[3]<=H+.01): issues.append('invalid_yolo_box')
            converted.append(b)
    except Exception as e: issues.append('yolo_error:'+type(e).__name__)
    pending=list(converted); max_error=0
    for b in boxes:
        if not pending: issues.append('box_count_mismatch'); break
        distances=[max(abs(v-u) for v,u in zip(b,p)) for p in pending]
        i=min(range(len(distances)),key=distances.__getitem__); max_error=max(max_error,distances[i]); pending.pop(i)
    if len(converted)!=len(boxes): issues.append('box_count_mismatch')
    if max_error>.01: issues.append('coordinate_mismatch')
    return {'agrees':not issues,'issues':sorted(set(issues)),'class_ids':ids,'boxes':converted,'max_coordinate_error_px':max_error}

def build(output,sources,test):
    output=Path(output); sources={k:Path(v) for k,v in sources.items()}; test=Path(test)
    audit=output/'audit'; datasets=output/'datasets'; audit.mkdir(parents=True,exist_ok=True)
    def save(path,data):
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    def image(path,modality):
        with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        r={'path':path.resolve().as_posix(),'basename':path.name,'modality':modality,'sha256':digest}
        try: r['width'],r['height']=jpeg_size(path); r['image_valid']=True
        except Exception as e: r.update(width=0,height=0,image_valid=False,image_error=str(e))
        return r
    def coco(records):
        images=[]; annotations=[]
        for i,r in enumerate(sorted(records,key=lambda r:r['path']),1):
            images.append({'id':i,'file_name':r['path'],'width':r['width'],'height':r['height']})
            for x,y,X,Y in r['boxes']:
                annotations.append({'id':len(annotations)+1,'image_id':i,'category_id':1,'bbox':[x,y,X-x,Y-y],'area':(X-x)*(Y-y),'iscrowd':0})
        return {'info':{'description':'Existing fire boxes only. No UV-derived truth. Coordinates retained as continuous xyxy; no speculative VOC -1 shift.'},'images':images,'annotations':annotations,'categories':[{'id':1,'name':'fire','supercategory':'fire'}]}
    rows=[]; row_errors=[]
    for line_no,line in enumerate((test/'paper_sample.txt').read_text().splitlines(),1):
        f=line.split()
        if len(f)!=4:
            row_errors.append({'line':line_no,'reason':'not_four_columns'}); continue
        try: tmax=float(f[2]); uv=float(f[3]); assert math.isfinite(tmax) and math.isfinite(uv)
        except Exception: row_errors.append({'line':line_no,'reason':'nonfinite_numeric'}); continue
        rows.append({'line':line_no,'rgb':f[0],'thermal':f[1],'Tmax':tmax,'UV_input':uv})
    test_records={m:[image(p,m) for p in sorted((test/folder).glob('*.jpg'))] for m,folder in [('rgb','bgr'),('thermal','thermal')]}
    reserved={r['sha256'] for rs in test_records.values() for r in rs}
    all_sources={}; stats={}
    for m,source in sources.items():
        records=[]; xml_dir=source/('voc_annotations' if m=='thermal' else 'annotations')
        for p in sorted((source/'images').glob('*.jpg')):
            r=image(p,m); r.update(read_annotation(xml_dir/(p.stem+'.xml'),(r['width'],r['height'])))
            if not r['image_valid']: r['valid']=False; r['issues'].append('image_error')
            if m=='thermal':
                r['txt_path']=(source/'annotations'/(p.stem+'.txt')).as_posix()
                r['txt_audit']=yolo_agreement(r['txt_path'],(r['width'],r['height']),r['boxes'])
                if not r['txt_audit']['agrees']: r['valid']=False; r['issues']+=r['txt_audit']['issues']
            records.append(r)
        # Consecutive timestamp bursts, split only at gaps >120 seconds.
        timestamped=[]
        for r in records:
            match=re.match(r'^(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d+)',r['basename'])
            if match:
                values=list(map(int,match.groups())); ts=datetime.datetime(*values[:6]).timestamp()+values[6]/10000
                timestamped.append((ts,r))
            else:
                base=re.sub(r'\.rf\.[0-9a-f]+$','',Path(r['basename']).stem,flags=re.I)
                base=re.sub(r'_aug\d+','',base)
                r['group']=m+':origin:'+base; r['group_method']='augmentation_origin' if '.rf.' in r['basename'] else 'singleton_unknown_provenance'
        last=None; burst=None
        for ts,r in sorted(timestamped,key=lambda v:v[0]):
            if last is None or ts-last>120: burst=r['basename']
            r['group']=m+':burst:'+burst; r['group_method']='timestamp_gap_120s'; last=ts
        all_sources[m]=records
        names={Path(r['path']).stem for r in records}
        stats[m]={'source':source.as_posix(),'images':len(records),'xml_files':len(list(xml_dir.glob('*.xml'))),'orphan_xml':[p.as_posix() for p in xml_dir.glob('*.xml') if p.stem not in names], 'observed_classes':dict(collections.Counter(c for r in records for c in r['classes'])),'valid_images':sum(r['valid'] for r in records),'negative_images':sum(not r['boxes'] and r['valid'] for r in records),'issues':dict(collections.Counter(v for r in records for v in set(r['issues']))),'group_methods':dict(collections.Counter(r['group_method'] for r in records))}
        if m=='thermal':
            stats[m]['txt_files']=len(list((source/'annotations').glob('*.txt')))
            stats[m]['orphan_txt']=[p.as_posix() for p in (source/'annotations').glob('*.txt') if p.stem not in names]
            stats[m]['txt_xml_agree_images']=sum(r['txt_audit']['agrees'] for r in records)
            stats[m]['txt_xml_coordinate_agreement_images']=sum(not any(i in ('coordinate_mismatch','box_count_mismatch') or i.startswith('yolo_error') for i in r['txt_audit']['issues']) for r in records)
            stats[m]['yolo_class_ids']=dict(collections.Counter(c for r in records for c in r['txt_audit']['class_ids']))
    flat=[r for rs in all_sources.values() for r in rs]
    by_hash=collections.defaultdict(list); by_name=collections.defaultdict(list)
    for r in flat: by_hash[r['sha256']].append(r); by_name[r['basename']].append(r)
    duplicates=[{'sha256':h,'paths':[r['path'] for r in rs]} for h,rs in by_hash.items() if len(rs)>1]
    test_summary={}; basename_conflicts=[]; exact_matches=[]; candidate_records=[]
    for m,records in test_records.items():
        covered=[]
        for r in records:
            same=by_hash.get(r['sha256'],[])
            r['exact_annotation_candidates']=[{'source':s['path'],'xml':s['xml_path'],'valid':s['valid'],'modality':s['modality']} for s in same]
            for s in same: exact_matches.append({'test_path':r['path'],'source_path':s['path'],'sha256':r['sha256'],'valid_annotation':s['valid']})
            for s in by_name.get(r['basename'],[]):
                if s['sha256']!=r['sha256']: basename_conflicts.append({'test_path':r['path'],'source_path':s['path'],'test_sha256':r['sha256'],'source_sha256':s['sha256']})
            valid=[s for s in same if s['valid'] and s['modality']==m and r['image_valid']]
            signatures={json.dumps(sorted(s['boxes'])) for s in valid}
            if valid and len(signatures)==1:
                r['boxes']=valid[0]['boxes']; r['gt_status']='exact_hash_existing_annotation'; r['annotation_provenance']=[s['xml_path'] for s in valid]; covered.append(r)
            else: r['gt_status']='conflicting_candidates' if len(signatures)>1 else 'unlabeled'; r['boxes']=None
            candidate_records.append(r)
        save(datasets/m/'test_existing_gt.json',coco(covered))
        test_summary[m]={'images':len(records),'gt_covered':len(covered),'gt_uncovered':len(records)-len(covered),'gt_negative':sum(not r['boxes'] for r in covered),'gt_boxes':sum(len(r['boxes']) for r in covered),'invalid_images':sum(not r['image_valid'] for r in records),'distinct_hashes':len({r['sha256'] for r in records}),'coverage_definition':'valid source annotation AND identical SHA256 AND same modality; uncovered images never treated as negative'}
    manifest=[]; split_paths={}
    # Block duplicate hashes across modalities too, even though models are independent.
    cross_group={}
    for r in flat:
        cross_group.setdefault(r['sha256'],r['group'])
        if len({s['modality'] for s in by_hash[r['sha256']]})>1: r['group']=cross_group[r['sha256']]
    for m,records in all_sources.items():
        train,val,excluded=split_records(records,reserved)
        for split,rs in [('train',train),('val',val),('excluded',excluded)]:
            for r in rs: r['split']=split
            if split!='excluded': save(datasets/m/(split+'.json'),coco(rs))
        stats[m].update(train_images=len(train),val_images=len(val),excluded_images=len(excluded),train_negative=sum(not r['boxes'] for r in train),val_negative=sum(not r['boxes'] for r in val),exclusion_reasons=dict(collections.Counter(r['exclusion_reason'] for r in excluded)),retained_groups=len({r['split_group'] for r in train+val}),train_boxes=sum(len(r['boxes']) for r in train),val_boxes=sum(len(r['boxes']) for r in val),val_fraction=len(val)/len(train+val) if train or val else None)
        assert not ({r['sha256'] for r in train}&{r['sha256'] for r in val})
        assert not ({r['sha256'] for r in train+val}&reserved)
        assert not ({r['group'] for r in train}&{r['group'] for r in val})
        manifest.extend(records)
        split_paths[m]={'train':(datasets/m/'train.json').as_posix(),'val':(datasets/m/'val.json').as_posix(),'test_existing_gt':(datasets/m/'test_existing_gt.json').as_posix(),'image_dir':'','file_name_policy':'absolute original read-only image paths','category_id':1,'category_name':'fire','keep_empty':True,'allow_empty':True}
        cfg='metric: COCO\nnum_classes: 1\nTrainDataset:\n  name: COCODataSet\n  image_dir: ""\n  anno_path: '+json.dumps(split_paths[m]['train'])+'\n  dataset_dir: ""\n  data_fields: [image, gt_bbox, gt_class, is_crowd]\n  allow_empty: true\n  empty_ratio: 1.0\nEvalDataset:\n  name: COCODataSet\n  image_dir: ""\n  anno_path: '+json.dumps(split_paths[m]['val'])+'\n  dataset_dir: ""\n  allow_empty: true\n  data_fields: [image, gt_bbox, gt_class, is_crowd]\n'
        (datasets/m/'ppyolov2_dataset.yml').write_text(cfg,encoding='utf-8')
    pair_names={m:{r[m] for r in rows} for m in ('rgb','thermal')}
    inventory_names={m:{r['basename'] for r in rs} for m,rs in test_records.items()}
    report={'sources':stats,'test':test_summary,'test_pairs':{'rows':len(rows),'invalid_rows':row_errors,'missing_referenced_images':{m:sorted(pair_names[m]-inventory_names[m]) for m in pair_names},'unreferenced_images':{m:sorted(inventory_names[m]-pair_names[m]) for m in pair_names},'UV_role':'fourth-column sensor input ONLY; never ground truth','scene_ground_truth':'not supplied; no scene labels inferred'},'exact_test_source_matches':exact_matches,'same_basename_different_hash':basename_conflicts,'internal_source_duplicate_hashes':duplicates,'paths':split_paths,'policy':{'split':'deterministic greedy nearest 20 percent by SHA256-sorted group identifiers; whole burst/origin/hash groups','seed':'qualityDetection-v1','timestamp_burst_gap_seconds':120,'reserved_policy':'exclude exact reserved image hashes in BOTH modalities plus entire source temporal/augmentation/duplicate groups touching those hashes','classes':'observed fire / Fire explicitly case-normalized to fire; other names invalid and require human mapping','box_policy':'strict actual JPEG size and 0<=xmin<xmax<=width; no clipping or unverified coordinate shift','txt_xml_tolerance_px':.01,'ground_truth_policy':'export candidates only if valid and identical SHA256 and nonconflicting source boxes'},'limitations':['RGB numerical filenames do not identify videos/scenes; temporal independence cannot be certified; numeric adjacency is not invented video truth.','Filename timestamp bursts are heuristic, not verified video identifiers.','Nonidentical near-duplicates are not detected by SHA256; augmentation names are grouped where recognizable.','Source folder tags are not validated scene ground truth.','Configs are dataset fragments, not complete PPYOLOv2 model/training configs; no training performed.','JPEG header dimensions inspected with stdlib; full pixel decode/corruption verification not performed.']}
    save(audit/'audit.json',report); save(audit/'manifest.json',manifest); save(audit/'test_manifest.json',candidate_records); save(audit/'test_pairs.json',rows); save(datasets/'dataset_paths.json',split_paths)
    return report

if __name__=='__main__':
    result=build(Path('C:/lyk/fire/qualityDetection'),{'rgb':Path('C:/lyk/fire/_paddleDetection/paddleDetection/dataset/fire/fireOrigin'),'thermal':Path('C:/lyk/fire/TorchDetection/datasets/thermal/totals')},Path('C:/lyk/paper_data/paper_data_/paper_data'))
    print(json.dumps({'sources':result['sources'],'test':result['test'],'exact_matches':len(result['exact_test_source_matches']),'basename_conflicts':len(result['same_basename_different_hash']),'duplicate_hash_groups':len(result['internal_source_duplicate_hashes'])},indent=2))

