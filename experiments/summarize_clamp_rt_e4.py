import json,glob,collections,statistics as st
rows=collections.defaultdict(list)
for f in glob.glob('results/clamp_rt/e4/runs/*.json'):
    r=json.load(open(f)); c=r['config']; fin=r['final']
    m=r['run_id'].split('__')[0]
    rows[(c['dataset'],c['num_clients'],m,c['attack'],c['num_byzantine'])].append((fin['dpd_soft'],fin['auc']))
def mean(x): return round(st.mean(x),3)
for ds in ['bail','credit','pokec_z','pokec_n']:
    print('==',ds)
    for K in (5,10):
        keys=sorted(k for k in rows if k[0]==ds and k[1]==K)
        for k in keys:
            v=rows[k]; print(K,k[2],k[3],'f',k[4],'n',len(v),'dpd',mean([x[0] for x in v]),'auc',mean([x[1] for x in v]))
