import json, re, sys
from collections import defaultdict

XDBG = re.compile(r"\[XDBG\] (\S+) t=(\d+)(?: reqs=\['([^']+)'\]| transfer=(\S+)| req=(\S+))?")
TTFT = re.compile(r'\[TTFT\]\s*(\{.*\})')

def parse(path):
    xdbg = []           # (name, t_epoch_ns, id)
    ttft = defaultdict(dict)  # rid -> event -> ts_ns (perf)
    for line in open(path):
        m = XDBG.search(line)
        if m:
            xdbg.append((m.group(1), int(m.group(2)), m.group(3) or m.group(4) or m.group(5)))
        t = TTFT.search(line)
        if t:
            e = json.loads(t.group(1))
            rid = str(e.get('req_id') or '')
            if rid.startswith('chatcmpl'):
                ttft[rid][e['event']] = e['ts_ns']
    return xdbg, ttft

def stem(rid):
    if rid is None: return None
    for p in ('chatcmpl-', 'xfer-'):
        if rid.startswith(p): rid = rid[len(p):]
    parts = rid.split('-')
    if len(parts[-1]) != 12: parts = parts[:-1]  # drop proxy suffix (uuid 尾段定长12)
    return '-'.join(parts)

px, pttft = parse('/tmp/ttft_exp/X2_prefill.log')
dx, dttft = parse('/tmp/ttft_exp/X2_decode.log')

# --- calibrate: P perf_counter -> epoch ---
pairs = []
for rid, ev in pttft.items():
    if 'first_token_generated' in ev:
        st = stem(rid)
        for name, t, xid in px:
            if name == 'P_REQ_FINISHED' and stem(xid) == st:
                pairs.append((t - ev['first_token_generated']))
                break
pairs.sort()
off_ns = pairs[len(pairs)//2]
print(f'calib pairs={len(pairs)} offset_ms={off_ns/1e6:.1f}  (spread {[round(p/1e6,2) for p in pairs]})')

# --- collect per-request events ---
reqs = defaultdict(dict)  # stem -> dict
for name, t, xid in px:
    reqs[stem(xid)][name] = t
for name, t, xid in dx:
    reqs[stem(xid)]['D_' + name if name.startswith('P_') else name] = t
# P-side perf events -> epoch
for rid, ev in pttft.items():
    s = stem(rid)
    for k, v in ev.items():
        reqs[s]['P_' + k] = v + off_ns
for rid, ev in dttft.items():
    s = stem(rid)
    for k, v in ev.items():
        reqs[s]['D_' + k] = v  # perf of D process, only for intra-D diffs

order = ['P_req_queued','P_req_scheduled','P_worker_execute_start','D_SEND_PULL','P_RECV_PULL',
         'P_WAIT_WAKEUP','P_READY','P_SEND_EXEC','P_model_forward_start','P_model_forward_end',
         'P_first_token_generated','P_REQ_FINISHED','P_POST_RECORD',
         'D_req_queued','D_req_scheduled','D_worker_execute_start','D_SEND_PULL2','D_first_token_generated']
print(f'\n{"stem":>12} ' + ' '.join(f'{o.replace("P_","").replace("D_","")[:10]:>10}' for o in order))
for s, r in sorted(reqs.items(), key=lambda kv: kv[1].get('D_SEND_PULL', 0)):
    if 'D_SEND_PULL' not in r: continue
    base = r['D_SEND_PULL']
    row = []
    for o in order:
        key = o if o in r else ('D_SEND_PULL' if o == 'D_SEND_PULL2' else o)
        if o == 'D_SEND_PULL2':
            row.append(f'{(r["D_SEND_PULL"]-base):>10.1f}'); continue
        v = r.get(o)
        if v is None: row.append(f'{"--":>10}')
        elif o.startswith('D_') and o != 'D_SEND_PULL':
            # D perf clock, show diff vs D req_queued
            dq = r.get('D_req_queued')
            row.append(f'{(v-dq)/1e6:>10.1f}' if dq else f'{"?":>10}')
        else:
            row.append(f'{(v-base)/1e6:>10.1f}')
    print(f'{s[-8:]:>12} ' + ' '.join(row))
print('\nclient-side TTFT from X2_seed123.txt:')
for line in open('/tmp/ttft_exp/X2_seed123.txt'):
    if 'TTFT' in line or 'avg' in line: print(' ', line.strip())
