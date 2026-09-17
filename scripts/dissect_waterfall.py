import json, re, sys
from collections import defaultdict
from datetime import datetime

LINE = re.compile(r'\(EngineCore pid=(\d+)\) (DEBUG|INFO) (\d+-\d+ \d+:\d+:\d+) \[([^\]]+)\] (.*)')
TTFT = re.compile(r'\[TTFT\]\s*(\{.*\})')

def wall_ns(s):  # '09-16 14:41:36' -> epoch ns (assume 2026, local); second precision
    dt = datetime.strptime('2026-' + s, '%Y-%m-%d %H:%M:%S')
    return int(dt.timestamp() * 1e9)

def parse(path, role):
    ttft = defaultdict(dict)   # rid -> event -> (ts_ns, wall)
    anchors = defaultdict(dict)  # rid -> name -> wall
    done_lines = []            # prefill: (wall, took)
    order = []                 # global (wall, ts_or_None, desc)
    for line in open(path):
        m = LINE.match(line)
        if not m: continue
        pid, lvl, wall, src, msg = m.groups()
        w = wall_ns(wall)
        t = TTFT.search(msg)
        if t:
            e = json.loads(t.group(1))
            rid = str(e.get('req_id') or '')
            if not rid.startswith('chatcmpl'): continue
            ttft[rid][e['event']] = (e['ts_ns'], w)
            order.append((e['ts_ns'], w, rid[:20] + ' ' + e['event']))
            continue
        if role == 'decode':
            for pat, name in [
                (r"update_state_after_alloc: req_id=(chatcmpl-[\w-]+)", 'alloc'),
                (r"Sending kv transfer request for \{'(chatcmpl-[\w-]+)'\}", 'sendpull'),
                (r"pulling kv_caches for \['(chatcmpl-[\w-]+)'\] finished", 'pullfin'),
                (r"Finished recving KV transfer for request (chatcmpl-[\w-]+)", 'finrecv'),
            ]:
                mm = re.search(pat, msg)
                if mm:
                    anchors[mm.group(1)][name] = w
                    order.append((None, w, mm.group(1)[-12:] + ' ' + name))
        else:
            for pat, name in [
                (r"MooncakeConnector request_finished, req_id=(chatcmpl-[\w-]+)", 'p_finished'),
                (r"Sending kv_caches for request (chatcmpl-[\w-]+) \((\d+) blocks\)", 'p_send_start'),
            ]:
                mm = re.search(pat, msg)
                if mm:
                    anchors[mm.group(1)][name] = w
                    order.append((None, w, mm.group(1)[-12:] + ' ' + name))
            mm = re.search(r"Sending to [\d.]+:\d+ done, took ([\d.]+)", msg)
            if mm:
                done_lines.append((w, float(mm.group(1))))
    return ttft, anchors, done_lines, order

for tag in ['B3', 'A3']:
    print(f'########## {tag} ##########')
    d_ttft, d_anc, _, d_order = parse(f'/tmp/ttft_exp/{tag}_decode.log', 'decode')
    p_ttft, p_anc, p_done, _ = parse(f'/tmp/ttft_exp/{tag}_prefill.log', 'prefill')
    # map prefill done-lines to requests by order of p_send_start
    senders = sorted([(v.get('p_send_start'), rid) for rid, v in p_anc.items() if 'p_send_start' in v])
    senders = [rid for _, rid in sorted([(w, rid) for w, rid in senders])]
    for i, rid in enumerate(senders):
        if i < len(p_done):
            p_anc[rid]['p_done'], p_anc[rid]['took'] = p_done[i]
    for rid in sorted(d_ttft, key=lambda r: d_ttft[r].get('req_queued', (0,0))[0]):
        e = d_ttft[rid]; a = d_anc.get(rid, {}); p = p_anc.get(rid, {})
        def ms(ev): return e[ev][0]/1e6 if ev in e else float('nan')
        q, s, x = ms('req_queued'), ms('req_scheduled'), ms('worker_execute_start')
        fwd = ms('model_forward_end') - ms('model_forward_start') if 'model_forward_end' in e else float('nan')
        took = p.get('took', float('nan'))
        print(f"{rid[-12:]}: queued→sched={s-q:6.1f} sched→exec={x-s:6.1f} fwd={fwd:5.1f} took={took*1e3 if took==took else float('nan'):6.1f}ms")
        # cross-component second-level diffs
        if a and p:
            def wd(dic, k): return dic[k]//10**9 if k in dic else None
            print(f"    D: sendpull={wd(a,'sendpull')} pullfin={wd(a,'pullfin')} finrecv={wd(a,'finrecv')} | P: pfin={wd(p,'p_finished')} pstart={wd(p,'p_send_start')} pdone={wd(p,'p_done')}")
