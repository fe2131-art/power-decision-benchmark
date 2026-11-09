"""HTTP integration smoke test against fake local SystemOne servers. NOT a model run."""
import contextlib
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

import benchmark

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length=int(self.headers['Content-Length'])
        body=json.loads(self.rfile.read(length))
        if self.path!='\u002fv1/systemone':
            self.send_response(404);self.end_headers();return
        text=body['state']
        gold=benchmark.data(benchmark.HERE / 'data' / 'legacy' / 'demo_10_cases.jsonl')[0]['gold']  # transport fixture; no inference
        answer={}
        for f,v in body['questions'].items():
            labels=list(v['criteria'])
            answer[f]={'type':'choice','choice':gold[f],
                       'probabilities':{k:(0.9 if k==gold[f] else 0.1/(len(labels)-1)) for k in labels},'confidence':0.9}
        payload=json.dumps({'model':body.get('model','loaded-local-model'),'answers':answer,'usage':{'input_tokens':77,'output_tokens':0}}).encode()
        self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    def log_message(self,*args): pass

servers=[]
threads=[]
try:
    for _ in range(2):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        servers.append(server)
        t=threading.Thread(target=server.serve_forever,daemon=True)
        t.start();threads.append(t)
    with tempfile.TemporaryDirectory() as d:
        d1_base=f'http://127.0.0.1:{servers[0].server_address[1]}'
        clef_base=f'http://127.0.0.1:{servers[1].server_address[1]}'
        cmd=[sys.executable, str(Path(benchmark.__file__)), '--backends','d1-2b,clef-flash', '--d1-url',d1_base,
             '--clef-url',clef_base,'--dataset',str(benchmark.HERE / 'data' / 'legacy' / 'demo_10_cases.jsonl'),'--limit','4','--outdir',d]
        p=subprocess.run(cmd,text=True,capture_output=True,timeout=15)
        print(p.stdout)
        if p.returncode:
            print(p.stderr,file=sys.stderr)
            raise SystemExit(p.returncode)
        raw=json.loads((Path(d)/'summary.json').read_text('utf-8'))
        assert set(raw['summary'])=={'d1-2b','clef-flash'}
        assert all(x['success']==4 for x in raw['summary'].values())
        assert (Path(d)/'summary.csv').exists()
        assert (Path(d)/'case_comparison.csv').exists()
        import csv
        with (Path(d)/'case_comparison.csv').open(encoding='utf-8-sig',newline='') as fp:
            cases=list(csv.DictReader(fp))
        assert len(cases)==8
        assert cases[0]['gold_equipment']=='breaker'
        assert cases[0]['pred_equipment']=='breaker'
        assert len((Path(d)/'predictions.jsonl').read_text().splitlines())==8
        print('HTTP full pipeline: OK (mock servers, no actual inference)')
finally:
    for server in servers:
        server.shutdown();server.server_close()
