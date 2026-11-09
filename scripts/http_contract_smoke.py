"""Four-backend HTTP contract smoke tests without paid API or model weights.

This validates request formatting, transport, and response extraction.
It DOES NOT validate real model capabilities or hardware support.
"""
from __future__ import annotations
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from types import SimpleNamespace
from unittest.mock import patch
import benchmark as b

GOLD = b.data()[0]['gold']
QUERIES = []

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n=int(self.headers.get('Content-Length','0'))
        obj=json.loads(self.rfile.read(n))
        QUERIES.append((self.path,obj))
        if self.path == '/v1/responses':
            assert obj['model']=='gpt-6-luna'
            assert b.SYSTEM in obj['input'][0]['content']
            payload={'status':'completed','output':[{'content':[{'type':'output_text','text':json.dumps(GOLD)}]}],
                     'usage':{'input_tokens':120,'output_tokens':18}}
        elif self.path == '/v1/decisions':
            assert obj['model']=='gpt-6-luna'
            assert len(obj['questions'])==3
            assert obj['questions'][0]['type']=='choice'
            assert b.SYSTEM in obj['input']
            answers=[]
            for q in obj['questions']:
                fld=q['name']
                ans={'name':fld,'type':'choice','choice':GOLD[fld],
                     'probabilities':[{'value':v['value'],'probability':float(v['value']==GOLD[fld])} for v in q['choices']],
                     'confidence':0.99}
                answers.append(ans)
            payload={'model':'gpt-6-luna','answers':answers,'usage':{'input_tokens':150}}
        elif self.path == '/v1/systemone':
            assert b.SYSTEM in obj['state']
            assert len(obj['questions'])==3
            answers={}
            for name,q in obj['questions'].items():
                assert q['type']=='choice'
                answers[name]={'type':'choice','choice':GOLD[name],
                                'probabilities':{k:float(k==GOLD[name]) for k in q['criteria']}}
            payload={'model':obj.get('model','loaded-local-model'),'answers':answers,'usage':{'input_tokens':160}}
        else:
            self.send_response(404);self.end_headers();return
        body=json.dumps(payload).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self,*args):pass

server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
port=server.server_address[1]
server_thread=threading.Thread(target=server.serve_forever,daemon=True)
server_thread.start()
try:
    base=f'http://127.0.0.1:{port}'
    args=SimpleNamespace(reasoning='none',timeout=3,clef_url=base,clef_model='Cloudflare/clef-flash',d1_url=base,d1_model='')
    record=b.data()[0]['text']
    with patch.object(b,'OPENAI_URL',base+'/v1'),patch.dict(os.environ,{'OPENAI_API_KEY':'MOCK_KEY'}):
        for backend in b.BACKENDS:
            pred,meta=b.call_backend(backend,record,args)
            assert pred==GOLD,(backend,pred)
            if backend=='luna-llm':
                assert meta['probabilities']=={}
            else:
                assert len(meta['probabilities'])==3
                assert all(b.valid_distribution(f,meta['probabilities'][f]) for f in b.FIELDS)
            print(backend,'mock HTTP OK')
    assert len(QUERIES)==4,len(QUERIES)
    local_requests=[body for path,body in QUERIES if path=='/v1/systemone']
    assert len(local_requests)==2
    assert sum('model' in req for req in local_requests)==1,local_requests
    print('HTTP contract integration: OK; four mock routes; no real model, no API usage')
finally:
    server.shutdown()
    server.server_close()
