"""Read-only local readiness diagnostic. Does not send records to remote services."""
import argparse
import os
import platform
import socket
import sys
from urllib.parse import urlparse


def reachable(url, timeout=0.8):
    parsed=urlparse(url)
    host,port=parsed.hostname,parsed.port
    if not host or not port:
        return False, 'Invalid host/port'
    try:
        with socket.create_connection((host,port), timeout=timeout):
            return True,'TCP port reachable (model readiness NOT verified)'
    except OSError as e:
        return False,f'Port unavailable: {type(e).__name__}'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--clef-url',default='http://127.0.0.1:30000')
    ap.add_argument('--d1-url',default='http://127.0.0.1:8080')
    args=ap.parse_args()
    print('Python:',platform.python_version(), '('+platform.system()+' '+platform.machine()+')')
    print('Python >= 3.10:',sys.version_info >= (3,10))
    print('OPENAI_API_KEY:','SET (value not displayed)' if os.getenv('OPENAI_API_KEY') else 'NOT SET')
    for name,url in [('clef-flash',args.clef_url),('d1-2b',args.d1_url)]:
        ok,msg=reachable(url)
        print(f'{name} {url}: {msg}')
    print('This checks ports only, not GPU/model loading, response schema or accuracy.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
