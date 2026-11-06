"""Local identity store. Designed to replace identity provider with Entra tenant+oid."""
from __future__ import annotations
import argparse, hashlib, hmac, json, os, re, secrets, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUTH_DIR = ROOT / 'auth_data'
USERS = AUTH_DIR / 'users.json'

def _load():
    if not USERS.exists(): return {}
    return json.loads(USERS.read_text(encoding='utf-8'))

def _save(data):
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.users_', dir=AUTH_DIR)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(data,f,indent=2)
            f.flush(); os.fsync(f.fileno())
        os.chmod(tmp,0o600)
        os.replace(tmp,USERS)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def _hash(password, salt):
    return hashlib.scrypt(password.encode('utf-8'),salt=bytes.fromhex(salt),n=2**14,r=8,p=1).hex()

def add_user(username,password):
    username=username.strip().lower()
    if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{2,63}',username):
        raise ValueError('Username must have 3-64 lowercase letters, digits, dot, underscore or hyphen')
    if len(password)<12: raise ValueError('Password must contain at least 12 characters')
    data=_load()
    if username in data: raise ValueError('User already exists')
    salt=secrets.token_hex(16)
    uid=secrets.token_hex(16)  # immutable internal identity, independent of username
    data[username]={'id':uid,'salt':salt,'password_hash':_hash(password,salt),'active':True,'provider':'local'}
    _save(data)
    return uid

def authenticate(username,password):
    user=_load().get(username.strip().lower())
    if not user or not user.get('active'):
        # expend comparable work for unknown users
        _hash(password,'00'*16)
        return None
    if hmac.compare_digest(_hash(password,user['salt']),user['password_hash']):
        return {'id':user['id'],'username':username.strip().lower(),'provider':'local'}
    return None

def is_active(identity):
    user=_load().get(identity.get("username", ""))
    return bool(user and user.get("active") and user.get("id")==identity.get("id"))

def identity_storage_key(identity):
    """Stable opaque path key; Entra uses provider=entra, tenant_id, object_id."""
    if identity['provider']=='entra':
        raw=f"entra:{identity['tenant_id']}:{identity['object_id']}"
    else:
        raw=f"local:{identity['id']}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]

def user_output(identity):
    return ROOT/'output'/'users'/identity_storage_key(identity)

def main():
    p=argparse.ArgumentParser(description='Manage local users')
    p.add_argument('action',choices=['add','disable','enable','list'])
    p.add_argument('username',nargs='?')
    args=p.parse_args()
    if args.action=='list':
        for name,u in _load().items(): print(name, 'active' if u.get('active') else 'disabled')
        return
    if not args.username: p.error('username required')
    if args.action=='add':
        import getpass
        password=getpass.getpass('Password (min 12 characters): ')
        print('Created user:',add_user(args.username,password))
    else:
        data=_load(); key=args.username.lower()
        if key not in data: p.error('unknown user')
        data[key]['active']=args.action=='enable'; _save(data)
        print(args.action,key)

if __name__=='__main__': main()
