"""Short-lived server-side browser sessions for local Streamlit accounts.

The browser stores a bearer token in localStorage via a dedicated Streamlit
component. This is suitable for an HTTPS-protected internal deployment, but
unlike an HttpOnly cookie it is accessible to browser JavaScript. Prefer
reverse-proxy-managed HttpOnly cookies or Entra OIDC for internet exposure.
"""
from __future__ import annotations
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from pathlib import Path

import auth_local

DB = auth_local.ROOT / 'auth_data' / 'browser_sessions.sqlite3'
SESSION_SECONDS = 8 * 60 * 60


def _connect():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB, timeout=15)
    con.execute('PRAGMA busy_timeout=15000')
    con.execute('CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, username TEXT NOT NULL, expires INTEGER NOT NULL)')
    return con


def create(identity):
    token = secrets.token_urlsafe(48)
    with _connect() as con:
        con.execute('DELETE FROM sessions WHERE expires < ?', (int(time.time()),))
        con.execute('INSERT INTO sessions VALUES (?,?,?,?)', (
            hashlib.sha256(token.encode()).hexdigest(), identity['id'], identity['username'],
            int(time.time()) + SESSION_SECONDS))
    try: os.chmod(DB, 0o600)
    except OSError: pass
    return token


def resolve(token):
    if not isinstance(token, str) or len(token) > 256 or len(token) < 40:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    with _connect() as con:
        row = con.execute('SELECT user_id,username,expires FROM sessions WHERE token_hash=?', (digest,)).fetchone()
    if not row or row[2] <= time.time():
        return None
    identity = {'id': row[0], 'username': row[1], 'provider': 'local'}
    return identity if auth_local.is_active(identity) else None


def revoke(token):
    if isinstance(token, str):
        with _connect() as con:
            con.execute('DELETE FROM sessions WHERE token_hash=?', (hashlib.sha256(token.encode()).hexdigest(),))
