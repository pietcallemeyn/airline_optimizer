# Crew Planning v3.1 — browser session across tabs

Replace `app.py` and add `browser_sessions.py` and the entire `auth_browser_component/` directory. Keep `auth_local.py`, `optimizer.py` and `scratch_boards.py` from the ZIP as supplied. Do not overwrite existing `auth_data/users.json` or `output/users/`.

The first login issues a random server-side session token with an eight-hour absolute lifetime. A small Streamlit component saves that token in the browser's localStorage; a new tab restores it and the server verifies the token before displaying data. The token is not passed in the URL. Sign out revokes it server-side and removes it from localStorage.

**Security:** localStorage tokens can be read by JavaScript on the application origin and are not equivalent to HttpOnly Secure cookies. Deploy only behind HTTPS on a trusted internal network; for internet-facing use, prefer Microsoft Entra OIDC or a reverse proxy that provides server-managed HttpOnly cookies. A browser tab already logged in may retain its in-memory Streamlit session until its next authentication check even after logout elsewhere. Session lifetime is eight hours, not an idle timeout. No automated real-browser test was performed; test login, new-tab opening, reload, and logout in your deployed Streamlit version.

The login session is shared among tabs in the same browser profile. A private/incognito window has its own session. Existing personal output paths and scratch boards remain unchanged.
