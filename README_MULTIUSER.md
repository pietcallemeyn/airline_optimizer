# Multi-user Crew Planning — local accounts, Entra-ready identities

## Installation

1. Back up the existing application, especially `data/`, `output/` and `scratch_board_data/`.
2. Copy `app.py`, `optimizer.py`, `scratch_boards.py`, and **new** `auth_local.py` to the application directory. Keep existing `visualize.py`, `data/` and other project files.
3. Create local accounts **on the server terminal**, not through a public registration page:

   ```bash
   python auth_local.py add planner1
   python auth_local.py add planner2
   python auth_local.py list
   ```

   Each add prompts for a password of at least 12 characters. Passwords are stored as salted scrypt hashes in `auth_data/users.json` (not plaintext).
4. Start Streamlit normally (`streamlit run app.py`). Users sign in with their own credentials. No anonymous access to the GUI is permitted.
5. For deployment, enforce HTTPS through a trusted reverse proxy, restrict filesystem access to the server account, and protect `auth_data/` and `output/` in backups. Local login is an initial deployment solution, not an enterprise SSO substitute. Configure session timeouts and rate limiting at the proxy; this version does **not** implement brute-force throttling or persistent login across tabs.

## Data isolation

- Shared, unchanged input: `data/*.csv`.
- User-specific optimizer results and process status: `output/users/<opaque-user-key>/`.
- User-specific scratch JSON: `output/users/<opaque-user-key>/scratch_boards/*.json`.
- Each optimizer subprocess receives `CREW_OUTPUT_DIR` pointing to its user's output folder. Two different users can run independent optimizer processes concurrently. Running two optimizer jobs in the **same** user account is blocked by its PID check.
- Original `output/*.json` and `scratch_board_data/*.json` are **not** automatically migrated or displayed to other users. To preserve older scenarios, migrate them deliberately to the appropriate user's scratch_boards folder after reviewing ownership.
- Shared input data editing in the existing GUI remains shared across users. Do not simultaneously edit shared CSVs; this version isolates **results**, not input datasets or editor writes.
- For a new browser tab, the user signs in again. Identity is not passed in query parameters or URLs.

## Entra ID preparation

Local accounts have immutable internal IDs, separate from usernames. Storage folders use a hash of `local:<internal-id>`. The same storage API supports future `entra:<tenant-id>:<object-id>` identities. A future OIDC integration must validate ID tokens (issuer, audience, nonce, signature) and authorize tenant access server-side; do not trust query parameters or client-provided IDs. Mapping an existing local account's data to an Entra account requires an explicit administrator-controlled migration.

## Test coverage and limits

Verified: Python compilation, local account creation/login rejection, distinct user directory generation, scratch JSON isolation via mock Streamlit state, and output directory override in optimizer. Full interactive browser and multi-process production load tests have **not** been performed.
