# Security policy

## Reporting a vulnerability
Please **do not open a public issue**. Use GitHub's "Report a vulnerability" (Security tab) or contact the maintainer privately.

## Threat model (summary)
- The web console only listens on `127.0.0.1` and rejects requests whose `Host`/`Origin` is not local (CSRF / DNS rebinding).
- Cartridge imports are untrusted input: archives are checked against path traversal and size limits, and foreign SQLite files
  containing unknown triggers or views are rejected.
- Memory content and LLM output are HTML-escaped before display.
- The console is **not** designed to be exposed to a network. Do not bind it to `0.0.0.0` or put it behind a public proxy without authentication.
- Your memory (`body/` or `HMEM_HOME`) contains personal data. Never commit it.
