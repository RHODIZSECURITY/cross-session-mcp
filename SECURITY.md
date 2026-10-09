# Security policy

## Supported versions

Only the latest release receives security fixes.

## Reporting a vulnerability

Please do not open a public issue. Report it privately through
[GitHub security advisories](https://github.com/reiarseni/cross-session-mcp/security/advisories/new).
You can expect an acknowledgement within a few days.

## Scope

cross-session-mcp is read-only on the harness stores: it must never write
to them. Secrets and personal data found in transcripts must be redacted
before they are sent to the Gemini API or stored in the memory bank, and
the API key must never appear in URLs, logs or tool output. Reports that
break any of these guarantees are treated as high priority.
