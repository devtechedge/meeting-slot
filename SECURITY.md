# Security

**Date:** 2026-09-06 - Prime Intellect training/eval environment. No public website. Keep API keys and env secrets in the host platform only; do not commit credentials.

## Accepted dependency risk

Two Dependabot alerts are open by choice and are not a maintenance oversight.
Both are pinned by the eval framework rather than by a version range we control:

- **`datasets` 4.6.1** (medium, path traversal in folder-based builders). `verifiers`
  requires `datasets>=3.0.0,<4.7.0`, so 5.x cannot be installed. The affected builder
  path is not used: this task ships a single local dataset, not folder-based loading.
- **`fsspec` 2026.2.0** (high, SSTI to RCE in `ReferenceFileSystem`). `datasets 4.6.1`
  caps it at `<=2026.2.0`. Reachable only when reading datasets over HTTP, which this
  task does not do. This is the one to revisit first if anything changes.

Re-evaluate both when `verifiers` ships a release that allows `datasets` 5.x, which
would unblock the pair.
