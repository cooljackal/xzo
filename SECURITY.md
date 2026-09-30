# Security

## Reporting a problem

Please report security problems privately through GitHub: on this repository, open the
**Security** tab and choose **Report a vulnerability**. Please don't open a public issue for them.

## What xzo does and doesn't protect

- **No authentication.** Every endpoint is open to anyone who can reach the port. xzo listens only
  on `127.0.0.1` by default and refuses to listen anywhere else unless `XZO_ALLOW_REMOTE=1` is set.
  If you expose it, put something in front of it that handles authentication.
- **Plaintext storage.** The conversation store (`XZO_DB`, default `xzo_memory.sqlite`) holds
  conversation text and tool results unencrypted. `XZO_DB=none` keeps nothing on disk;
  `XZO_CHUNK_TTL_DAYS` plus `/prune` limits how long anything is kept.
- **Conversations are isolated from each other.** A request can only retrieve pieces of the
  conversation it sent, even though all conversations share one store.
- **Recalled text is fenced.** Text brought back from earlier in a conversation is marked as notes.
  Text that came from tool results is presented as data, not as instructions. This is framing, not
  enforcement: a model may still follow text it was told not to.
- **Content fingerprints are not cryptographic.** Two different texts deliberately crafted to share
  a fingerprint could be confused. This only matters if untrusted parties share one xzo instance,
  which the lack of authentication already rules out.
