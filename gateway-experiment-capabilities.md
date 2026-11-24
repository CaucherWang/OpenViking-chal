# Gateway experiment capabilities

This document records the behavior preserved from `stash@{0}` while rebasing the
experiment infrastructure onto the current OpenViking `main`.

## Trial memory isolation

- A request opts in with `X-OpenViking-Memory-Isolation: trial` and a valid
  `X-OpenViking-Actor-Peer`.
- Capture, replay, recall, stored response IDs, and OpenViking sessions are scoped
  to the trial peer so concurrent trials cannot share state.
- Recall is restricted to actor-peer memory and rejects results outside that peer.
- Captured user messages retain their peer ID. Trial sessions enable peer memory
  extraction and disable self memory extraction.
- Explicit OpenViking tools are bounded to the peer tree and the trial's sessions.
- Trial requests fail closed when isolation preparation fails; they are never
  forwarded without memory isolation.
- Trial mode verifies that the parent user memory is empty before the model call.
  Development mode uses the dedicated `benchmark-trials-v1` memory user.

## Upstream billing evidence

- Every model, summary, and hidden-tool continuation call gets a separate ledger
  entry with timing, status, completion, model, service tier, and reported usage.
- Missing token counters remain missing; provider-reported cache and reasoning
  counters are retained without copying prompts, credentials, or arbitrary fields.
- The client response carries `X-OpenViking-Request-ID`, correlating it with the
  persisted request log.
- Failed calls and client-disconnected streams still close their ledger entries.
- Billing evidence observes calls without changing provider replies.

## Source safety

The original tracked changes remain in `stash@{0}`. The implementation on this
branch is adapted to the current request-record, peer, capture, and tool-loop APIs
rather than restoring the previous Gateway implementation wholesale.
