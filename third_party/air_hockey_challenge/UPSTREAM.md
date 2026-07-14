# Vendored: air_hockey_challenge (2023 tournament framework)

- Upstream: https://github.com/AirHockeyChallenge/air_hockey_challenge
- Branch: `tournament`, base commit `5e25d6ca59bb6ee5b113e67737d3f7db0d6e7f04`
  ("Update website links")
- Vendored on 2026-07-14 from the former sibling clone
  `~/air_hockey_challenge_2023` (its local branch `data-collection`,
  head `5fe3335`).

Local commits applied on top of upstream, as reviewable patches in
[`upstream-patches/`](upstream-patches/):

1. `0001` Add green goal-mouth markers to the table
2. `0002` Color the table rims red
3. `0003` Make the absorbing guards robust to contact softness and spin

These are data-generation changes for the SA world-model dataset pipeline in
this repo; they are already applied in this vendored tree (the patches are
provenance, not something to apply).

The 2023 pipeline is frozen on this era's APIs (see `setup_env_2023.bash` at
the repo root for the pinned stack). To pull upstream changes anyway:
re-clone upstream at `tournament`, re-apply `upstream-patches/` with
`git am`, and replace this directory with a `git archive` export.
