---
title: "Reading Log: Local-First Software"
workspace: personal
tags: [books, software, architecture]
created: 2026-08-05
---

# Reading Log: Local-First Software

## Thesis

Local-first software keeps the user's data on their device, with sync happening in the background. Users get the availability of local apps and the collaboration of cloud apps. The key conflict is between **offline access** and **multi-device sync**.

## CRDTs

Conflict-free Replicated Data Types (CRDTs) allow concurrent edits to converge without a central server. The two main families are state-based (merge full state) and operation-based (merge deltas). This matters for a note app because two devices can edit the same note offline and reconcile later.

## What I want to try

- Build a small sync layer using a CRDT library on top of plain markdown files.
- Keep the vault as the source of truth; treat the sync log as derived state.
- Compare against the [[note1|Q3 design review]] idea of reducing onboarding friction — a local-first pitch might actually *increase* it.

## Related links

- Obsidian vaults are already local-first markdown.
- See [[note2|the analytics proposal]] for why telemetry needs to be privacy-compliant in a local-first world.
