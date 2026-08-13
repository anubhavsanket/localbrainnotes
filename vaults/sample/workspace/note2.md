---
title: "Analytics Proposal"
workspace: work
tags: [analytics, product, tracking]
created: 2026-07-28
---

# Analytics Proposal

## Why we need it

We currently have no visibility into where users drop off in onboarding. The only signal is a manual survey run after week two, which is slow and biased.

## Proposed events

| Event | Trigger | Properties |
|-------|---------|------------|
| `onboarding_started` | Wizard step one loads | `entry_source`, `has_account` |
| `template_selected` | User picks a template | `template_id`, `step_duration_ms` |
| `onboarding_completed` | Final step confirmed | `total_duration_ms`, `selected_template` |

## Naming convention

All events use `snake_case` with a feature prefix. The feature prefix for onboarding is `onboarding`. See [[note1|the Q3 design review]] for the timeline.

## Rollout

Events will be instrumented behind a feature flag. Telemetry must be **privacy-compliant** — no PII in event properties. We will validate the schema against the design review before enabling by default.
