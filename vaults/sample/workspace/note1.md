---
title: "Q3 Design Review"
workspace: work
tags: [design, meeting, q3]
created: 2026-08-01
---

# Q3 Design Review

## Context

The design team reviewed the Q3 roadmap. The core theme this quarter is **reducing friction in the onboarding flow**. Key stakeholders include the product, design, and engineering leads.

## Decisions

- The new onboarding flow will use a **three-step wizard**: account creation, workspace setup, then template selection.
- We are **dropping the email-required gate** on the first screen — users can browse templates before signing up. This is expected to lift activation by roughly 8%.
- Dark mode will ship in the **same release** as the wizard, not a follow-up patch.

## Open questions

- Should the wizard support Google single sign-on in the initial release, or defer to Q4?
- Analytics events need a naming convention — see [[note2|the analytics page]] for the proposal.

## Action items

- [ ] Update the Figma wireframes to remove the email gate.
- [ ] Prototype the template-selection step with 3 sample templates.
- [ ] Schedule a cross-functional review before the sprint cut-off.
