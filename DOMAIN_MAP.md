# Meridian Project Delivery Advisor: domain mapping

Source: process-operations-agent. The architecture, UI and LangSmith wiring stay the same.
The domain changes, and tracing moves from a tracing project to agent addressing.

| Structural role | Process operations | Capital project delivery |
|---|---|---|
| Entity | HX-2500-4417 exchanger | CP-2500-4417 compressor package, Saltmarsh LNG Expansion |
| Catalog | MP-HX-2500, 2500 kW duty | MP-CP-2500, 2500 kW rated |
| Code / threshold | ALM-441 / 95 C trip | TD-441 / 4.5 mm/s shaft vibration trip |
| Low stress vs actual | 30% check / 90% campaign | 25% recycle run / 100% load performance test |
| History / root cause | two deferred cleanings | two waived hot-alignment hold points |
| Two-limit entitlement | support: date or 2000 h | vendor warranty: 2027-10-14 or 2000 h |
| Two-limit schedule | 500 h or 12 months | coupling alignment check: 500 h or 12 months |
| Superseded item | PN-4926553 -> PN-4938966 bundle | PN-4926553 -> PN-4938966 coupling |
| Availability | zero stock, 6/7/9 days | zero stock, Houston 6 / Rotterdam 7 / Singapore 9 days, vs 4 days float |
| Customer / partner / case | SITE / SVC / CASE | PRJ / VND / TQ (technical query) |
| Numeric KB rule | Q = m Cp dT = 627 kW | Pt = 1.5 P (St/S) = 1.5 x 40 x 1.25 = 75 barg |
| Forbidden mutation | setpoints, interlocks, batch release | test acceptance, hold-point waivers, TQ close-out, change orders, certificates, permits, trip settings, equipment start/stop |
| Specialists | historian / realtime / knowledge | quality records / project controls / engineering |
| Tracing destination | LANGSMITH_PROJECT | LANGSMITH_AGENT_ID + LANGSMITH_AGENT_ENVIRONMENT |
| Resource grouping | `Application` tag | `Agent` tag (value = agent id) |

Sectors covered: LNG (CP-2500-4417), chemicals (SP-1000-8820), mining (PP-1250-3390),
green hydrogen (EL-600-7712) and gas processing (HX-150-2204, stale snapshot).

Kept from the source: 17 KB documents, 3 Context Hub skills, 4 datasets (16/14/12/22),
9 offline evaluator keys (`equipment_action_blocked` is now `project_action_blocked`),
trace anonymization, the feedback queue, online evaluation (`equipment_action_safe` is now
`project_action_safe`, with the old key in `LEGACY_FEEDBACK_KEYS`), Insights and both CI
workflows. The planted weaknesses stay the same: numeric recall, a low-load diagnostic
shortcut and an injection-only classifier.
