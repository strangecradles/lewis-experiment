# Decisions Log

## [2026-04-06] Project initialization
- Target hardware: GH200 (141GB unified memory)
- All models loaded once, kept in memory across conditions
- GQA as primary benchmark (tagged question types enable per-capability analysis)
- 39 total conditions: 7 main + 6 controls × 3 seeds each = 39 runs
- Cross-attention connectors: 2-layer, 256 hidden dim, 4 heads, ~5M params per pair
