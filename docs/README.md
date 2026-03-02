# CAN-IDS Documentation

**Last Updated**: March 1, 2026

Complete index of all project documentation, organized by topic.

---

## Guides

Start here if you're new to the project.

| Document | Description |
|----------|-------------|
| [GETTING_STARTED.md](guides/GETTING_STARTED.md) | Installation, virtual CAN setup, first run, Pi 4 deployment |
| [configuration.md](guides/configuration.md) | Complete parameter reference — config files, CLI args, models |
| [PROJECT_CONTEXT.md](guides/PROJECT_CONTEXT.md) | Project status, key files, roadmap, tuning strategy |
| [rules_guide.md](guides/rules_guide.md) | Detection rule writing guide — all 18 rule types |
| [can_packet_format.md](guides/can_packet_format.md) | CAN bus packet structure reference |
| [traffic_monitoring_guide.md](guides/traffic_monitoring_guide.md) | Live traffic monitoring usage |

---

## Implementation

Technical details on what's built and how it works.

| Document | Description |
|----------|-------------|
| [IMPLEMENTATION_STATUS.md](implementation/IMPLEMENTATION_STATUS.md) | **Authoritative status** — rule engine, ML engine, measured performance, known gaps |
| [UNIMPLEMENTED_FEATURES.md](implementation/UNIMPLEMENTED_FEATURES.md) | Feature inventory with code references and remaining work |
| [INTEGRATION_STATUS.md](implementation/INTEGRATION_STATUS.md) | Vehicle_Models integration — model loading, config changes |
| [enhanced_features_integration.md](implementation/enhanced_features_integration.md) | 8 research-based ML features (TCE-IDS, BTMonitor, etc.) |
| [multistage_rule_integration.md](implementation/multistage_rule_integration.md) | Dual detection architecture design |
| [ADAPTIVE_TIMING_IMPLEMENTATION_SUMMARY.md](implementation/ADAPTIVE_TIMING_IMPLEMENTATION_SUMMARY.md) | Per-CAN-ID adaptive timing thresholds |
| [TIMING_DETECTION_TUNING.md](implementation/TIMING_DETECTION_TUNING.md) | Statistical analysis of timing detection tuning |
| [FUZZING_DETECTION_IMPLEMENTATION.md](implementation/FUZZING_DETECTION_IMPLEMENTATION.md) | Fuzzing attack detection implementation |
| [RULE_GENERATION_SUMMARY.md](implementation/RULE_GENERATION_SUMMARY.md) | Data-driven rule generation from baseline traffic |

---

## ML & Model Documentation

Machine learning detection engine details.

| Document | Description |
|----------|-------------|
| [ML_OPTIMIZATION_GUIDE.md](ml/ML_OPTIMIZATION_GUIDE.md) | 7 strategies to improve ML inference throughput |
| [ML_MODEL_ISSUE.md](ml/ML_MODEL_ISSUE.md) | Known ML model issues and workarounds |
| [ML_DETECTION_NOT_ENABLED.md](ml/ML_DETECTION_NOT_ENABLED.md) | Why IsolationForest `ml_based` mode is deprecated for real-time use |
| [ML_TESTING_BUG.md](ml/ML_TESTING_BUG.md) | ML testing bugs encountered and fixes |
| [DETECTION_TUNING_COMPARISON.md](ml/DETECTION_TUNING_COMPARISON.md) | Comparison of detection tuning approaches |
| [SOLUTION_1_RETRAIN_GUIDE.md](ml/SOLUTION_1_RETRAIN_GUIDE.md) | Guide to retraining ML models |

---

## Performance

Throughput optimization and benchmarking.

| Document | Description |
|----------|-------------|
| [BUILD_PLAN_7000_MSG_SEC.md](performance/BUILD_PLAN_7000_MSG_SEC.md) | 3-day plan for 7,000+ msg/s architecture |
| [ACHIEVING_7000_MSG_PER_SEC.md](performance/ACHIEVING_7000_MSG_PER_SEC.md) | Technical analysis of the 7K target |
| [COMPARISON_ORIGINAL_VS_7K_PLAN.md](performance/COMPARISON_ORIGINAL_VS_7K_PLAN.md) | Incremental vs. architectural approach comparison |
| [HYBRID_APPROACH_CAPABILITIES.md](performance/HYBRID_APPROACH_CAPABILITIES.md) | Week-by-week capability progression |
| [PERFORMANCE_OPTIMIZATION_ROADMAP.md](performance/PERFORMANCE_OPTIMIZATION_ROADMAP.md) | 4-phase optimization plan |
| [PERFORMANCE_ISSUES.md](performance/PERFORMANCE_ISSUES.md) | Real-world performance gap analysis |
| [PREFILTER_IMPLEMENTATION_SUMMARY.md](performance/PREFILTER_IMPLEMENTATION_SUMMARY.md) | Pre-filter stage implementation |
| [BATCH_PROCESSING_TEST_RESULTS.md](performance/BATCH_PROCESSING_TEST_RESULTS.md) | Batch processing benchmarks |
| [COMPREHENSIVE_METRICS_REPORT.md](performance/COMPREHENSIVE_METRICS_REPORT.md) | Full metrics report |
| [LIBRARY_OPTIMIZATION_RECOMMENDATIONS.md](performance/LIBRARY_OPTIMIZATION_RECOMMENDATIONS.md) | Python library optimization notes |
| [resource_monitoring.md](performance/resource_monitoring.md) | CPU/memory monitoring system |

---

## Deployment

Raspberry Pi 4 setup and production deployment.

| Document | Description |
|----------|-------------|
| [RASPBERRY_PI_DEPLOYMENT_GUIDE.md](deployment/RASPBERRY_PI_DEPLOYMENT_GUIDE.md) | Full deployment guide for Pi 4 |
| [RASPBERRY_PI_SETUP.md](deployment/RASPBERRY_PI_SETUP.md) | Pi 4 hardware and OS setup |
| [PI4_VALIDATION_GUIDE.md](deployment/PI4_VALIDATION_GUIDE.md) | Validation testing on Pi 4 |
| [raspberry_pi4_optimization_guide.md](deployment/raspberry_pi4_optimization_guide.md) | Pi 4 performance tuning |

---

## Planning & Architecture

Design documents and improvement plans.

| Document | Description |
|----------|-------------|
| [current_architecture_design.md](planning/current_architecture_design.md) | System architecture overview |
| [ARCHITECTURE_IMPROVEMENT_PLAN.md](planning/ARCHITECTURE_IMPROVEMENT_PLAN.md) | Architecture improvement proposals |
| [IMPROVEMENT_ROADMAP.md](planning/IMPROVEMENT_ROADMAP.md) | 4-phase improvement roadmap |
| [design_modification_analysis.md](planning/design_modification_analysis.md) | Design change impact analysis |
| [PHASE3_ML_OPTION_DECISION.md](planning/PHASE3_ML_OPTION_DECISION.md) | ML option decision for Phase 3 |
| [real_dataset_integration.md](planning/real_dataset_integration.md) | Real CAN dataset integration plan |
| [vehicle_models_integration_plan.md](planning/vehicle_models_integration_plan.md) | Vehicle_Models project integration |

---

## Testing

Test results and methodology.

| Document | Description |
|----------|-------------|
| [TESTING_RESULTS.md](testing/TESTING_RESULTS.md) | Performance test results summary |
| [testing_results_detailed.md](testing/testing_results_detailed.md) | Detailed test results |
| [PERFORMANCE_TESTING_GUIDE.md](testing/PERFORMANCE_TESTING_GUIDE.md) | How to run performance tests |
| [REAL_ATTACK_TEST_RESULTS.md](testing/REAL_ATTACK_TEST_RESULTS.md) | Results from real attack dataset testing |
| [PCA_TEST_RESULTS_DEC17_2025.md](testing/PCA_TEST_RESULTS_DEC17_2025.md) | PCA dimensionality reduction testing |
| [PROOF_OF_RESULTS.md](testing/PROOF_OF_RESULTS.md) | Evidence and reproducibility of results |
| [STAGE2_VERIFICATION_REPORT.md](testing/STAGE2_VERIFICATION_REPORT.md) | Stage 2 rule indexing verification |
| [GAP_ANALYSIS.md](testing/GAP_ANALYSIS.md) | Testing gap analysis |
| [TESTING_ISSUES_DEC16_2025.md](testing/TESTING_ISSUES_DEC16_2025.md) | Testing issues encountered Dec 16 |

---

## Development Logs

Chronological record of implementation sessions. See [CHANGELOG.md](development_logs/CHANGELOG.md) for the consolidated summary.

| Document | Date | Topic |
|----------|------|-------|
| [PROJECT_SUMMARY.md](development_logs/PROJECT_SUMMARY.md) | Initial | Project creation and structure |
| [SESSION_LOG_20251130.md](development_logs/SESSION_LOG_20251130.md) | Nov 30 | RPi4 testing, entropy bug fix |
| [PHASE_1_COMPLETE.md](development_logs/PHASE_1_COMPLETE.md) | Dec 2 | Rule engine Phase 1 (3 critical rule types) |
| [PHASE_2_COMPLETE.md](development_logs/PHASE_2_COMPLETE.md) | Dec 2 | Rule engine Phase 2 (4 important rule types) |
| [PHASE_3_COMPLETE.md](development_logs/PHASE_3_COMPLETE.md) | Dec 2 | Rule engine Phase 3 (4 specialized rule types) |
| [100_PERCENT_COMPLETE.md](development_logs/100_PERCENT_COMPLETE.md) | Dec 2 | 18/18 rule types + ML complete |
| [DECEMBER_3_SESSION_SUMMARY.md](development_logs/DECEMBER_3_SESSION_SUMMARY.md) | Dec 3 | Real-world testing, ML perf issues |
| [TONIGHT_SUMMARY.md](development_logs/TONIGHT_SUMMARY.md) | Dec 11 | Dual-sigma adaptive timing |
| [DEVELOPMENT_SUMMARY_DEC14_2025.md](development_logs/DEVELOPMENT_SUMMARY_DEC14_2025.md) | Dec 14 | Tier 3 payload repetition analysis |
| [PHASE2_EARLY_EXIT_COMPLETE.md](development_logs/PHASE2_EARLY_EXIT_COMPLETE.md) | Dec 14 | Priority-based early exit optimization |
| [STAGE2_RULE_INDEXING_COMPLETE.md](development_logs/STAGE2_RULE_INDEXING_COMPLETE.md) | Dec 14 | O(1) rule indexing → 7,002 msg/s |
| [PHASE3_IMPLEMENTATION_COMPLETE.md](development_logs/PHASE3_IMPLEMENTATION_COMPLETE.md) | Dec 14 | Decision tree ML detector (Stage 3) |