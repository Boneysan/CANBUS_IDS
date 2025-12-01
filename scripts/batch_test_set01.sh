#!/bin/bash
#
# Batch test script for all CSV files in set_01/train_01
# Runs comprehensive tests on each dataset and saves results
#

set -e

# Configuration
DATA_DIR="/media/boneysan/Memorex USB/Masters Project/cantrainandtest/can-train-and-test/set_01/train_01"
OUTPUT_BASE="academic_test_results/batch_set01_$(date +%Y%m%d_%H%M%S)"
RULES_FILE="config/rules.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
source "$PROJECT_DIR/venv/bin/activate"

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Log file
LOG_FILE="$OUTPUT_BASE/batch_test.log"
SUMMARY_FILE="$OUTPUT_BASE/batch_summary.txt"

echo "======================================" | tee "$LOG_FILE"
echo "CAN-IDS Batch Testing - Set 01" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Array of CSV files to test
CSV_FILES=(
    "attack-free-1.csv"
    "attack-free-2.csv"
    "DoS-1.csv"
    "DoS-2.csv"
    "accessory-1.csv"
    "accessory-2.csv"
    "force-neutral-1.csv"
    "force-neutral-2.csv"
    "rpm-1.csv"
    "rpm-2.csv"
    "standstill-1.csv"
    "standstill-2.csv"
)

# Summary arrays
declare -a TEST_NAMES
declare -a TEST_RESULTS
declare -a TEST_DURATIONS
declare -a TEST_MESSAGES
declare -a TEST_THROUGHPUT
declare -a TEST_CPU
declare -a TEST_PRECISION
declare -a TEST_RECALL

# Run tests
TOTAL_TESTS=${#CSV_FILES[@]}
CURRENT_TEST=0

for csv_file in "${CSV_FILES[@]}"; do
    CURRENT_TEST=$((CURRENT_TEST + 1))
    
    echo "" | tee -a "$LOG_FILE"
    echo "======================================" | tee -a "$LOG_FILE"
    echo "Test $CURRENT_TEST/$TOTAL_TESTS: $csv_file" | tee -a "$LOG_FILE"
    echo "======================================" | tee -a "$LOG_FILE"
    
    # Extract test name (remove extension)
    TEST_NAME="${csv_file%.csv}"
    OUTPUT_DIR="$OUTPUT_BASE/${TEST_NAME}"
    
    # Run test
    START_TIME=$(date +%s)
    
    if python "$PROJECT_DIR/scripts/comprehensive_test.py" \
        "$DATA_DIR/$csv_file" \
        --output "$OUTPUT_DIR" \
        --rules "$RULES_FILE" 2>&1 | tee -a "$LOG_FILE"; then
        
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        echo "✓ Test completed successfully in ${DURATION}s" | tee -a "$LOG_FILE"
        
        # Extract metrics from JSON
        SUMMARY_JSON="$OUTPUT_DIR/comprehensive_summary.json"
        if [ -f "$SUMMARY_JSON" ]; then
            MESSAGES=$(jq -r '.performance.messages_processed' "$SUMMARY_JSON")
            THROUGHPUT=$(jq -r '.performance.throughput_msg_per_sec' "$SUMMARY_JSON")
            CPU_AVG=$(jq -r '.system.cpu_percent.mean' "$SUMMARY_JSON")
            PRECISION=$(jq -r '.performance.detection_accuracy.precision' "$SUMMARY_JSON")
            RECALL=$(jq -r '.performance.detection_accuracy.recall' "$SUMMARY_JSON")
            
            TEST_NAMES+=("$TEST_NAME")
            TEST_RESULTS+=("PASS")
            TEST_DURATIONS+=("$DURATION")
            TEST_MESSAGES+=("$MESSAGES")
            TEST_THROUGHPUT+=("$THROUGHPUT")
            TEST_CPU+=("$CPU_AVG")
            TEST_PRECISION+=("$PRECISION")
            TEST_RECALL+=("$RECALL")
        fi
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        echo "✗ Test failed after ${DURATION}s" | tee -a "$LOG_FILE"
        
        TEST_NAMES+=("$TEST_NAME")
        TEST_RESULTS+=("FAIL")
        TEST_DURATIONS+=("$DURATION")
        TEST_MESSAGES+=("0")
        TEST_THROUGHPUT+=("0")
        TEST_CPU+=("0")
        TEST_PRECISION+=("0")
        TEST_RECALL+=("0")
    fi
done

# Generate summary report
echo "" | tee "$SUMMARY_FILE"
echo "======================================" | tee -a "$SUMMARY_FILE"
echo "CAN-IDS Batch Test Summary - Set 01" | tee -a "$SUMMARY_FILE"
echo "Completed: $(date)" | tee -a "$SUMMARY_FILE"
echo "======================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

printf "%-25s %-10s %-10s %-12s %-12s %-10s %-12s %-12s\n" \
    "Test Name" "Result" "Duration" "Messages" "Throughput" "CPU %" "Precision" "Recall" | tee -a "$SUMMARY_FILE"
printf "%-25s %-10s %-10s %-12s %-12s %-10s %-12s %-12s\n" \
    "-------------------------" "----------" "----------" "------------" "------------" "----------" "------------" "------------" | tee -a "$SUMMARY_FILE"

for i in "${!TEST_NAMES[@]}"; do
    printf "%-25s %-10s %-10ss %-12s %-12.2f %-10.2f %-12.4f %-12.4f\n" \
        "${TEST_NAMES[$i]}" \
        "${TEST_RESULTS[$i]}" \
        "${TEST_DURATIONS[$i]}" \
        "${TEST_MESSAGES[$i]}" \
        "${TEST_THROUGHPUT[$i]}" \
        "${TEST_CPU[$i]}" \
        "${TEST_PRECISION[$i]}" \
        "${TEST_RECALL[$i]}" | tee -a "$SUMMARY_FILE"
done

echo "" | tee -a "$SUMMARY_FILE"
echo "All test results saved to: $OUTPUT_BASE" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Count results
PASSED=$(echo "${TEST_RESULTS[@]}" | tr ' ' '\n' | grep -c "PASS" || echo 0)
FAILED=$(echo "${TEST_RESULTS[@]}" | tr ' ' '\n' | grep -c "FAIL" || echo 0)

echo "Summary: $PASSED passed, $FAILED failed out of $TOTAL_TESTS tests" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

echo "======================================" | tee -a "$LOG_FILE"
echo "Batch testing complete!" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"

cat "$SUMMARY_FILE"
