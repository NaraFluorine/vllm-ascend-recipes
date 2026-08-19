#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo "run.sh is configured through MULTI_NODE_* environment variables" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
: "${MULTI_NODE_PLAN:?MULTI_NODE_PLAN is required}"

cd "$REPOSITORY_ROOT"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    set +u
    # shellcheck source=/dev/null
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
fi

if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    set +u
    # shellcheck source=/dev/null
    source /usr/local/Ascend/nnal/atb/set_env.sh
    set -u
fi

if [[ "${MULTI_NODE_VALIDATE_ONLY:-false}" == "true" ]]; then
    exec python3 -u "$SCRIPT_DIR/runner.py" \
        --plan "$MULTI_NODE_PLAN" \
        --validate-only
fi

: "${MULTI_NODE_NODE_INDEX:?MULTI_NODE_NODE_INDEX is required}"
: "${MULTI_NODE_CLUSTER_IPS:?MULTI_NODE_CLUSTER_IPS is required}"
if [[ ! "$MULTI_NODE_NODE_INDEX" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "MULTI_NODE_NODE_INDEX must be a non-negative integer" >&2
    exit 1
fi

node_count=$(PLAN_PATH="$MULTI_NODE_PLAN" \
    PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
import os
from pathlib import Path

from plan import load_plan

print(len(load_plan(Path(os.environ["PLAN_PATH"])).nodes))
PY
)
if [[ -n "${MULTI_NODE_NODE_COUNT:-}" && "$MULTI_NODE_NODE_COUNT" != "$node_count" ]]; then
    echo "MULTI_NODE_NODE_COUNT does not match plan.nodes: $MULTI_NODE_NODE_COUNT != $node_count" >&2
    exit 1
fi
export MULTI_NODE_NODE_COUNT=$node_count

if ((MULTI_NODE_NODE_INDEX < 0 || MULTI_NODE_NODE_INDEX >= node_count)); then
    echo "MULTI_NODE_NODE_INDEX is outside the plan node range: $MULTI_NODE_NODE_INDEX" >&2
    exit 1
fi
node_id="node${MULTI_NODE_NODE_INDEX}"
hosts_file="/tmp/multi-node-hosts-${MULTI_NODE_NODE_INDEX}.yaml"

cluster_ips=()
IFS=',' read -r -a cluster_ips <<< "$MULTI_NODE_CLUSTER_IPS"

if [[ ${#cluster_ips[@]} -ne $node_count ]]; then
    echo "MULTI_NODE_CLUSTER_IPS count does not match plan.nodes: ${#cluster_ips[@]} != $node_count" >&2
    exit 1
fi
MULTI_NODE_CLUSTER_IPS=$(IFS=,; echo "${cluster_ips[*]}")
export MULTI_NODE_CLUSTER_IPS

{
    echo "version: 1"
    echo "hosts:"
    for ((index = 0; index < node_count; index++)); do
        echo "  node${index}:"
        echo "    address: ${cluster_ips[$index]}"
        if [[ $index -eq $MULTI_NODE_NODE_INDEX && -n "${MULTI_NODE_INTERFACE:-}" ]]; then
            echo "    interface: $MULTI_NODE_INTERFACE"
        fi
    done
} > "$hosts_file"

if [[ -z "${MULTI_NODE_VISIBLE_DEVICES:-}" ]]; then
    if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
        export MULTI_NODE_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES
    elif [[ -n "${ASCEND_VISIBLE_DEVICES:-}" ]]; then
        export MULTI_NODE_VISIBLE_DEVICES=$ASCEND_VISIBLE_DEVICES
    fi
fi
echo "Multi-node framework node: index=$MULTI_NODE_NODE_INDEX id=$node_id ip=${cluster_ips[$MULTI_NODE_NODE_INDEX]}"

artifact_root=${MULTI_NODE_ARTIFACT_ROOT:-/tmp/multi-node}
# Copy low-level Ascend logs after the runner has stopped all managed services.
# shellcheck disable=SC2329  # Invoked by the EXIT trap.
collect_plogs() {
    [[ -n "${MULTI_NODE_PLOG_ROOT:-}" && -d /root/ascend/log ]] || return 0
    plog_directory="$MULTI_NODE_PLOG_ROOT/$node_id"
    mkdir -p "$plog_directory"
    cp -a /root/ascend/log/. "$plog_directory/" 2>/dev/null || true
}
trap collect_plogs EXIT

runner_pid=""
# Forward controller termination to the runner process group and let it clean up.
# shellcheck disable=SC2329  # Invoked by the TERM/INT traps.
forward_signal() {
    if [[ -n "$runner_pid" ]]; then
        kill "-$1" "$runner_pid" 2>/dev/null || true
    elif [[ $1 == TERM ]]; then
        exit 143
    else
        exit 130
    fi
}
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT

python3 -u "$SCRIPT_DIR/runner.py" \
    --plan "$MULTI_NODE_PLAN" \
    --hosts "$hosts_file" \
    --node-id "$node_id" \
    --vllm-ascend-root "${VLLM_ASCEND_ROOT:-/vllm-workspace/vllm-ascend}" \
    --control-port "${MULTI_NODE_CONTROL_PORT:-29599}" \
    --startup-timeout-seconds "${MULTI_NODE_STARTUP_TIMEOUT_SECONDS:-1800}" \
    --run-timeout-seconds "${MULTI_NODE_RUN_TIMEOUT_SECONDS:-7200}" \
    --progress-interval-seconds "${MULTI_NODE_PROGRESS_INTERVAL_SECONDS:-30}" \
    --artifact-root "$artifact_root" &
runner_pid=$!

# A signal interrupts bash's wait before the Python Runner necessarily finishes
# its own process-group cleanup. Keep the entrypoint alive until it exits.
runner_status=0
while kill -0 "$runner_pid" 2>/dev/null; do
    wait "$runner_pid" || runner_status=$?
done
wait "$runner_pid" 2>/dev/null || {
    status=$?
    if [[ $status -ne 127 ]]; then
        runner_status=$status
    fi
}
exit "$runner_status"
