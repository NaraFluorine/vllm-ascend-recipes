#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${LWS_WORKER_INDEX:?LWS_WORKER_INDEX is required}"
: "${LWS_LEADER_ADDRESS:?LWS_LEADER_ADDRESS is required}"
: "${MULTI_NODE_NODE_COUNT:?MULTI_NODE_NODE_COUNT is required}"

if [[ ! "$LWS_WORKER_INDEX" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "LWS_WORKER_INDEX must be a non-negative integer" >&2
    exit 1
fi
if [[ ! "$MULTI_NODE_NODE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "MULTI_NODE_NODE_COUNT must be a positive integer" >&2
    exit 1
fi
if ((LWS_WORKER_INDEX >= MULTI_NODE_NODE_COUNT)); then
    echo "LWS_WORKER_INDEX is outside the LWS node range: $LWS_WORKER_INDEX" >&2
    exit 1
fi

# Use one deadline for all Pod-local startup work: preparing the shared
# AISBench environment and resolving every LWS member address.
startup_deadline=$((SECONDS + ${MULTI_NODE_STARTUP_TIMEOUT_SECONDS:-1800}))
# Return the remaining shared startup budget, failing once it is exhausted.
remaining_startup_seconds() {
    local remaining=$((startup_deadline - SECONDS))
    if ((remaining <= 0)); then
        echo "LWS startup timed out" >&2
        return 1
    fi
    printf '%s\n' "$remaining"
}

# Node 0 prepares the pinned AISBench installation once on the shared PVC.
# Other nodes wait for the atomically published environment file.
if [[ ${MULTI_NODE_VALIDATE_ONLY:-false} != true ]]; then
    : "${MULTI_NODE_RUN_ROOT:?MULTI_NODE_RUN_ROOT is required}"
    aisbench_environment="$MULTI_NODE_RUN_ROOT/aisbench.env"
    aisbench_failure="${aisbench_environment}.failed"
    if ((LWS_WORKER_INDEX == 0)); then
        aisbench_environment_tmp="${aisbench_environment}.tmp"
        aisbench_failure_tmp="${aisbench_failure}.tmp"
        rm -f -- "$aisbench_environment" "$aisbench_environment_tmp" \
            "$aisbench_failure" "$aisbench_failure_tmp"
        if timeout --foreground "$(remaining_startup_seconds)s" \
            bash "$SCRIPT_DIR/../install_aisbench.sh" \
            --env-file "$aisbench_environment_tmp"; then
            mv "$aisbench_environment_tmp" "$aisbench_environment"
        else
            install_status=$?
            printf 'AISBench preparation failed on node0 with exit code %s\n' \
                "$install_status" > "$aisbench_failure_tmp"
            mv "$aisbench_failure_tmp" "$aisbench_failure"
            exit "$install_status"
        fi
    else
        echo "waiting for node0 to prepare AISBench"
        while [[ ! -s "$aisbench_environment" ]]; do
            if [[ -s "$aisbench_failure" ]]; then
                cat "$aisbench_failure" >&2
                exit 1
            fi
            remaining_startup_seconds >/dev/null || exit 1
            sleep 5
        done
    fi
    # shellcheck source=/dev/null
    source "$aisbench_environment"
    export MULTI_NODE_AISBENCH_BIN MULTI_NODE_AISBENCH_CACHE_KEY MULTI_NODE_AISBENCH_SOURCE
fi

# LWS supplies the leader service DNS name. Derive the other member names from
# it so the common runner only needs a stable, ordered list of node IPs.
IFS='.' read -r leader_name group_name namespace_name _ <<< "$LWS_LEADER_ADDRESS"
if [[ -z "$leader_name" || -z "$group_name" || -z "$namespace_name" ]]; then
    echo "Invalid LWS_LEADER_ADDRESS: $LWS_LEADER_ADDRESS" >&2
    exit 1
fi

# Resolve one LWS member through cluster DNS within the shared startup budget.
resolve_ipv4() {
    local dns=$1
    local address=""
    echo "Waiting for LWS DNS: $dns" >&2
    while remaining_startup_seconds >/dev/null; do
        address=$(getent ahostsv4 "$dns" 2>/dev/null | awk 'NR == 1 {print $1}' || true)
        if [[ -n "$address" ]]; then
            printf '%s\n' "$address"
            return 0
        fi
        sleep 1
    done
    echo "Unable to resolve LWS DNS: $dns" >&2
    return 1
}

# Preserve plan node order when exporting addresses for hosts.yaml generation.
cluster_ips=()
for ((index = 0; index < MULTI_NODE_NODE_COUNT; index++)); do
    if [[ $index -eq 0 ]]; then
        dns_name=$LWS_LEADER_ADDRESS
    else
        dns_name="${leader_name}-${index}.${group_name}.${namespace_name}"
    fi
    cluster_ips+=("$(resolve_ipv4 "$dns_name")")
done

export MULTI_NODE_NODE_INDEX=$LWS_WORKER_INDEX
MULTI_NODE_CLUSTER_IPS=$(IFS=,; echo "${cluster_ips[*]}")
export MULTI_NODE_CLUSTER_IPS
# Pass only the remaining portion to the Runner so AISBench preparation, DNS,
# coordinator/service readiness, and gateway health share one startup budget.
MULTI_NODE_STARTUP_TIMEOUT_SECONDS=$(remaining_startup_seconds)
export MULTI_NODE_STARTUP_TIMEOUT_SECONDS
exec bash "$SCRIPT_DIR/../run.sh"
