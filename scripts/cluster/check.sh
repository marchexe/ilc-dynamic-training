#!/usr/bin/env bash
# check.sh — Query all GPU nodes and report availability.
#
# Usage:
#   ./scripts/cluster/check.sh                     # full report
#   ./scripts/cluster/check.sh --recommend-test    # best single free GPU (highest perf score)
#   ./scripts/cluster/check.sh --recommend-training  # cluster with highest total perf score
#
# A GPU is FREE if: memory used < 500 MiB AND utilization < 5%.
# BUSY GPUs display the usernames of active compute processes (e.g. "BUSY (alice, bob)").
#
# GPU performance scores (FP16 tensor TFLOPS, rounded):
#   A100-SXM4       → 10  (NVLink interconnect bonus)
#   A100-PCIe-80GB  →  9  (largest VRAM)
#   A100-PCIe-40GB  →  8
#   RTX A6000       →  4
#   RTX 3090        →  4
#   RTX 4000 Ada    →  3
#   V100S           →  2
#   V100            →  1
#   other/unknown   →  0
#
# Training heuristic: max Σ score(free GPU) across all nodes.
# Test heuristic:     max score among all individual free GPUs.

set -euo pipefail

FREE_MEM_THRESHOLD_MiB=500
FREE_UTIL_THRESHOLD_PCT=5
MODE="${1:-full}"

case "$MODE" in
    full|--recommend-test|--recommend-training) ;;
    *)
        echo "Error: unknown argument '$MODE'"
        echo "Valid modes are: full | --recommend-test | --recommend-training"
        exit 2
        ;;
esac

read -r -a NODES <<< "${ML_PFA_CLUSTER_NODES:-iutgpu01 iutgpu02 iutgpu03 iutgpu04 iutgpu05 iutgpu06 iutgpu07}"
SSH_BATCH_MODE=yes
if [[ "${ML_PFA_ALLOW_PASSWORD:-0}" == "1" ]]; then
    SSH_BATCH_MODE=no
fi
SSH_OPTS=(-o ConnectTimeout=5 -o ConnectionAttempts=1 -o BatchMode="$SSH_BATCH_MODE" -o StrictHostKeyChecking=no)
if [[ -n "${ML_PFA_SSH_CONFIG:-}" && -f "${ML_PFA_SSH_CONFIG}" ]]; then
    SSH_OPTS=(-F "${ML_PFA_SSH_CONFIG}" "${SSH_OPTS[@]}")
fi
LOCAL_HOST_SHORT="$(hostname -s 2>/dev/null || hostname || true)"
LOCAL_HOST_FQDN="$(hostname -f 2>/dev/null || hostname || true)"

declare -A NODE_FREE_GPUS    # node -> space-separated free GPU indices
declare -A NODE_FREE_DETAIL  # node -> "idx:score ..." for each free GPU
declare -A NODE_TOTAL_SCORE  # node -> sum of scores of free GPUs

# Return a numeric performance score for a GPU model name (from nvidia-smi).
gpu_score() {
    local name="$1"
    case "$name" in
        *A100*SXM*)  echo 10 ;;
        *A100*80*)   echo 9  ;;
        *A100*)      echo 8  ;;
        *A6000*)     echo 4  ;;
        *3090*)      echo 4  ;;
        *4000*Ada*)  echo 3  ;;
        *V100S*)     echo 2  ;;
        *V100*)      echo 1  ;;
        *)           echo 0  ;;
    esac
}

query_node() {
    local node="$1"
    local result
    local error_file
    error_file="$(mktemp)"
    local query_script='
gpu_data=$(nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,uuid --format=csv,noheader,nounits 2>/dev/null) || exit 1
printf "%s\n" "$gpu_data"
echo "---PROCS---"
nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>/dev/null \
  | while IFS="," read -r pid uuid; do
        pid=$(echo "$pid" | tr -d " ")
        uuid=$(echo "$uuid" | tr -d " ")
        [ -z "$pid" ] && continue
        user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d " ")
        [ -z "$user" ] && user="?"
        printf "%s|%s\n" "$uuid" "$user"
    done
echo "---END---"
'
    # Single SSH call: GPU stats (with UUID) + process→user mapping.
    # Output format:
    #   <index>, <name>, <mem_used>, <util>, <uuid>       (one line per GPU)
    #   ---PROCS---
    #   <gpu_uuid>|<username>                              (one line per compute process)
    #   ---END---
    if [[ "$node" == "$LOCAL_HOST_SHORT" || "$node" == "$LOCAL_HOST_FQDN" || "$node" == "localhost" ]]; then
        result=$(bash -c "$query_script" 2>"$error_file") || {
            [[ "$MODE" == "full" ]] && echo "  [UNREACHABLE]"
            if [[ "${ML_PFA_DEBUG:-0}" == "1" && -s "$error_file" ]]; then
                sed 's/^/    /' "$error_file"
            fi
            rm -f "$error_file"
            return 0
        }
    else
        if [[ "${ML_PFA_ALLOW_PASSWORD:-0}" == "1" ]]; then
            result=$(ssh "${SSH_OPTS[@]}" "$node" "$query_script") || {
                [[ "$MODE" == "full" ]] && echo "  [UNREACHABLE]"
                rm -f "$error_file"
                return 0
            }
        else
            result=$(ssh "${SSH_OPTS[@]}" "$node" "$query_script" 2>"$error_file") || {
                [[ "$MODE" == "full" ]] && echo "  [UNREACHABLE]"
                if [[ "${ML_PFA_DEBUG:-0}" == "1" && -s "$error_file" ]]; then
                    sed 's/^/    /' "$error_file"
                fi
                rm -f "$error_file"
                return 0
            }
        fi
    fi
    rm -f "$error_file"

    # Split output into GPU-stats section and process section.
    local gpu_section procs_section
    gpu_section=$(printf '%s\n' "$result" | awk '/^---PROCS---/{exit} 1')
    procs_section=$(printf '%s\n' "$result" | awk '/^---PROCS---/{found=1;next} /^---END---/{exit} found{print}')

    # Build gpu_uuid → deduplicated username list (e.g. "alice, bob").
    declare -A uuid_users
    while IFS='|' read -r uuid user; do
        [[ -z "$uuid" ]] && continue
        local existing="${uuid_users[$uuid]:-}"
        if [[ -z "$existing" ]]; then
            uuid_users[$uuid]="$user"
        elif [[ "$existing" != *"$user"* ]]; then
            uuid_users[$uuid]="$existing, $user"
        fi
    done <<< "$procs_section"

    local free_list="" detail_list="" total_score=0

    while IFS=',' read -r idx name mem_used util uuid; do
        idx=$(echo "$idx" | xargs)
        name=$(echo "$name" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        util=$(echo "$util" | xargs)
        uuid=$(echo "$uuid" | xargs)

        local score status
        score=$(gpu_score "$name")

        if [[ "$mem_used" -lt "$FREE_MEM_THRESHOLD_MiB" && "$util" -lt "$FREE_UTIL_THRESHOLD_PCT" ]]; then
            status="FREE"
            free_list="$free_list $idx"
            detail_list="$detail_list ${idx}:${score}"
            total_score=$(( total_score + score ))
        else
            local users_str="${uuid_users[$uuid]:-}"
            if [[ -n "$users_str" ]]; then
                status="BUSY ($users_str)"
            else
                status="BUSY"
            fi
        fi

        [[ "$MODE" == "full" ]] && \
            printf "    GPU %s | %-32s | mem: %5s MiB | util: %3s %% | score: %2s | %s\n" \
                "$idx" "$name" "$mem_used" "$util" "$score" "$status"
    done <<< "$gpu_section"

    NODE_FREE_GPUS["$node"]="${free_list# }"
    NODE_FREE_DETAIL["$node"]="${detail_list# }"
    NODE_TOTAL_SCORE["$node"]="$total_score"
}

[[ "$MODE" == "full" ]] && echo "========================================"
[[ "$MODE" == "full" ]] && echo "  GPU Cluster Status"
[[ "$MODE" == "full" ]] && echo "========================================"

for node in "${NODES[@]}"; do
    [[ "$MODE" == "full" ]] && echo "" && echo "--- $node ---"
    query_node "$node"
done

if [[ "$MODE" == "--recommend-test" ]]; then
    # Single GPU: pick the free GPU with the highest performance score.
    best_node=""
    best_gpu=""
    best_score=-1
    for node in "${NODES[@]}"; do
        detail="${NODE_FREE_DETAIL[$node]:-}"
        [[ -z "$detail" ]] && continue
        for pair in $detail; do
            idx="${pair%%:*}"
            score="${pair##*:}"
            if [[ "$score" -gt "$best_score" ]]; then
                best_score=$score
                best_node=$node
                best_gpu=$idx
            fi
        done
    done
    if [[ -z "$best_node" ]]; then echo "NONE"; exit 1; fi
    echo "${best_node}:${best_gpu}"
    exit 0
fi

if [[ "$MODE" == "--recommend-training" ]]; then
    # Training: node with the highest total performance score across all free GPUs.
    best_node=""
    best_score=-1
    best_gpus=""
    for node in "${NODES[@]}"; do
        free="${NODE_FREE_GPUS[$node]:-}"
        [[ -z "$free" ]] && continue
        total="${NODE_TOTAL_SCORE[$node]:-0}"
        if [[ "$total" -gt "$best_score" ]]; then
            best_score=$total
            best_node=$node
            best_gpus=$(echo "$free" | tr ' ' ',')
        fi
    done
    if [[ -z "$best_node" ]]; then echo "NONE"; exit 1; fi
    echo "${best_node}:${best_gpus}"
    exit 0
fi

# Full report summary
echo ""
echo "========================================"
echo "  Summary — Free GPUs per node"
echo "========================================"
for node in "${NODES[@]}"; do
    free="${NODE_FREE_GPUS[$node]:-}"
    count=0; [[ -n "$free" ]] && count=$(echo "$free" | wc -w | tr -d ' ')
    total="${NODE_TOTAL_SCORE[$node]:-0}"
    printf "  %-12s  %d free GPU(s)  total score: %3d  [%s]\n" "$node" "$count" "$total" "$free"
done
