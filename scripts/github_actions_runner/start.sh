#!/usr/bin/env bash
# Modified from https://github.com/beikeni/github-runner-dockerfile/blob/main/start.sh

# Set default values for configuration
REPOSITORY=${REPOSITORY:-"volcengine/verl"}
REG_TOKEN=${REG_TOKEN:-""}
RUNNER_NAME=${RUNNER_NAME:-"verl-$(hostname)"}
RUNNER_LABELS=${RUNNER_LABELS:-""} # Delimiter is comma
RUNNER_GROUP=${RUNNER_GROUP:-"Default"}
RUNNER_WORK_DIR=${RUNNER_WORK_DIR:-"_work"}

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Error handling
set -euo pipefail
trap 'log "Error on line $LINENO"' ERR

# Check required environment variables
if [ -z "${REG_TOKEN}" ]; then
    log "ERROR: REG_TOKEN is not set"
    exit 1
fi

# Configure Docker proxy if either proxy is set
if [ -n "${http_proxy:-}" ] || [ -n "${https_proxy:-}" ]; then
    log "Configuring Docker proxy..."
    mkdir -p /home/docker/.docker
    tee /home/docker/.docker/config.json <<EOF
{
 "proxies": {
   "default": {
     "httpProxy": "${http_proxy:-${https_proxy:-}}",
     "httpsProxy": "${https_proxy:-${http_proxy:-}}",
     "noProxy": "${no_proxy:-"127.0.0.0/8"}"
   }
 }
}
EOF
fi

# Change to runner directory
cd /home/docker/actions-runner || exit 1

# Configure runner
log "Configuring runner..."
CONFIG_ARGS=(
    "--url" "https://github.com/${REPOSITORY}"
    "--token" "${REG_TOKEN}"
    "--name" "${RUNNER_NAME}"
    "--labels" "${RUNNER_LABELS}"
    "--runnergroup" "${RUNNER_GROUP}"
    "--work" "${RUNNER_WORK_DIR}"
    "--unattended"
)

./config.sh "${CONFIG_ARGS[@]}"

# Cleanup function
cleanup() {
    log "Removing runner..."
    ./config.sh remove --unattended --token "${REG_TOKEN}"
}

# Set up signal handlers
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# Start runner
log "Starting runner..."
./run.sh & wait $!