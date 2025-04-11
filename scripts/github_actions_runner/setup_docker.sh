#!/usr/bin/env bash
set -uxo pipefail
# Install: https://docs.docker.com/engine/install/debian/#install-using-the-repository

DOCKER_DATA_HOME=${DOCKER_DATA_HOME:-}

# Proxy for Docker itself
if [ -n "${http_proxy}" ]; then
    sudo mkdir -p /etc/systemd/system/docker.service.d
    sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf <<EOF
[Service]
Environment="http_proxy=${http_proxy:-${https_proxy}}"
Environment="https_proxy=${http_proxy:-${https_proxy}}"
EOF
fi

if [ -n "${DOCKER_DATA_HOME}" ]; then
  sudo mkdir -p /etc/docker
  sudo tee /etc/docker/daemon.json <<EOF
{
  "data-root": "${DOCKER_DATA_HOME}"
}
EOF
fi

if [ -n "${http_proxy}" ] || [ -n "${https_proxy}" ]; then
  mkdir -p "${HOME}"/.docker
  tee "${HOME}"/.docker/config.json <<EOF
{
 "proxies": {
   "default": {
     "httpProxy": "${http_proxy-${https_proxy}}",
     "httpsProxy": "${https_proxy-${http_proxy}}",
     "noProxy": "127.0.0.0/8"
   }
 }
}
EOF
fi

sudo systemctl daemon-reload
# sudo systemctl show docker --property ...
sudo systemctl restart docker