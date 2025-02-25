#!/bin/bash
# Stop and delete SN1 API systemd service so it does not start at boot.
set -e

SERVICE_FILE="/etc/systemd/system/sn1api.service"

# Check if the service file exists.
if [ ! -f "${SERVICE_FILE}" ]; then
    echo "SN1 API service file not found. Nothing to do."
    exit 0
fi

# Stop the service if it's running.
sudo systemctl stop sn1api.service || true

# Disable the service from starting at boot.
sudo systemctl disable sn1api.service || true

# Remove the service file.
sudo rm -f "${SERVICE_FILE}"

# Reload systemd configuration.
sudo systemctl daemon-reload

echo "SN1 API service has been stopped and removed."