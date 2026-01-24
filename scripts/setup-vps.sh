#!/usr/bin/env bash
# =============================================================================
# Media Processing API - VPS Initial Setup Script
# =============================================================================
# Run this ONCE on a new VPS to prepare it for deployments.
#
# Usage (run locally):
#   ./scripts/setup-vps.sh
#
# What it does:
#   1. Creates deployment directory structure
#   2. Configures Docker for GHCR access
#   3. Sets proper permissions
#   4. Installs curl for health checks
#
# Prerequisites:
#   - SSH access to VPS configured
#   - Docker installed on VPS
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
VPS_HOST="${VPS_HOST:-157.66.100.59}"
VPS_USER="${VPS_USER:-root}"
DEPLOY_PATH="/opt/services/media-processing-api"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

ssh_cmd() {
    ssh -o StrictHostKeyChecking=no "${VPS_USER}@${VPS_HOST}" "$@"
}

# =============================================================================
# Setup
# =============================================================================

log_info "Setting up VPS for Media Processing API deployments..."

ssh_cmd << 'SETUP_SCRIPT'
set -euo pipefail

DEPLOY_PATH="/opt/services/media-processing-api"

echo "Creating directory structure..."
mkdir -p "${DEPLOY_PATH}/data"
chmod 755 "${DEPLOY_PATH}"
chmod 755 "${DEPLOY_PATH}/data"

echo "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo "ERROR: Docker Compose is not installed"
    exit 1
fi

echo "Docker version: $(docker --version)"
echo "Docker Compose version: $(docker compose version)"

echo "Installing curl for health checks..."
apt-get update -qq && apt-get install -y -qq curl > /dev/null 2>&1 || true

echo "Verifying port 9300 is available..."
if netstat -tuln 2>/dev/null | grep -q ":9300 " || ss -tuln 2>/dev/null | grep -q ":9300 "; then
    echo "WARNING: Port 9300 is already in use!"
    netstat -tuln 2>/dev/null | grep ":9300 " || ss -tuln 2>/dev/null | grep ":9300 "
else
    echo "Port 9300 is available"
fi

echo ""
echo "=========================================="
echo "VPS Setup Complete!"
echo "=========================================="
echo "Deploy path: ${DEPLOY_PATH}"
echo ""
echo "Next steps:"
echo "1. Add these secrets to your GitHub repository:"
echo "   - VPS_SSH_PRIVATE_KEY: Your SSH private key"
echo "   - VPS_HOST: ${HOSTNAME:-your-vps-ip}"
echo "   - VPS_USER: $(whoami)"
echo ""
echo "2. Push to main branch to trigger deployment"
echo ""
echo "3. Or deploy manually:"
echo "   ./scripts/deploy.sh"
echo "=========================================="

SETUP_SCRIPT

log_info "VPS setup complete!"
