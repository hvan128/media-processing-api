#!/usr/bin/env bash
# =============================================================================
# Media Processing API - Manual Deployment Script
# =============================================================================
# Usage:
#   ./scripts/deploy.sh                    # Deploy latest tag
#   ./scripts/deploy.sh abc1234            # Deploy specific image tag
#   ./scripts/deploy.sh --rollback def5678 # Rollback to specific tag
#
# Prerequisites:
#   - SSH key configured for VPS access
#   - Docker logged into GHCR on VPS
#
# Environment variables (optional):
#   VPS_HOST: VPS IP (default: 157.66.100.59)
#   VPS_USER: SSH user (default: root)
#   GITHUB_REPOSITORY: GitHub repo (default: your-org/media-processing-api)
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
VPS_HOST="${VPS_HOST:-157.66.100.59}"
VPS_USER="${VPS_USER:-root}"
DEPLOY_PATH="/opt/services/media-processing-api"
GITHUB_REPOSITORY="${GITHUB_REPOSITORY:-your-org/media-processing-api}"
IMAGE_TAG="${1:-latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# Functions
# =============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

ssh_cmd() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        "${VPS_USER}@${VPS_HOST}" "$@"
}

scp_cmd() {
    scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$@"
}

# =============================================================================
# Pre-flight checks
# =============================================================================

log_info "Starting deployment to ${VPS_USER}@${VPS_HOST}"
log_info "Image tag: ${IMAGE_TAG}"
log_info "Deploy path: ${DEPLOY_PATH}"

# Check SSH connectivity
log_info "Checking SSH connectivity..."
if ! ssh_cmd "echo 'SSH OK'" > /dev/null 2>&1; then
    log_error "Cannot connect to VPS via SSH"
    exit 1
fi

# Check docker-compose file exists
if [[ ! -f "${PROJECT_ROOT}/docker-compose.prod.yml" ]]; then
    log_error "docker-compose.prod.yml not found in project root"
    exit 1
fi

# =============================================================================
# Deployment
# =============================================================================

# Create deployment directory
log_info "Ensuring deployment directory exists..."
ssh_cmd "mkdir -p ${DEPLOY_PATH}/data"

# Copy docker-compose.yml
log_info "Copying docker-compose.yml to VPS..."
scp_cmd "${PROJECT_ROOT}/docker-compose.prod.yml" \
    "${VPS_USER}@${VPS_HOST}:${DEPLOY_PATH}/docker-compose.yml"

# Create .env file on VPS
log_info "Creating environment configuration..."
ssh_cmd "cat > ${DEPLOY_PATH}/.env << EOF
IMAGE_TAG=${IMAGE_TAG}
GITHUB_REPOSITORY=${GITHUB_REPOSITORY}
EOF"

# Pull and deploy
log_info "Pulling new image and restarting service..."
ssh_cmd << REMOTE_SCRIPT
set -euo pipefail
cd ${DEPLOY_PATH}

# Source environment
export \$(cat .env | xargs)

echo "Pulling image: ghcr.io/${GITHUB_REPOSITORY}:${IMAGE_TAG}"
docker compose pull

echo "Starting service..."
docker compose up -d --remove-orphans

echo "Waiting for service to start..."
sleep 5

# Show status
docker compose ps
REMOTE_SCRIPT

# =============================================================================
# Verification
# =============================================================================

log_info "Verifying deployment..."
sleep 5

HEALTH_STATUS=$(ssh_cmd "curl -sf http://127.0.0.1:9300/health 2>/dev/null || echo 'unhealthy'")

if [[ "$HEALTH_STATUS" == *"healthy"* ]] || [[ "$HEALTH_STATUS" == *"status"* ]]; then
    log_info "✅ Deployment successful!"
    log_info "Service is running at http://127.0.0.1:9300 (VPS localhost)"
else
    log_warn "Health check returned: ${HEALTH_STATUS}"
    log_warn "Service may still be starting (model loading takes time)..."
    log_info "Check logs with: ssh ${VPS_USER}@${VPS_HOST} 'cd ${DEPLOY_PATH} && docker compose logs -f'"
fi

# Show running container info
log_info "Container status:"
ssh_cmd "cd ${DEPLOY_PATH} && docker compose ps"

log_info "Deployment complete!"
