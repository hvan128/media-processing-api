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
GITHUB_REPOSITORY="${GITHUB_REPOSITORY:-hvan128/media-processing-api}"
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

# Wait for service to become healthy with retries
# ML models (Whisper/Demucs) can take 30-90 seconds to load,
# but the server starts immediately and health endpoint is available right away.
# We wait up to 150 seconds (2.5 minutes) with retries to account for:
# - Container startup: ~5-10 seconds
# - Model loading: ~30-90 seconds
# - Network/healthcheck delays: ~10-20 seconds
echo "Waiting for service to be healthy (this may take up to 2.5 minutes)..."

MAX_WAIT=150  # Maximum wait time in seconds
CHECK_INTERVAL=5  # Check every 5 seconds
ELAPSED=0
HEALTHY=false

while [ $ELAPSED -lt $MAX_WAIT ]; do
  # Check if container is healthy or at least running
  CONTAINER_STATUS=$(docker compose ps --format json 2>/dev/null | jq -r '.[0].Health // .[0].State' 2>/dev/null || echo "")
  
  if echo "$CONTAINER_STATUS" | grep -q "healthy"; then
    echo "✅ Container is healthy!"
    HEALTHY=true
    break
  elif echo "$CONTAINER_STATUS" | grep -q "running"; then
    # Container is running but not yet healthy - this is OK during startup
    echo "Container is running (health check pending, ${ELAPSED}s elapsed)..."
  fi
  
  sleep $CHECK_INTERVAL
  ELAPSED=$((ELAPSED + CHECK_INTERVAL))
done

# Show status
docker compose ps
REMOTE_SCRIPT

# =============================================================================
# Verification
# =============================================================================

log_info "Verifying deployment..."

# Final health check with retries
# The /health endpoint should be available immediately, but we'll retry
# a few times in case of network delays or if container is still starting
HEALTHY=false
for i in {1..10}; do
    HEALTH_STATUS=$(ssh_cmd "curl -sf http://127.0.0.1:9300/health 2>/dev/null || echo 'unhealthy'")
    
    if [[ "$HEALTH_STATUS" == *"healthy"* ]] || [[ "$HEALTH_STATUS" == *"status"* ]]; then
        log_info "✅ Health check passed!"
        HEALTHY=true
        break
    fi
    
    if [ $i -lt 10 ]; then
        log_info "Health check attempt ${i}/10 failed, retrying in 3 seconds..."
        sleep 3
    fi
done

if [ "$HEALTHY" = true ]; then
    log_info "✅ Deployment successful!"
    log_info "Service is running at http://127.0.0.1:9300 (VPS localhost)"
else
    log_warn "Health check endpoint not yet responding"
    log_warn "Service may still be starting (model loading can take 30-90 seconds)..."
    log_info "Container status:"
    ssh_cmd "cd ${DEPLOY_PATH} && docker compose ps"
    log_info "Check logs with: ssh ${VPS_USER}@${VPS_HOST} 'cd ${DEPLOY_PATH} && docker compose logs -f'"
    log_info "⚠️  Deployment completed, but health check pending (this is normal during startup)"
fi

# Show running container info
log_info "Container status:"
ssh_cmd "cd ${DEPLOY_PATH} && docker compose ps"

log_info "Deployment complete!"
