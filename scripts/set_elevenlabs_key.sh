#!/bin/bash
#
# Set ElevenLabs API Key on VPS
# =============================
# This script SSH's into the VPS and writes the ElevenLabs API key
# to the config file used by the TTS service.
#
# Usage:
#   export ELEVENLABS_API_KEY="your_api_key_here"
#   ./scripts/set_elevenlabs_key.sh
#
# Environment Variables:
#   ELEVENLABS_API_KEY (required): Your ElevenLabs API key
#   VPS_HOST (optional): VPS IP or hostname (default: 157.66.100.59)
#   VPS_USER (optional): SSH username (default: root)
#

set -e

# Configuration
VPS_HOST="${VPS_HOST:-157.66.100.59}"
VPS_USER="${VPS_USER:-root}"
DEPLOY_PATH="/opt/services/media-processing-api"
CONFIG_DIR="/data/config"
KEY_FILE="${CONFIG_DIR}/elevenlabs_api_key.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}   Set ElevenLabs API Key on VPS${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Check for required environment variable
if [ -z "${ELEVENLABS_API_KEY}" ]; then
    echo -e "${RED}Error: ELEVENLABS_API_KEY environment variable is not set.${NC}"
    echo ""
    echo "Usage:"
    echo "  export ELEVENLABS_API_KEY=\"your_api_key_here\""
    echo "  ./scripts/set_elevenlabs_key.sh"
    exit 1
fi

# Validate key is not empty or just whitespace
TRIMMED_KEY=$(echo "${ELEVENLABS_API_KEY}" | xargs)
if [ -z "${TRIMMED_KEY}" ]; then
    echo -e "${RED}Error: ELEVENLABS_API_KEY is empty or contains only whitespace.${NC}"
    exit 1
fi

echo "VPS Host: ${VPS_USER}@${VPS_HOST}"
echo "Config file: ${KEY_FILE}"
echo ""

# SSH into VPS and set the key
echo -e "${YELLOW}Setting ElevenLabs API key on VPS...${NC}"

ssh "${VPS_USER}@${VPS_HOST}" bash -s << EOF
set -e

# Create config directory if it doesn't exist
mkdir -p "${CONFIG_DIR}"

# Write the API key to file
echo -n "${TRIMMED_KEY}" > "${KEY_FILE}"

# Set secure permissions (readable by owner only)
chmod 600 "${KEY_FILE}"

# Verify the file was created
if [ -f "${KEY_FILE}" ] && [ -s "${KEY_FILE}" ]; then
    echo "API key written successfully"
else
    echo "Error: Failed to write API key" >&2
    exit 1
fi
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   ElevenLabs API Key Set Successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "The API key has been saved to: ${KEY_FILE}"
    echo "The TTS endpoint is now ready to use."
    echo ""
    echo "Test the TTS endpoint:"
    echo "  curl -X POST http://${VPS_HOST}:8000/tts \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"text\":\"Hello world\",\"voice_id\":\"YOUR_VOICE_ID\"}'"
else
    echo ""
    echo -e "${RED}Failed to set API key on VPS.${NC}"
    exit 1
fi
