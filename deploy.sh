#!/bin/bash
set -e

# Configuration
IMAGE_NAME="pi-voice-assistant"
TAG="latest"
REGISTRY="ghcr.io/navedr"

echo "=== Building and pushing Docker image for ARM32v7 (Raspberry Pi 3) ==="
docker buildx build --platform linux/arm/v7 \
  -t ${REGISTRY}/${IMAGE_NAME}:${TAG} \
  --push .

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "On the Raspberry Pi, run:"
echo "  docker pull ${REGISTRY}/${IMAGE_NAME}:${TAG}"
echo "  docker run -d --restart=unless-stopped \\"
echo "    --name beans \\"
echo "    --device /dev/snd \\"
echo "    -e GROQ_API_KEY=your_groq_key \\"
echo "    -e OPENAI_API_KEY=your_openai_key \\"
echo "    ${REGISTRY}/${IMAGE_NAME}:${TAG}"
