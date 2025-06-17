#!/bin/bash
set -e
VERSION=1.2

echo "AI-Rebase v$VERSION"

# Check for required Git configuration files
if [ ! -f "$HOME/.gitconfig" ]; then
	>&2 echo "Error: Git configuration file not found at $HOME/.gitconfig"
	exit 1
fi

if [ ! -f "$HOME/.git-credentials" ]; then
	>&2 echo "Error: Git credentials file not found at $HOME/.git-credentials"
	exit 1
fi

docker run --rm -it \
	-v "$(pwd):/repo" \
	-v "$HOME/.gitconfig:/home/appuser/.gitconfig:ro" \
	-v "$HOME/.git-credentials:/home/appuser/.git-credentials:ro" \
	-e GEMINI_API_KEY="$GEMINI_API_KEY" \
	-e GEMINI_MODEL="${GEMINI_MODEL:-gemini-1.5-flash}" \
	-u "$(id -u):$(id -g)" \
	"senomas/git-rebase:$VERSION" "$@"
