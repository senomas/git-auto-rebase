#!/bin/bash
set -e
VERSION=1.1

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

if [ ! "$1" = "--delete-backups" ]; then
	IFS='/' read -r part1 part2 <<<"$1"

	if [ -z "$part1" ] || [ -z "$part2" ]; then
		echo no fetch
	else
		echo git fetch $part1 $part2
		git fetch $part1 $part2
	fi
fi

docker run --rm -it \
	-v "$(pwd):/repo" \
	-v "$HOME/.gitconfig:/home/appuser/.gitconfig:ro" \
	-v "$HOME/.git-credentials:/home/appuser/.git-credentials:ro" \
	-e GEMINI_API_KEY="$GEMINI_API_KEY" \
	-e GEMINI_MODEL="${GEMINI_MODEL:-gemini-1.5-flash}" \
	-u "$(id -u):$(id -g)" \
	"senomas/git-rebase:$VERSION" "$@"
