# AI-Powered Git Rebase Assistant

This tool leverages the Google Gemini API to assist with interactive Git rebases by suggesting `fixup` and `reword` operations.

## Installation

```bash
mkdir -p ~/bin && curl -fLRo ~/bin/ai-rebase.sh https://raw.githubusercontent.com/senomas/git-auto-rebase/refs/heads/master/ai-rebase.sh && chmod +x ~/bin/ai-rebase.sh
```

## Features

- **AI Rebase Assistance (Fixup & Reword):**
  - Analyzes a range of commits (compared to an upstream branch).
  - Suggests `fixup` operations for commits that appear to be minor corrections to their predecessors.
  - Suggests improved commit messages (`reword` operations) for commits that don't follow conventions or lack clarity.
  - Can optionally attempt to perform the rebase automatically in two phases (fixup first, then reword).
- **Interactive Confirmation:** Prompts for user confirmation before attempting automatic rebases.
- **Contextual Awareness (Rebase):** Can request content of specific files _at specific commits_ (with user permission) to improve fixup suggestions.
- **Safety:** Creates backup branches before attempting automatic rebases. Handles automatic abort on failure.
- **Dockerized:** Runs in a Docker container for a consistent environment.

## Prerequisites

- **Git:** Must be installed on your host machine.
- **Docker:** Must be installed and running on your host machine.
- **Gemini API Key:** You need an API key from Google AI Studio or Google Cloud.

## Setup

1.  **Build the Docker Image:**
    Build manually:

    ```bash
    docker build -t docker.senomas.com/git-rebase:1.0 .
    ```

    _(Note: The tag `docker.senomas.com/git-rebase:1.0` is used by default in `ai-rebase.sh`. You can override it using the `DOCKER_IMAGE_NAME` environment variable if needed, e.g., `export DOCKER_IMAGE_NAME=my-custom-image:latest`.)_

2.  **Set Environment Variables:**
    Export your Gemini API key:
    ```bash
    export GEMINI_API_KEY='YOUR_API_KEY'
    ```
    Optionally, set the Gemini model to use (defaults to `gemini-1.5-flash-latest` if not set):
    ```bash
    export GEMINI_MODEL='gemini-1.5-pro-latest' # Or another compatible model
    ```
    _(Replace `YOUR_API_KEY` with your actual key. Add these lines to your shell profile (e.g., `.bashrc`, `.zshrc`, `config.fish`) for persistence.)_

## Usage

Run the `ai-rebase.sh` script directly.

### Using `ai-rebase.sh` Directly

The `ai-rebase.sh` script runs the `git_rebase_ai.py` script inside a Docker container to perform the analysis and rebase operations.

```bash
# Compare against upstream/master (default) and attempt auto-rebase (fixup then reword)
./ai-rebase.sh

# Compare against origin/develop and attempt auto-rebase
./ai-rebase.sh origin/develop

# Compare against upstream/main, show instructions only (no auto-rebase)
./ai-rebase.sh --instruct

# Compare against origin/develop, show instructions only
./ai-rebase.sh origin/develop --instruct

# Delete backup branches created by the tool for the current branch
./ai-rebase.sh --delete-backups
```

### Workflow Details

1.  **Rebase (Fixup & Reword):**

    - Ensure your working directory is clean (`git status`).
    - Ensure your branch is up-to-date with the remote if necessary.
    - Run `./ai-rebase.sh [upstream_ref] [--instruct] [--delete-backups]`.
    - The tool analyzes commits between the `upstream_ref` (defaults to `upstream/main`) and your current branch (`HEAD`).
    - **Phase 1: Fixup Suggestions**
      - It identifies commits that look like small fixes to their immediate predecessor.
      - It may ask for permission to read specific files _at specific commits_ for better context.
      - It prints the suggested `fixup` operations.
      - **If `--instruct` is NOT used:**
        - It creates a backup branch (e.g., `your-branch-backup-TIMESTAMP`).
        - It asks for confirmation to attempt the automatic fixup rebase.
        - If confirmed, it runs `git rebase -i` non-interactively using a generated script to apply the fixups.
        - If the automatic rebase fails (e.g., conflicts), it attempts to abort and provides instructions for manual rebase, stopping the process.
        - If successful, it proceeds to Phase 2.
      - **If `--instruct` IS used:**
        - It prints the fixup suggestions and proceeds directly to Phase 2 suggestions without attempting any rebase.
    - **Phase 2: Reword Suggestions** (Runs only if Phase 1 was successful or `--instruct` was used)
      - It analyzes the (potentially already fixup-rebased) commits between the `upstream_ref` and `HEAD`.
      - It identifies commits with messages that could be improved (clarity, convention adherence).
      - It generates a complete new commit message (subject and body) for each suggested reword.
      - It prints the suggested `reword` operations with the new messages.
      - **If `--instruct` is NOT used (and Phase 1 auto-fixup succeeded):**
        - It asks for confirmation to attempt the automatic reword rebase.
        - If confirmed, it runs `git rebase -i` non-interactively using generated scripts to mark commits for reword and supply the new messages.
        - If the automatic reword rebase fails, it attempts to abort and provides instructions for manual rebase.
      - **If `--instruct` IS used:**
        - It prints instructions on how to perform the rebase manually using both the fixup (from Phase 1) and reword suggestions.

## How it Works

1.  The `ai-rebase.sh` script is the main entry point.
2.  It runs the `git_rebase_ai.py` script inside the specified Docker container (`docker.senomas.com/git-rebase:1.0` by default).
3.  It mounts the current directory (`pwd`) as `/repo`, `.gitconfig`, and `.git-credentials` (read-only) into the container.
4.  It passes the `GEMINI_API_KEY` and `GEMINI_MODEL` environment variables into the container.
5.  It forwards relevant arguments (like `upstream_ref`, `--instruct`, `--delete-backups`) to the Python script.
6.  **`git_rebase_ai.py`:**
    - Parses arguments (`upstream_ref`, `--instruct`, `--delete-backups`).
    - Checks for clean Git status.
    - Handles `--delete-backups` if present.
    - Finds merge base (`git merge-base`).
    - **Fixup Phase:**
      - Gets commits, file structure, and diff for the range.
      - Interacts with Gemini API, potentially requesting file content at specific commits (`REQUEST_FILES: [...]`), to suggest `FIXUP: ... INTO ...` lines.
      - Parses suggestions.
      - If not `--instruct`, creates backup, confirms, and attempts automatic fixup rebase using `GIT_SEQUENCE_EDITOR` script. Handles potential failures and aborts. If successful, proceeds.
      - If `--instruct`, stores suggestions and proceeds.
    - **Reword Phase:** (If fixup succeeded or `--instruct`)
      - Re-gathers commits and diff for the (potentially modified) range.
      - Interacts with Gemini API to suggest `REWORD: ... NEW_MESSAGE: ... END_MESSAGE` blocks.
      - Parses suggestions.
      - If not `--instruct`, confirms, and attempts automatic reword rebase using `GIT_SEQUENCE_EDITOR` (changes `pick` to `r`) and `GIT_EDITOR` (supplies new message via env var) scripts. Handles potential failures and aborts.
      - If `--instruct`, prints combined manual instructions (fixup + reword).

## Files

- `ai-rebase.sh`: Main entry script that runs the Python script in Docker.
- `git_rebase_ai.py`: Handles AI rebase fixup and reword suggestions and automation.
- `Dockerfile`: Defines the Docker image environment.
- `README.md`: This file.
