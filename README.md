# AI-Powered Git Assistant

This tool leverages the Google Gemini API to assist with common Git tasks, including generating commit messages, suggesting `fixup` operations for interactive rebases, and suggesting `reword` operations with improved commit messages.

## Features

- **AI Commit Message Generation:** Analyzes staged Git diffs and generates conventional commit messages (subject and body). Supports amending previous commits.
- **AI Rebase Fixup Suggestions:** Analyzes a range of commits (compared to an upstream branch) and suggests `fixup` operations for commits that appear to be minor corrections to their predecessors. Can optionally attempt to perform the rebase automatically.
- **AI Rebase Reword Suggestions:** Analyzes a range of commits and suggests improved commit messages (`reword` operations) for those that don't follow conventions or lack clarity. Can optionally attempt to perform the rebase automatically.
- **Interactive Confirmation:** Prompts for user confirmation before executing Git commands like `commit` or `rebase`.
- **Contextual Awareness (Commit):** Can request content of specific project files (with user permission) to generate more accurate commit messages.
- **Contextual Awareness (Rebase):** Can request content of specific files *at specific commits* (with user permission) to improve fixup suggestions.
- **Safety:** Creates backup branches before attempting automatic rebases.
- **Dockerized:** Runs in a Docker container for a consistent environment.
- **Makefile:** Provides convenient shortcuts for common tasks.

## Prerequisites

- **Git:** Must be installed on your host machine.
- **Docker:** Must be installed and running on your host machine.
- **Gemini API Key:** You need an API key from Google AI Studio or Google Cloud.
- **Make (Optional):** For using the Makefile shortcuts.

## Setup

1.  **Build the Docker Image:**
    You can use the provided Makefile:
    ```bash
    make build
    ```
    Or build manually:
    ```bash
    docker build -t docker.senomas.com/commit:1.0 .
    ```
    _(Note: The tag `docker.senomas.com/commit:1.0` is used by default. You can override it using the `DOCKER_IMAGE_NAME` environment variable if needed, e.g., `export DOCKER_IMAGE_NAME=my-custom-image:latest`.)_

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

You can use the `Makefile` for convenience or run the `ai-commit.sh` script directly.

### Using Makefile

-   **Generate Commit Message:** Stage changes (`git add ...`), then run:
    ```bash
    make commit
    ```
    To amend the previous commit:
    ```bash
    make commit ARGS="-a"
    ```
-   **Suggest Rebase Fixups:** Compares current branch to `upstream/main` by default.
    ```bash
    make rebase
    ```
    To compare against a different upstream reference and only show instructions (no auto-rebase):
    ```bash
    make rebase UPSTREAM_REF="origin/develop" ARGS="--instruct"
    ```
-   **Suggest Rebase Rewords:** Compares current branch to `upstream/main` by default.
    ```bash
    make reword
    ```
    To compare against a different upstream reference and only show instructions (no auto-rebase):
    ```bash
    make reword UPSTREAM_REF="origin/develop" ARGS="--instruct"
    ```

### Using `ai-commit.sh` Directly

The `ai-commit.sh` script acts as the main entry point, selecting the appropriate Python script based on flags.

-   **Generate Commit Message (Default):**
    Stage changes (`git add ...`), then run:
    ```bash
    ./ai-commit.sh
    ```
    To amend the previous commit:
    ```bash
    ./ai-commit.sh -a
    # or
    ./ai-commit.sh --amend
    ```
-   **Suggest Rebase Fixups (`-r` flag):**
    ```bash
    # Compare against upstream/main (default) and attempt auto-rebase
    ./ai-commit.sh -r

    # Compare against origin/develop and attempt auto-rebase
    ./ai-commit.sh -r origin/develop

    # Compare against upstream/main, show instructions only (no auto-rebase)
    ./ai-commit.sh -r --instruct

    # Compare against origin/develop, show instructions only
    ./ai-commit.sh -r origin/develop --instruct
    ```
-   **Suggest Rebase Rewords (`-w` flag):**
    ```bash
    # Compare against upstream/main (default) and attempt auto-reword
    ./ai-commit.sh -w

    # Compare against origin/develop and attempt auto-reword
    ./ai-commit.sh -w origin/develop

    # Compare against upstream/main, show instructions only (no auto-reword)
    ./ai-commit.sh -w --instruct

    # Compare against origin/develop, show instructions only
    ./ai-commit.sh -w origin/develop --instruct
    ```

### Workflow Details

1.  **Commit:**
    - Stage your changes (`git add ...`).
    - Run `make commit` or `./ai-commit.sh` (optionally with `-a`).
    - The tool analyzes the staged diff.
    - It may ask for permission to read specific project files for better context.
    - It generates a commit message.
    - It prints the message and asks for confirmation (`y/n`).
    - If confirmed, it runs `git commit` (or `git commit --amend`).

2.  **Rebase Fixup:**
    - Ensure your branch is up-to-date with the remote if necessary.
    - Run `make rebase [UPSTREAM_REF=...] [ARGS=...]` or `./ai-commit.sh -r [upstream_ref] [--instruct]`.
    - The tool analyzes commits between the `upstream_ref` (e.g., `upstream/main`) and your current branch (`HEAD`).
    - It identifies commits that look like small fixes to their immediate predecessor.
    - It may ask for permission to read specific files *at specific commits* for better context.
    - It prints the suggested `fixup` operations.
    - **If `--instruct` is NOT used:**
        - It creates a backup branch (e.g., `your-branch-backup-TIMESTAMP`).
        - It asks for confirmation to attempt the automatic rebase.
        - If confirmed, it runs `git rebase -i` non-interactively using a generated script to apply the fixups.
        - If the automatic rebase fails (e.g., conflicts), it attempts to abort and provides instructions for manual rebase.
    - **If `--instruct` IS used:**
        - It prints instructions on how to perform the rebase manually using the suggestions.

3.  **Rebase Reword:**
    - Ensure your branch is up-to-date.
    - Run `make reword [UPSTREAM_REF=...] [ARGS=...]` or `./ai-commit.sh -w [upstream_ref] [--instruct]`.
    - The tool analyzes commits between the `upstream_ref` and `HEAD`.
    - It identifies commits with messages that could be improved (clarity, convention adherence).
    - It generates a complete new commit message (subject and body) for each suggested reword.
    - It prints the suggested `reword` operations with the new messages.
    - **If `--instruct` is NOT used:**
        - It creates a backup branch.
        - It asks for confirmation to attempt the automatic rebase.
        - If confirmed, it runs `git rebase -i` non-interactively using generated scripts to mark commits for reword and supply the new messages.
        - If the automatic rebase fails, it attempts to abort and provides instructions for manual rebase.
    - **If `--instruct` IS used:**
        - It prints instructions on how to perform the rebase manually, including the suggested new messages to copy.

## How it Works

1.  The `Makefile` provides simple targets that call `ai-commit.sh` with appropriate flags and arguments.
2.  The `ai-commit.sh` script determines which Python script to run based on the presence of `-r` (rebase fixup) or `-w` (rebase reword) flags. If neither is present, it defaults to commit message generation.
3.  It runs the selected Python script (`git_commit_ai.py`, `git_rebase_ai.py`, or `git_reword_ai.py`) inside the specified Docker container (`docker.senomas.com/commit:1.0` by default).
4.  It mounts the current directory (`pwd`) as `/repo`, `.gitconfig`, and `.git-credentials` (read-only) into the container.
5.  It passes the `GEMINI_API_KEY` and `GEMINI_MODEL` environment variables into the container.
6.  It forwards relevant arguments (like `-a`, `--amend`, `upstream_ref`, `--instruct`) to the Python script.
7.  **`git_commit_ai.py`:**
    - Parses arguments (`--amend`).
    - Gets staged diff (`git diff --staged` or `git diff HEAD~1 --staged`).
    - Gets project file list (`git ls-tree`).
    - Interacts with Gemini API, potentially requesting file content (`Request content for file: ...`), to generate a commit message.
    - Prompts user for confirmation.
    - Runs `git commit`.
8.  **`git_rebase_ai.py`:**
    - Parses arguments (`upstream_ref`, `--instruct`).
    - Finds merge base (`git merge-base`).
    - Gets commits, file structure, and diff for the range.
    - Interacts with Gemini API, potentially requesting file content at specific commits (`REQUEST_FILES: [...]`), to suggest `FIXUP: ... INTO ...` lines.
    - Parses suggestions.
    - If not `--instruct`, creates backup, confirms, and attempts automatic rebase using `GIT_SEQUENCE_EDITOR` script to change `pick` to `f`. Handles potential failures and aborts.
    - If `--instruct`, prints manual instructions.
9.  **`git_reword_ai.py`:**
    - Parses arguments (`upstream_ref`, `--instruct`).
    - Finds merge base.
    - Gets commits and diff for the range.
    - Interacts with Gemini API to suggest `REWORD: ... NEW_MESSAGE: ... END_MESSAGE` blocks.
    - Parses suggestions.
    - If not `--instruct`, creates backup, confirms, and attempts automatic rebase using `GIT_SEQUENCE_EDITOR` (changes `pick` to `r`) and `GIT_EDITOR` (supplies new message via env var) scripts. Handles potential failures and aborts.
    - If `--instruct`, prints manual instructions with new messages.

## Files

- `ai-commit.sh`: Main entry script, selects Python script based on flags.
- `git_commit_ai.py`: Handles AI commit message generation.
- `git_rebase_ai.py`: Handles AI rebase fixup suggestions and automation.
- `git_reword_ai.py`: Handles AI rebase reword suggestions and automation.
- `Dockerfile`: Defines the Docker image environment.
- `Makefile`: Provides convenience targets (`build`, `commit`, `rebase`, `reword`).
- `README.md`: This file.
