# Makefile for AI Commit Tool

# === Variables ===
# Override these on the command line like: make build IMAGE_TAG=2.0
IMAGE_NAME ?= senomas/git-rebase
IMAGE_TAG  ?= 1.2
# Full image reference used in commands
FULL_IMAGE_NAME = $(IMAGE_NAME):$(IMAGE_TAG)

# Pass arguments to the run script via ARGS
# Example: make run ARGS="-a"
ARGS ?=

# === Targets ===

# Phony targets are not files
.PHONY: FORCE
# Default target when running 'make'
.DEFAULT_GOAL := run

build: ## Build the Docker image
	@echo "Building Docker image: $(FULL_IMAGE_NAME)..."
	docker build -t $(FULL_IMAGE_NAME) .
	@echo "Build complete: $(FULL_IMAGE_NAME)"
	docker push $(FULL_IMAGE_NAME)

run: build FORCE
	@echo "Running ai-commit.sh using image: $(FULL_IMAGE_NAME)..."
	@echo "Passing arguments: $(ARGS)"
	# Pass the image name via environment variable and arguments to the script
	DOCKER_IMAGE_NAME=$(FULL_IMAGE_NAME) ./ai-rebase.sh $(ARGS) origin/master

delete-backups: build FORCE
	@echo "Running ai-commit.sh using image: $(FULL_IMAGE_NAME)..."
	@echo "Passing arguments: $(ARGS)"
	# Pass the image name via environment variable and arguments to the script
	DOCKER_IMAGE_NAME=$(FULL_IMAGE_NAME) ./ai-rebase.sh $(ARGS) --delete-backups
