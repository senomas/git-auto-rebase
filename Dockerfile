# Use a Python base image (slim version to keep it small)
FROM python:3.9-slim-buster

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Set the home directory environment variable
ENV HOME=/home/appuser

# Create a non-root user and group
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup -d /home/appuser -m appuser

# Install any necessary dependencies (in this case, just the google-generativeai library)
RUN pip install --no-cache-dir google-generativeai==0.8.5

# Copy the Python script into the container
COPY git_rebase_ai.py /app/

# Make the script executable
RUN chmod +x /app/git_rebase_ai.py

WORKDIR /repo

# Change ownership of the app and repo directories
RUN chown -R appuser:appgroup /app && chown -R appuser:appgroup /repo

# Switch to the non-root user
USER appuser

# Set the entrypoint to run the script when the container starts
ENTRYPOINT ["python", "/app/git_rebase_ai.py"]

