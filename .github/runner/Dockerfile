FROM ghcr.io/actions/actions-runner:latest

USER root

# Install Kaniko executor
RUN curl -L -o /usr/local/bin/kaniko-executor \
    https://github.com/GoogleContainerTools/kaniko/releases/download/v1.23.0/executor_linux_amd64 \
    && chmod +x /usr/local/bin/kaniko-executor

# Create kaniko dirs with proper permissions
RUN mkdir -p /kaniko/.docker \
    && chown -R runner:runner /kaniko \
    && chmod -R 777 /kaniko   # Kaniko often needs this for config.json

# Optional: install useful tools if your workflows need them (git, etc. are already in base image)
# RUN apt-get update && apt-get install -y --no-install-recommends ... && rm -rf /var/lib/apt/lists/*

USER runner
