FROM ghcr.io/actions/actions-runner:latest

USER root

# Install Kaniko
RUN curl -L -o /usr/local/bin/kaniko-executor \
    https://github.com/GoogleContainerTools/kaniko/releases/download/v1.23.0/executor_linux_amd64 \
    && chmod +x /usr/local/bin/kaniko-executor

# Create kaniko directory
RUN mkdir -p /kaniko/.docker && chmod -R 777 /kaniko

# The official image uses /home/runner/run.sh as the command
USER runner
