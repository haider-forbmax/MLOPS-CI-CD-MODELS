# .github/runner/Dockerfile
FROM ghcr.io/actions/actions-runner:latest

USER root

# Install Kaniko executor binary
RUN curl -L -o /usr/local/bin/kaniko-executor \
    https://github.com/GoogleContainerTools/kaniko/releases/download/v1.23.0/executor_linux_amd64 \
    && chmod +x /usr/local/bin/kaniko-executor

# Create kaniko directory with proper permissions
RUN mkdir -p /kaniko/.docker \
    && chown -R runner:runner /kaniko \
    && chmod -R 777 /kaniko

# Switch back to non-root runner user
USER runner
