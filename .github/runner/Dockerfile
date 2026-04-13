FROM summerwind/actions-runner:latest

USER root

# Install Kaniko
RUN curl -L -o /usr/local/bin/kaniko-executor \
    https://github.com/GoogleContainerTools/kaniko/releases/download/v1.23.0/executor_linux_amd64 \
    && chmod +x /usr/local/bin/kaniko-executor

# Create kaniko directory with proper permissions
RUN mkdir -p /kaniko/.docker && chmod -R 777 /kaniko

# Switch back to runner user
USER runner

# Verify the entrypoint exists (for debugging)
RUN ls -la /home/runner/ || ls -la / || ls -la /actions-runner/
