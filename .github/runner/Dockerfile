# .github/runner/Dockerfile
FROM ghcr.io/actions/actions-runner:latest

USER root

# Install Kaniko
RUN curl -L -o /usr/local/bin/kaniko-executor \
    https://github.com/GoogleContainerTools/kaniko/releases/download/v1.23.0/executor_linux_amd64 \
    && chmod +x /usr/local/bin/kaniko-executor

# Ensure /home/runner directory and all runner files are correctly owned and executable
RUN mkdir -p /kaniko/.docker \
    && chown -R runner:runner /home/runner /kaniko \
    && chmod -R 755 /home/runner \
    && chmod +x /home/runner/run.sh || true

# Switch back to runner user (required by ARC)
USER runner

# Set the correct entrypoint (very important)
ENTRYPOINT ["/home/runner/run.sh"]
