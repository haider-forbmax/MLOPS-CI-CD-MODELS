FROM ghcr.io/actions/actions-runner:latest

USER root

# Install Kaniko
RUN curl -L -o /usr/local/bin/kaniko-executor \
    https://github.com/GoogleContainerTools/kaniko/releases/download/v1.23.0/executor_linux_amd64 \
    && chmod +x /usr/local/bin/kaniko-executor

# Create kaniko directory
RUN mkdir -p /kaniko/.docker && chmod -R 777 /kaniko

# Reset entrypoint so Kubernetes can control the command
USER root
RUN echo '#!/bin/bash\nexec "$@"' > /reset-entrypoint.sh && chmod +x /reset-entrypoint.sh

USER runner

# Override the parent image's entrypoint
ENTRYPOINT []
CMD []
