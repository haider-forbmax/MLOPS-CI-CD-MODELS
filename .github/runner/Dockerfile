FROM ghcr.io/actions/actions-runner:latest

USER root

RUN curl -L -o /usr/local/bin/kaniko-executor \
      https://github.com/GoogleContainerTools/kaniko/releases/download/v1.23.0/executor_linux_amd64 \
    && chmod +x /usr/local/bin/kaniko-executor

RUN mkdir -p /kaniko/.docker \
    && chown -R runner:runner /kaniko \
    && chmod -R 777 /kaniko

RUN chown -R runner:runner /home/runner \
    && chmod +x /home/runner/run.sh

USER runner
