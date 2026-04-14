FROM ghcr.io/actions/actions-runner:latest

USER root

# ✅ Copy full Kaniko (best practice)
COPY --from=gcr.io/kaniko-project/executor:v1.23.0 /kaniko /kaniko

# Prepare directories
RUN mkdir -p /kaniko/.docker \
    && chown -R runner:runner /home/runner /kaniko \
    && chmod -R 755 /home/runner \
    && chmod +x /home/runner/run.sh || true

USER runner

ENTRYPOINT ["/home/runner/run.sh"]
