# FROM ghcr.io/actions/actions-runner:latest

# USER root

# # ✅ Copy full Kaniko (best practice)
# COPY --from=gcr.io/kaniko-project/executor:v1.23.0 /kaniko /kaniko

# # Prepare directories
# RUN mkdir -p /kaniko/.docker \
#     && chown -R runner:runner /home/runner /kaniko \
#     && chmod -R 755 /home/runner \
#     && chmod +x /home/runner/run.sh || true

# USER runner

# ENTRYPOINT ["/home/runner/run.sh"]

FROM ghcr.io/actions/actions-runner:latest

USER root

# Install sudo so we can move the docker config to /kaniko/.docker during the job
RUN apt-get update && apt-get install -y sudo && \
    echo "runner ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Copy Kaniko executor and SSL certs from the official image
COPY --from=gcr.io/kaniko-project/executor:v1.23.0 /kaniko /kaniko

# Ensure the runner user owns the /kaniko directory
RUN mkdir -p /kaniko/.docker && chown -R runner:runner /kaniko

USER runner
ENTRYPOINT ["/home/runner/run.sh"]
