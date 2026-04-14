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

# Copy full Kaniko (best practice)
COPY --from=gcr.io/kaniko-project/executor:v1.23.0 /kaniko /kaniko

# Prepare directories with correct ownership and permissions
RUN mkdir -p /kaniko/.docker \
    && chown -R runner:runner /kaniko /home/runner \
    && chmod -R 755 /kaniko \
    && chmod +x /home/runner/run.sh \
    && ls -la /kaniko  # for build-time verification

USER runner

# Keep the official entrypoint
ENTRYPOINT ["/home/runner/run.sh"]
