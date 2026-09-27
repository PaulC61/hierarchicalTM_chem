FROM ghcr.io/prefix-dev/pixi:0.41.4 AS pixi

FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

COPY --from=pixi /usr/local/bin/pixi /usr/local/bin/pixi

# Important packages
RUN apt-get update && apt-get install -y \
    vim nano git sudo make clang gcc \
    locales xdg-utils wget \
    htop tmux curl \
    libffi-dev libglib2.0-dev openssh-client 

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs

# Set the locale. Required for TMUX and neovim
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Change user
ARG USER_ID
ARG GROUP_ID
ARG GROUP_NAME
ARG USER_NAME
RUN groupadd -f -g "${GROUP_ID}" "${GROUP_NAME}"
RUN useradd -m -l -s /bin/bash -d /home/"${USER_NAME}" -u "${USER_ID}" -g "${GROUP_ID}" "${USER_NAME}" && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER "${USER_NAME}"

CMD [ "tail", "-f", "/dev/null" ]




