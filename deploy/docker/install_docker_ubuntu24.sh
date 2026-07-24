#!/bin/bash

# Add Docker's official GPG key:
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update

sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add current user to the 'docker' group, which allows them to use docker commands (docker build, docker run, etc).
# See https://docs.docker.com/engine/install/linux-postinstall/
username=$(whoami)
sudo usermod -aG docker $username

# Configure docker to start automatically on system boot.
sudo systemctl enable docker.service
sudo systemctl enable containerd.service

# https://forums.docker.com/t/docker-credential-desktop-exe-executable-file-not-found-in-path-using-wsl2/100225/5
if [ ~/.docker/config.json ]; then
	sed -i 's/credsStore/credStore/g' ~/.docker/config.json
fi

echo ""
echo "********************************************************************"
echo "**** Restart to allow Docker permission changes to take effect. ****"
echo "********************************************************************"
echo ""
