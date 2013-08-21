#!/bin/sh

# Install binwalk/fmk pre-requisites and extraction tools
sudo apt-get install subversion build-essential mtd-utils zlib1g-dev liblzma-dev gzip bzip2 tar unrar arj p7zip openjdk-6-jdk python-magic python-matplotlib

# Get and build the firmware mod kit
sudo mkdir -p /opt/firmware-mod-kit
sudo chmod a+rwx /opt/firmware-mod-kit
rm -rf /opt/firmware-mod-kit/trunk
svn checkout http://firmware-mod-kit.googlecode.com/svn/trunk /opt/firmware-mod-kit/trunk
cd /opt/firmware-mod-kit/trunk/src && ./configure && make && cd -

# Install binwalk
sudo python setup.py install
