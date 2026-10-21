#!/bin/sh
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# container/install.sh - the inside of the Containerfile: installs XeStrata's rpm (container/rpm/*.rpm when there is
# one, else the release's, checked against its SHA256SUMS), then every GPU maker's packages its BUILD.json records
# ("runtime"), so that the image runs Intel's, AMD's and NVIDIA's GPUs; NVIDIA's driver libraries come from the host
# (CDI), and the libraries of NVIDIA's neither Fedora's nor Ubuntu's own repositories have (cuDNN, Fedora's cuBLAS)
# from pip into the data folder (mounted from the host) when xestrata --setup names them.  XESTRATA_NONFREE=0 leaves
# out oneMKL (and its repository).
set -eu
repo=https://github.com/MistVVK/XeStrata
local=$(find /ctx/rpm -maxdepth 1 -name '*.rpm' 2>/dev/null | head -n 1)
if [ -n "$local" ]; then
  rpm=$local
else
  tag=$XESTRATA_VERSION
  if [ "$tag" = latest ]; then
    tag=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "$repo/releases/latest")
    tag=${tag##*/}
  fi
  cd /tmp
  curl -fsSL -o SHA256SUMS "$repo/releases/download/$tag/SHA256SUMS"
  name=$(sed -n "s/^[0-9a-f]\{64\} [ *]\(xestrata-$XESTRATA_VARIANT-[^ ]*\.x86_64\.rpm\)$/\1/p" SHA256SUMS | head -n 1)
  if [ -z "$name" ]; then
    echo "release $tag has no rpm of xestrata-$XESTRATA_VARIANT (XESTRATA_VARIANT)" >&2
    exit 1
  fi
  curl -fsSL -o "$name" "$repo/releases/download/$tag/$name"
  grep " [ *]$name\$" SHA256SUMS | sha256sum -c -
  rpm=/tmp/$name
fi
dnf install -y -q "$rpm"

# the GPU makers' packages, by maker, from the engine's record; not pip's names (nvidia_pip), which xestrata --setup
# names for the data folder, with this image's pip
runtime=$(python3 -c 'import json, sys
r = json.load(open(sys.argv[1])).get("runtime") or {}
skip = ("nvidia_pip",) + (() if sys.argv[2] == "1" else ("onemkl",))
print(" ".join(n for k, v in r.items() if k not in skip for n in v))' /usr/lib64/xestrata/engine/BUILD.json \
  "$XESTRATA_NONFREE")
case " $runtime " in
  *" intel-oneapi-"*)
    # Intel's oneAPI repository (oneMKL), as tools/package/container.sh adds it
    intel_key=https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
    printf '%s\n' '[oneAPI]' 'name=Intel oneAPI repository' 'baseurl=https://yum.repos.intel.com/oneapi' 'enabled=1' \
      'gpgcheck=1' 'repo_gpgcheck=1' "gpgkey=$intel_key" > /etc/yum.repos.d/oneAPI.repo
    ;;
esac
# shellcheck disable=SC2086
dnf install -y -q $runtime mesa-vulkan-drivers python3-pip
dnf clean all
