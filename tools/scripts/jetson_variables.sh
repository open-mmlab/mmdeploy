#!/bin/bash
# This file is part of the jetson_stats package (https://github.com/rbonghi/jetson_stats or http://rnext.it).
# Copyright (c) 2020 Raffaello Bonghi.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


# TODO Add environments variables:
# - UID -> https://devtalk.nvidia.com/default/topic/996988/jetson-tk1/chip-uid/post/5100481/#5100481
# - GCID, BOARD, EABI

###########################
#### JETPACK DETECTION ####
###########################
# Write version of jetpack installed
# https://developer.nvidia.com/embedded/jetpack-archive
jetson_jetpack()
{
    local JETSON_L4T=$1
    local JETSON_JETPACK=""
    case $JETSON_L4T in
        "35.1.0") JETSON_JETPACK="5.0.2" ;;
        "34.1.1") JETSON_JETPACK="5.0.1 DP" ;;
        "34.1.0") JETSON_JETPACK="5.0 DP" ;;
        "32.7.3") JETSON_JETPACK="4.6.3" ;;
        "32.7.2") JETSON_JETPACK="4.6.2" ;;
        "32.7.1") JETSON_JETPACK="4.6.1" ;;
        "32.6.1") JETSON_JETPACK="4.6" ;;
        "32.5.1" | "32.5.2") JETSON_JETPACK="4.5.1" ;;
        "32.5.0" | "32.5") JETSON_JETPACK="4.5" ;;
        "32.4.4") JETSON_JETPACK="4.4.1" ;;
        "32.4.3") JETSON_JETPACK="4.4" ;;
        "32.4.2") JETSON_JETPACK="4.4 DP" ;;
        "32.3.1") JETSON_JETPACK="4.3" ;;
        "32.2.3") JETSON_JETPACK="4.2.3" ;;
        "32.2.1") JETSON_JETPACK="4.2.2" ;;
        "32.2.0" | "32.2") JETSON_JETPACK="4.2.1" ;;
        "32.1.0" | "32.1") JETSON_JETPACK="4.2" ;;
        "31.1.0" | "31.1") JETSON_JETPACK="4.1.1" ;;
        "31.0.2") JETSON_JETPACK="4.1" ;;
        "31.0.1") JETSON_JETPACK="4.0" ;;
        "28.4.0") JETSON_JETPACK="3.3.3" ;;
        "28.2.1") JETSON_JETPACK="3.3 | 3.2.1" ;;
        "28.2.0" | "28.2") JETSON_JETPACK="3.2" ;;
        "28.1.0" | "28.1") JETSON_JETPACK="3.1" ;;
        "27.1.0" | "27.1") JETSON_JETPACK="3.0" ;;
        "24.2.1") JETSON_JETPACK="3.0 | 2.3.1" ;;
        "24.2.0" | "24.2") JETSON_JETPACK="2.3" ;;
        "24.1.0" | "24.1") JETSON_JETPACK="2.2.1 | 2.2" ;;
        "23.2.0" | "23.2") JETSON_JETPACK="2.1" ;;
        "23.1.0" | "23.1") JETSON_JETPACK="2.0" ;;
        "21.5.0" | "21.5") JETSON_JETPACK="2.3.1 | 2.3" ;;
        "21.4.0" | "21.4") JETSON_JETPACK="2.2 | 2.1 | 2.0 | 1.2 DP" ;;
        "21.3.0" | "21.3") JETSON_JETPACK="1.1 DP" ;;
        "21.2.0" | "21.2") JETSON_JETPACK="1.0 DP" ;;
        *) JETSON_JETPACK="UNKNOWN" ;;
    esac
    # return type jetpack
    echo $JETSON_JETPACK
}
###########################

JETSON_MODEL="UNKNOWN"
# Extract jetson model name
if [ -f /sys/firmware/devicetree/base/model ]; then
    JETSON_MODEL=$(tr -d '\0' < /sys/firmware/devicetree/base/model)
fi

# Extract jetson chip id
JETSON_CHIP_ID=""
if [ -f /sys/module/tegra_fuse/parameters/tegra_chip_id ]; then
    JETSON_CHIP_ID=$(cat /sys/module/tegra_fuse/parameters/tegra_chip_id)
fi
# Ectract type board
JETSON_SOC=""
if [ -f /proc/device-tree/compatible ]; then
    # Extract the last part of name
    JETSON_SOC=$(tr -d '\0' < /proc/device-tree/compatible | sed -e 's/.*,//')
fi
# Extract boardids if exists
JETSON_BOARDIDS=""
if [ -f /proc/device-tree/nvidia,boardids ]; then
    JETSON_BOARDIDS=$(tr -d '\0' < /proc/device-tree/nvidia,boardids)
fi

# Code name
JETSON_CODENAME=""
JETSON_MODULE="UNKNOWN"
JETSON_CARRIER="UNKNOWN"
list_hw_boards()
{
    # Extract from DTS the name of the boards available
    # Reference:
    # 1. https://unix.stackexchange.com/questions/251013/bash-regex-capture-group
    # 2. https://unix.stackexchange.com/questions/144298/delete-the-last-character-of-a-string-using-string-manipulation-in-shell-script
    local s=$1
    local regex='p([0-9-]+)' # Equivalent to p([\d-]*-)
    while [[ $s =~ $regex ]]; do
        local board_name=$(echo "P${BASH_REMATCH[1]}" | sed 's/-*$//' )
        # Load jetson type
        # If jetson type is not empty the module name is the same
        if [ $JETSON_MODULE = "UNKNOWN" ] ; then
            JETSON_MODULE=$board_name
            echo "JETSON_MODULE=\"$JETSON_MODULE\""
        else
            JETSON_CARRIER=$board_name
            echo "JETSON_CARRIER=\"$JETSON_CARRIER\""
        fi
        # Find next match
        s=${s#*"${BASH_REMATCH[1]}"}
    done
}
# Read DTS file
# 1. https://devtalk.nvidia.com/default/topic/1071080/jetson-nano/best-way-to-check-which-tegra-board/
# 2. https://devtalk.nvidia.com/default/topic/1014424/jetson-tx2/identifying-tx1-and-tx2-at-runtime/
# 3. https://devtalk.nvidia.com/default/topic/996988/jetson-tk1/chip-uid/post/5100481/#5100481
if [ -f /proc/device-tree/nvidia,dtsfilename ]; then
    # ..<something>/hardware/nvidia/platform/t210/porg/kernel-dts/tegra210-p3448-0000-p3449-0000-b00.dts
    # The last third is the codename of the board
    JETSON_CODENAME=$(tr -d '\0' < /proc/device-tree/nvidia,dtsfilename)
    JETSON_CODENAME=$(echo ${JETSON_CODENAME#*"/hardware/nvidia/platform/"} | tr '/' '\n' | head -2 | tail -1 )
    # The basename extract the board type
    JETSON_DTS="$(tr -d '\0' < /proc/device-tree/nvidia,dtsfilename | sed 's/.*\///')"
    # List of all boards
    eval $(list_hw_boards "$JETSON_DTS")
fi

# Export variables
export JETSON_MODEL
export JETSON_CHIP_ID
export JETSON_SOC
export JETSON_BOARDIDS
export JETSON_CODENAME
export JETSON_MODULE
export JETSON_CARRIER

# Write CUDA architecture
# https://developer.nvidia.com/cuda-gpus
# https://devtalk.nvidia.com/default/topic/988317/jetson-tx1/what-should-be-the-value-of-cuda_arch_bin/
case $JETSON_MODEL in
    *Orin*) JETSON_CUDA_ARCH_BIN="8.7" ;;
    *Xavier*) JETSON_CUDA_ARCH_BIN="7.2" ;;
    *TX2*) JETSON_CUDA_ARCH_BIN="6.2" ;;
    *TX1* | *Nano*) JETSON_CUDA_ARCH_BIN="5.3" ;;
    *TK1*) JETSON_CUDA_ARCH_BIN="3.2" ;;
    * ) JETSON_CUDA_ARCH_BIN="NONE" ;;
esac
# Export Jetson CUDA ARCHITECTURE
export JETSON_CUDA_ARCH_BIN

# Serial number
# https://devtalk.nvidia.com/default/topic/1055507/jetson-nano/nano-serial-number-/
JETSON_SERIAL_NUMBER=""
if [ -f /sys/firmware/devicetree/base/serial-number ]; then
    JETSON_SERIAL_NUMBER=$(tr -d '\0' </sys/firmware/devicetree/base/serial-number)
fi
# Export NVIDIA Serial Number
export JETSON_SERIAL_NUMBER

# NVIDIA Jetson version
# reference https://devtalk.nvidia.com/default/topic/860092/jetson-tk1/how-do-i-know-what-version-of-l4t-my-jetson-tk1-is-running-/
# https://stackoverflow.com/questions/16817646/extract-version-number-from-a-string
# https://askubuntu.com/questions/319307/reliably-check-if-a-package-is-installed-or-not
# https://github.com/dusty-nv/jetson-inference/blob/7e81381a96c1ac5f57f1728afbfdec7f1bfeffc2/tools/install-pytorch.sh#L296
if [ -f /etc/nv_tegra_release ]; then
    # L4T string
    # First line on /etc/nv_tegra_release
    # - "# R28 (release), REVISION: 2.1, GCID: 11272647, BOARD: t186ref, EABI: aarch64, DATE: Thu May 17 07:29:06 UTC 2018"
    JETSON_L4T_STRING=$(head -n 1 /etc/nv_tegra_release)
    # Load release and revision
    JETSON_L4T_RELEASE=$(echo $JETSON_L4T_STRING | cut -f 2 -d ' ' | grep -Po '(?<=R)[^;]+')
    JETSON_L4T_REVISION=$(echo $JETSON_L4T_STRING | cut -f 2 -d ',' | grep -Po '(?<=REVISION: )[^;]+')
elif dpkg --get-selections | grep -q "^nvidia-l4t-core[[:space:]]*install$"; then
    # Read version
    JETSON_L4T_STRING=$(dpkg-query --showformat='${Version}' --show nvidia-l4t-core)
    # extract version
    JETSON_L4T_ARRAY=$(echo $JETSON_L4T_STRING | cut -f 1 -d '-')
    # Load release and revision
    JETSON_L4T_RELEASE=$(echo $JETSON_L4T_ARRAY | cut -f 1 -d '.')
    JETSON_L4T_REVISION=${JETSON_L4T_ARRAY#"$JETSON_L4T_RELEASE."}
else
    # Load release and revision
    JETSON_L4T_RELEASE="N"
    JETSON_L4T_REVISION="N.N"
fi
# Write Jetson description
JETSON_L4T="$JETSON_L4T_RELEASE.$JETSON_L4T_REVISION"
# Export Release L4T
export JETSON_L4T_RELEASE
export JETSON_L4T_REVISION
export JETSON_L4T

JETSON_JETPACK=$(jetson_jetpack $JETSON_L4T)
# Export Jetson Jetpack installed
export JETSON_JETPACK
# EOF
