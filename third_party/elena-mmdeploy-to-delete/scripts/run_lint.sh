#!/usr/bin/env bash
# CPPLint script for use with GitLab CI.
# Author: Pengcheng Xu <xupengcheng@sensetime.com>

PYTHON="${PYTHON:-python}"
FIND="${FIND:-find}"

realpath() {
    cd "$@" && pwd
}

SCRIPT_DIR="$(dirname ${BASH_SOURCE[0]})"
SRC_DIR="$(realpath ${SCRIPT_DIR}/../)"
THIRD_PARTY_DIR="${SRC_DIR}/3rdparty"
BUILD_DIR="${SRC_DIR}/build"
PYTHON_DIR="${SRC_DIR}/python"

echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "SRC_DIR=${SRC_DIR}"
echo "THIRD_PARTY_DIR=${THIRD_PARTY_DIR}"
echo "BUILD_DIR=${BUILD_DIR}"

CPPLINT="${SCRIPT_DIR}/cpplint.py"
LINTER="${LINTER:-python "${CPPLINT}"}"

die() {
    echo "Error: $1" >&2
    exit 1
}

# if ! $FIND --help &>/dev/null; then
#     die "GNU find is required."
# fi

echo "Checking *.cc *.cpp *.h *.hpp for cpplint errors..."

if ! $FIND "${SRC_DIR}" \
    -path "${BUILD_DIR}"  -prune -o \
    -path "${THIRD_PARTY_DIR}" -prune -o \
    -path "${PYTHON_DIR}" -prune -o \
    \( -iname '*.cc' -o -iname '*.cpp' -o -iname '*.h' -o -iname '*.hpp' \) \
    -exec $LINTER {} +; then
    die "Correct the errors before continuing."
else
    echo "Static checking passed."
fi
