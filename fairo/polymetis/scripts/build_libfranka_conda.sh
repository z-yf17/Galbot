#!/usr/bin/env bash
set -euo pipefail

# 使用：./scripts/build_libfranka.sh 0.15.0
# 若不传版本则不切换 tag，按当前 third_party/libfranka 的版本编

GIT_ROOT=$(git rev-parse --show-toplevel)
LIBFRANKA_VER=${1:-}
LIBFRANKA_PATH="$GIT_ROOT/polymetis/polymetis/src/clients/franka_panda_client/third_party/libfranka"

# 需要一个已激活的 conda 环境
: "${CONDA_PREFIX:?请先激活你的 conda 环境}"

# 彻底避免 CMake 误拾 ROS
unset ROS_DISTRO ROS_PACKAGE_PATH ROS_ROOT ROS_ETC_DIR
export CMAKE_IGNORE_PREFIX_PATH=/opt/ros:/usr/lib/ros
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/share/pkgconfig"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '^/opt/ros' | paste -sd: -)"

# 1) 源码位置检查 & 切版本
[ ! -d "$LIBFRANKA_PATH" ] && echo "Directory $LIBFRANKA_PATH does not exist" && exit 1
cd "$LIBFRANKA_PATH"
if [ -n "$LIBFRANKA_VER" ]; then
  git fetch --tags
  git checkout "$LIBFRANKA_VER"
fi
git submodule update --init --recursive

# 2) 干净构建 & 安装到 $CONDA_PREFIX
BUILD_PATH="$LIBFRANKA_PATH/build"
rm -rf "$BUILD_PATH"
mkdir -p "$BUILD_PATH"

echo "Building & installing libfranka into $CONDA_PREFIX"
cmake -S . -B "$BUILD_PATH" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DCMAKE_FIND_ROOT_PATH="$CONDA_PREFIX" \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY

cmake --build "$BUILD_PATH" -j"$(nproc)"
cmake --install "$BUILD_PATH"

# 3) 打印结果确认
echo "---- libfranka installed files ----"
ls -l "$CONDA_PREFIX/lib"/libfranka.so*
echo "---- readelf (检查有没有 /opt/ros 路径 & 多余依赖) ----"
readelf -d "$CONDA_PREFIX/lib/libfranka.so" | egrep 'NEEDED|RUNPATH|RPATH' || true

echo "Done."

