# -- CONFIGURATION -- #
export GCC_VERSION="5.4.0"
export CPUSPEC="IntelR-XeonR-E5-2650-0-200GHz"
#export CPUSPEC=`cat /proc/cpuinfo | grep "model name" | head -n1 | cut -d: -f2- | sed 's/^ *//; s/ \?(R)//g; s/ \?CPU//; s/@//; s/ \+/-/g;'` 
export GCC_ROOT="/opt/gcc-${GCC_VERSION}"
#export GCC_VERSION="$(cat ${GCC_ROOT}/gcc_version)"
#export BOOSTVER=`cat ${GCC_ROOT}/boost_version 2>/dev/null`
export PATH="${GCC_ROOT}/bin:$PATH"
export PATH=$(echo "$PATH" | awk -v RS=':' -v ORS=":" '!a[$1]++{if (NR > 1) printf ORS; printf $a[$1]}')
export LD_LIBRARY_PATH="${GCC_ROOT}/lib:${GCC_ROOT}/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | awk -v RS=':' -v ORS=":" '!a[$1]++{if (NR > 1) printf ORS; printf $a[$1]}')
# Is this necessary?
export LD_PRELOAD=/lib64/libstdc++.so.6 cmake
export CC="${GCC_ROOT}/bin/gcc"
export CXX="${GCC_ROOT}/bin/g++"
#export BOOST_ROOT="${GCC_ROOT}"
#export BOOST_DIR="${GCC_ROOT}"
export LDFLAGS=-L/usr/local/cuda-9.1/targets/x86_64-linux/lib/

if grep -q "(HACKED)"  ~/.bashrc
then
    echo ""
else
    echo "export PS1=\"`echo \$PS1 | cut -f1 -d]` \e[38;5;1m\](HACKED)\[\e[0m\]]`echo \$PS1 | cut -f2 -d]` \"" >> ~/.bashrc
fi
