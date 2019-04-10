#!/usr/bin/env bash

mpilibs=("/opt/ibm/spectrum_mpi/lib" \
         "/opt/ibm/spectrum_mpi/lib/spectrum_mpi" \
         "/opt/ibm/spectrum_mpi/jsm_pmix/lib" \
         "/opt/ibm/spectrum_mpi/lib/pami_port" \
         "/opt/mellanox/hcoll/lib" \
         "/opt/mellanox/sharp/lib" \
         "/opt/mellanox/mxm/lib")
deplibs=("libucp.so" "libuct.so" "libucs.so" \
         "libucm.so" "libnl-route-3.so" \
         "librdmacm.so" "libibumad.so" \
         "libibverbs.so" "libosmcomp.so" \
         "libnl-3.so" "librxe-rdmav2.so" \
         "libmlx4-rdmav2.so" "libmlx5-rdmav2.so")

for libs in ${deplibs[@]}; do
  for lib in /usr/lib64/${libs}*; do
     if [ -z "${SINGULARITY_CONTAINLIBS:-}" ]; then
            SINGULARITY_CONTAINLIBS="$lib"
        else
            SINGULARITY_CONTAINLIBS="$SINGULARITY_CONTAINLIBS,$lib"
    fi
  done
done

for libdir in ${mpilibs[@]}; do
  for lib in $libdir/*.so*; do
    if [ -z "${SINGULARITY_CONTAINLIBS:-}" ]; then
            SINGULARITY_CONTAINLIBS="$lib"
        else
            SINGULARITY_CONTAINLIBS="$SINGULARITY_CONTAINLIBS,$lib"
    fi
  done
done

export SINGULARITYENV_LD_PRELOAD=$(readlink -f /opt/ibm/spectrum_mpi/lib/libpami_cudahook.so)
export SINGULARITY_CONTAINLIBS
export SINGULARITYENV_PREPEND_PATH=/opt/anaconda2/bin:/opt/anaconda3/bin

