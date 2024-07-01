# Build FCS 


# Base images containing CUDA (or HPC SDK) and ROOT. 
```
docker.io/dingpf/root:6.30.04-cuda12.2.2-devel-ubuntu22.04
docker.io/dingpf/root:6.30.04-nvhpc23.9-devel-cuda12.2-ubuntu22.04
```

# FCS + CUDA
```
docker.io/dingpf/fcs-kokkos-cuda:6.30.04-cuda12.2.2-devel-ubuntu22.04
docker.io/dingpf/fcs-kokkos-cuda:6.30.04-nvhpc23.9-devel-cuda12.2-ubuntu22.04

# Run with "podman-hpc run --rm --gpu -it -v /global/cfs/cdirs/atlas/leggett/data/FastCaloSimInputs:/input <Image>"
# In container, run "export FCS_DATAPATH=/input; export LD_LIBRARY_PATH=/hep-mini-apps/Kokkos/install/lib:$LD_LIBRARY_PATH; source /hep-mini-apps/root/install/bin/thisroot.sh; source /hep-mini-apps/FCS-GPU/install/setup.sh; runTFCSSimulation --earlyReturn --energy 65536"

```

# FCS + Kokkos + CUDA

```
docker.io/dingpf/fcs-cuda:6.30.04-cuda12.2.2-devel-ubuntu22.04
docker.io/dingpf/fcs-cuda:6.30.04-nvhpc23.9-devel-cuda12.2-ubuntu22.04
# Run with "podman-hpc run --rm --gpu -it -v /global/cfs/cdirs/atlas/leggett/data/FastCaloSimInputs:/input <Image>"
# In container, run "export FCS_DATAPATH=/input; source /hep-mini-apps/root/install/bin/thisroot.sh; source /hep-mini-apps/FCS-GPU/install/setup.sh; runTFCSSimulation --earlyReturn --energy 65536"
```
