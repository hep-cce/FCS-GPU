# EnergyParametrization

## Getting the code

Central FastCaloSim from Athena (also depending on CaloGeoHelpers) is needed.
Athena setup is not needed, but the packages need to be checked out.
By default, check out the Athena and the EnergyParametrization package side-by-side.

[How to get Athena](https://gitlab.cern.ch/atlas-simulation-fastcalosim/FastCaloSimCommon#checking-out-athena)

*Note:* **Do not** compile those packages, **do not** setup any Athena release.

```
git clone https://:@gitlab.cern.ch:8443/atlas-simulation-fastcalosim/EnergyParametrization.git --recursive
```
More details how to clone and update are available
[here](https://gitlab.cern.ch/atlas-simulation-fastcalosim/FastCaloSimCommon#how-to-clone-and-update)


## Building

Building process is described in detail in the FastCaloSimCommon package
[here](https://gitlab.cern.ch/atlas-simulation-fastcalosim/FastCaloSimCommon#building-running).

```
lsetup cmake "root 6.14.08-x86_64-slc6-gcc62-opt"
mkdir build && cd build
cmake ../EnergyParametrization
make -j$(ncore)
```


## Running

Each time you run the working environment needs to be set-up:
```
lsetup cmake "root 6.14.08-x86_64-slc6-gcc62-opt"
cd run
source ../build/x86_64-slc6-gcc62-opt/setup.sh
``` 

To run, execute `run_epara` from any directory. For command-line options run `run_epara --help`.
