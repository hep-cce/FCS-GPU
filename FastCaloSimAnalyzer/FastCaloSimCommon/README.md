# FastCaloSimCommon

Common infrastructure for FastCaloSim standalone code.


## Contents

 - Common [CMake](cmake) configuration
 - Standalone [Athena build](AthenaBuild)
 - Standalone [CLHEP](CLHEP) interface implementation
 - Standalone [HepPDT](HepPDT) interface implementation
 - `TFCSSampleDiscovery` sample discovery class


## How to clone and update?

This repository uses [Git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules).
To clone, use
```
git clone https://:@gitlab.cern.ch:8443/atlas-simulation-fastcalosim/FastCaloSimCommon.git --recursive
```
When using the Atlas environment, `lsetup git` should be called first.

To update the repository an additional command is needed besides the usual pull
```
git pull --rebase
git submodule update --recursive
```

*Note:* as we use nested submodules it's always recommended to use `--recursive` flag when updating.

### Updating submodules

Each submodule acts as a "file" in the base repository. To update it just change the branch or
commit inside the submodule and call `git add` on it in the base repository.
As each submodule is its own repository make sure that you also commit and push inside
the submodule.

*Example:* You want to update FastCaloSimCommon inside FastCaloSimAnalyzer
```
cd FastCaloSimAnalyzer/FastCaloSimCommon
git commit -m "Some FastCaloSimCommon changes"
git push
cd ..
git add FastCaloSimCommon
git commit -m "Bump FastCaloSimCommon due to changes"
git push
```

### Checking out Athena

By default clone Athena side-by-side with the standalone FastCaloSim package.

First setup ATLAS environment with git if needed:
``` 
setupATLAS
lsetup git
```

Clone using Git sparse checkout (recommended):
``` 
git atlas init-workdir https://:@gitlab.cern.ch:8443/atlas/athena.git
cd athena
git atlas addpkg ISF_FastCaloSimEvent
git atlas addpkg ISF_FastCaloSimParametrization
git atlas addpkg CaloGeoHelpers
``` 

Clone directly:
```
git clone https://:@gitlab.cern.ch:8443/USERNAME/athena.git
cd athena
git remote add upstream https://:@gitlab.cern.ch:8443/atlas/athena.git
``` 

*Note:* **Do not** compile those packages, **do not** setup any Athena release.


## Building & Running

CMake (3.8+) and ROOT (6.14.08) are required. You should **always** make an out-of-source build!
```
lsetup cmake "root 6.14.08-x86_64-slc6-gcc62-opt"
mkdir build && cd build
cmake ../FastCaloSimCommon
make -j$(ncore)
```

After the build the working environment needs to be set-up:
```
mkdir run && cd run
source ../build/x86_64-slc6-gcc62-opt/setup.sh
```
Note that the platform folder might be different on your OS/architecture.


### CMake flags supported

Any configuration flag can be set to CMake using `-D<flag>=<value>`.

- `ATHENA_PATH` - set Athena path, `../` by default
- `DEBUG_LOGGING` - enable verbose debug logging, `OFF` by default
- `ENABLE_XROOTD` - enable XRootD support, `ON` by default
- `INPUT_PATH` - override all inputs path, empty and disabled by default
- `ROOT_VERSION` - override required ROOT version, `6.14.08` by default
