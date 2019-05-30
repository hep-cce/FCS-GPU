# FastCaloSimAnalyzer

_This package is standalone and thus does NOT require `asetup` BUT only requires `root 6.08` or later versions_ 
The package however depends on [`ISF_FastCaloSimEvent`](https://gitlab.cern.ch/atlas/athena/tree/master/Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent) and [`ISF_FastCaloSimParametrization`](https://gitlab.cern.ch/atlas/athena/tree/master/Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimParametrization) in `athena` and standalone [`EnergyParametrization`](https://gitlab.cern.ch/atlas-simulation-fastcalosim/EnergyParametrization). 

__Please see the [`Doxygen`](http://atlas-project-fastcalosim.web.cern.ch/atlas-project-fastcalosim/TFCSDoxygen/index.html) documentation for inheritence and dependency__

Please follow the following instruction to properly checkout the packages in the right location. For details on sparse checkout of Athena see [git tutorial](https://atlassoftwaredocs.web.cern.ch/gittutorial/git-clone/#sparse-checkout)


## Clone and setup
```
setupATLAS
lsetup git
cd /to/some/dir/
git atlas init-workdir https://:@gitlab.cern.ch:8443/atlas/athena.git
cd athena/
git fetch upstream 21.0
git checkout -b branch_21.0 upstream/21.0
git atlas addpkg ISF_FastCaloSimEvent
git atlas addpkg ISF_FastCaloSimParametrization
git atlas addpkg CaloGeoHelpers
cd ../
git clone https://:@gitlab.cern.ch:8443/atlas-simulation-fastcalosim/FastCaloSimAnalyzer.git --recursive
```
More details on cloning and git submodules are available in the FastCaloSimCommon package
[here](https://gitlab.cern.ch/atlas-simulation-fastcalosim/FastCaloSimCommon#how-to-clone-and-update)


## How to run

Building process is described in detail in the FastCaloSimCommon package
[here](https://gitlab.cern.ch/atlas-simulation-fastcalosim/FastCaloSimCommon#building-running).

```
lsetup cmake "root 6.14.08-x86_64-slc6-gcc62-opt"
mkdir build && cd build
cmake ../FastCaloSimAnalyzer
make -j$(ncore)
```

Each time you run the working environment needs to be set-up:
```
lsetup cmake "root 6.14.08-x86_64-slc6-gcc62-opt"
cd run
source ../build/x86_64-slc6-gcc62-opt/setup.sh
```

### Supported binaries and macros

* `runTFCSShapeValidation` - run toy simulation to validate the shape parametrization (not supporting command-line arguments yet)


## Legacy macro definitions - to be updated with CMake instructions

* Implement a steering macron in `macro/` based on your requirement. Currently available macros:
  * `runTFCS2DParametrizationHitogram.C`: to generate 2D shape histogram 
  *  `runTFCSCreateParametrization.cxx` : to create one big parametrization file from individual energy and shape parametrization
  *  `runTFCSCreateParamEtaSlices.cxx` : to create param eta slices
  *  `runTFCSMergeParamEtaSlices.cxx` : to merge the param eta slices to one file
  *  `runTFCSShapeValidation.cxx`: to run toy simulation to validate the shape parametrization
  *  `runTFCSEnergyInterpolationTGraph.C`: to create TGraphs used for energy interpolation
  *  `runTFCSWiggleDerivativeHistograms.C`: to create histograms that apply wiggles to hit used for accordion correction


 In /athena/Simulation/ISF/ISF_FastCaloSim/FastCaloSimAnalyzer/setup.sh, you will find the required setup for git and root

 

    $ source setup.sh 
    $ root -l  	 
    $ .x initTFCSAnalyzer.C
    $ .x runTFCS2DParametrizationHistogram.C(..) /.x runTFCSEnergyInterpolationTGraphs.C(..) / .x runTFCSShapeValidation.cxx / .x runTFCSCreateParamEtaSlices.cxx / .x runTFCSMergeParamEtaSlices.cxx

```
* Each macro as a set of default parameters that can be overwritten, for example the energy interpolation script has these parameters and default values
  * `int pid = 22` : 211 for pions, 11 for electrons
  * `float etamin = 0.` 
  * `float etamax = 5.` 
  * `bool useFit = false`: to fit the hit distribution instead of using the mean of the histrogram, experimental feature
  * `bool doSpline = true` 
  * `std::string inputDir = "/eos/atlas/atlascerngroupdisk/proj-simul/InputSamplesSummer18Complete/"` 
  * `std::string plotDir = "plot_Einterpol"` 
  * `std::string ver = "ver03"`: version of the big parametrisation file ~ depend mostly on production version
  * Hence `.x runTFCSEnergyInterpolation(211, 0, 3)` will run pion in the eta region from 0 to 3. 


## References

* Calorimeter layers:

```
PreSamplerB=0, EMB1, EMB2, EMB3,              // LAr barrel
PreSamplerE=4, EME1, EME2, EME3,                // LAr EM endcap
HEC0=8, HEC1, HEC2, HEC3,                       // Hadronic end cap cal.
TileBar0=12, TileBar1, TileBar2,                 // Tile barrel
TileGap1=15, TileGap2, TileGap3,                 // Tile gap (ITC & scint)
TileExt0=18, TileExt1, TileExt2,                 // Tile extended barrel
FCAL0=21, FCAL1, FCAL2,                          // Forward EM endcap
MINIFCAL0, MINIFCAL1, MINIFCAL2, MINIFCAL3,   // MiniFCAL 
```
