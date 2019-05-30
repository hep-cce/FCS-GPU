#include "GaudiKernel/DeclareFactoryEntries.h"
#include "ISF_FastCaloSimParametrization/FastCaloSimParamAlg.h"
#include "ISF_FastCaloSimParametrization/ISF_HitAnalysis.h"
#include "ISF_FastCaloSimParametrization/NativeFastCaloSimSvc.h"

#include "../FastCaloSimGeometryHelper.h"
#include "../FastCaloSimCaloExtrapolation.h"

DECLARE_TOOL_FACTORY( FastCaloSimGeometryHelper )
DECLARE_TOOL_FACTORY( FastCaloSimCaloExtrapolation )

DECLARE_ALGORITHM_FACTORY( FastCaloSimParamAlg )
DECLARE_ALGORITHM_FACTORY( ISF_HitAnalysis) 
DECLARE_NAMESPACE_SERVICE_FACTORY( ISF , NativeFastCaloSimSvc )

//DECLARE_FACTORY_ENTRIES(FastCaloSimParamAlg) {
//}

DECLARE_FACTORY_ENTRIES( ISF_FastCaloSimParametrization ) {
  DECLARE_ALGORITHM( FastCaloSimParamAlg )
  DECLARE_ALGORITHM( ISF_HitAnalysis )
  DECLARE_TOOL( FastCaloSimGeometryHelper )
  DECLARE_TOOL( FastCaloSimCaloExtrapolation )

  DECLARE_NAMESPACE_SERVICE( ISF , NativeFastCaloSimSvc )
}
