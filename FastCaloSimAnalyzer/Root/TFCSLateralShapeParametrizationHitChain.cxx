/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/
#include "FastCaloSimAnalyzer/TFCSLateralShapeParametrizationHitChain.h"
#include "ISF_FastCaloSimEvent/FastCaloSim_CaloCell_ID.h"

#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"
#include "FastCaloSimAnalyzer/TFCSShapeValidation.h"
#include <mutex>

#include <mutex>

#if defined USE_GPU || defined USE_OMPGPU
#  include <typeinfo>
#  include "ISF_FastCaloSimEvent/TFCS1DFunction.h"
#  include "ISF_FastCaloSimEvent/TFCSCenterPositionCalculation.h"
#  include "FastCaloSimAnalyzer/TFCSHistoLateralShapeParametrization.h"
#  include "FastCaloSimAnalyzer/TFCSHitCellMappingWiggle.h"
#  include "ISF_FastCaloSimEvent/TFCSHitCellMapping.h"

#  include "FastCaloGpu/FastCaloGpu/CaloGpuGeneral.h"
#  include "FastCaloGpu/FastCaloGpu/GeoLoadGpu.h"
#  include "FastCaloGpu/FastCaloGpu/Args.h"
#  include "HepPDT/ParticleData.hh"
#  include "HepPDT/ParticleDataTable.hh"
#  include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"

#  include <chrono>

#endif

static bool FCS_dump_hitcount {false};
static std::once_flag calledGetEnv {};

//=============================================
//======= TFCSLateralShapeParametrization =========
//=============================================

TFCSLateralShapeParametrizationHitChain::TFCSLateralShapeParametrizationHitChain( const char* name, const char* title )
    : TFCSLateralShapeParametrization( name, title ), m_number_of_hits_simul( nullptr ) {}

TFCSLateralShapeParametrizationHitChain::TFCSLateralShapeParametrizationHitChain(
    TFCSLateralShapeParametrizationHitBase* hitsim )
    : TFCSLateralShapeParametrization( TString( "hit_chain_" ) + hitsim->GetName(),
                                       TString( "hit chain for " ) + hitsim->GetTitle() )
    , m_number_of_hits_simul( nullptr ) {
  set_pdgid_Ekin_eta_Ekin_bin_calosample( *hitsim );

  m_chain.push_back( hitsim );
  
}

void TFCSLateralShapeParametrizationHitChain::set_geometry( ICaloGeometry* geo ) {
  TFCSLateralShapeParametrization::set_geometry( geo );
  if ( m_number_of_hits_simul ) m_number_of_hits_simul->set_geometry( geo );
}

int TFCSLateralShapeParametrizationHitChain::get_number_of_hits( TFCSSimulationState&          simulstate,
                                                                 const TFCSTruthState*         truth,
                                                                 const TFCSExtrapolationState* extrapol ) const {
  // TODO: should we still do it?
  if ( m_number_of_hits_simul ) {
    int n = m_number_of_hits_simul->get_number_of_hits( simulstate, truth, extrapol );
    if ( n < 1 ) n = 1;
    return n;
  }
  for ( TFCSLateralShapeParametrizationHitBase* hitsim : m_chain ) {
    int n = hitsim->get_number_of_hits( simulstate, truth, extrapol );
    if ( n > 0 ) return n;
  }
  return 1;
}

FCSReturnCode TFCSLateralShapeParametrizationHitChain::simulate( TFCSSimulationState&          simulstate,
                                                                 const TFCSTruthState*         truth,
                                                                 const TFCSExtrapolationState* extrapol ) {

  auto ss0 = simulstate.cells().size();
  bool onGPU=false;
  auto start = std::chrono::system_clock::now();

  int cs = calosample();
  // Call get_number_of_hits() only once, as it could contain a random number
  int nhit = get_number_of_hits( simulstate, truth, extrapol );
  if ( nhit <= 0 ) {
    ATH_MSG_ERROR( "TFCSLateralShapeParametrizationHitChain::simulate(): number of hits could not be calculated" );
    return FCSFatal;
  }

  //std::cout << "-------- calosample cs  and   nhit ----------" << cs << " " << nhit << std::endl;
  float Ehit = simulstate.E( cs ) / nhit;

  bool debug = msgLvl( MSG::DEBUG );
  if ( debug ) { ATH_MSG_DEBUG( "E(" << cs << ")=" << simulstate.E( cs ) << " #hits=" << nhit ); }

#if defined USE_GPU || defined USE_OMPGPU
  /*
    std::string sA[5]={"TFCSCenterPositionCalculation","TFCSValidationHitSpy","TFCSHistoLateralShapeParametrization",
           "TFCSHitCellMappingWiggle", "TFCSValidationHitSpy" } ;
    std::string sB[3]={"TFCSCenterPositionCalculation","TFCSHistoLateralShapeParametrization",
           "TFCSHitCellMappingWiggle" } ;
   */

  if ( debug ) {
    std::cout << "---xxx---nhits=" << nhit << ", ";
    for ( TFCSLateralShapeParametrizationHitBase* hitsim : m_chain )
      std::cout << "-----In TFCSLateralShapeParametizationHitChain:" << typeid( *hitsim ).name() << " " << hitsim
                << std::endl;
    std::cout << std::endl;
  }
  int  ichn       = 0;
  bool our_chainA = false;
  bool our_chainB = false;
  //  bool our_chainC = false;
  /*
     for(TFCSLateralShapeParametrizationHitBase* hitsim : m_chain) {
        if (std::string(typeid( * hitsim ).name()).find(sA[ichn++]) == std::string::npos)
             { our_chainA= false ; break ; }
    }

    ichn=0 ;
    for(TFCSLateralShapeParametrizationHitBase* hitsim : m_chain) {
        if (std::string(typeid( * hitsim ).name()).find(sB[ichn++]) == std::string::npos)
             { our_chainB= false ; break ; }
    }
  */

  if ( cs == 0 || cs == 4 || ( cs >= 8 && cs < 21 ) )
    our_chainB = true;
  else if ( cs > 0 && cs < 8 && cs != 4 )
    our_chainA = true;
  // else
  //   our_chainC = true;

  //    if ( nhit > MIN_GPU_HITS && (our_chainA || our_chainB) ) {
  if ( nhit > MIN_GPU_HITS && our_chainA ) {
    onGPU=true;
    GeoLoadGpu* gld = (GeoLoadGpu*)simulstate.get_geold();

    Chain0_Args args;

    args.debug            = debug;
    args.cs               = cs;
    args.extrapol_eta_ent = extrapol->eta( cs, SUBPOS_ENT );
    args.extrapol_eta_ext = extrapol->eta( cs, SUBPOS_EXT );
    args.extrapol_phi_ent = extrapol->phi( cs, SUBPOS_ENT );
    args.extrapol_phi_ext = extrapol->phi( cs, SUBPOS_EXT );
    args.extrapol_r_ent   = extrapol->r( cs, SUBPOS_ENT );
    args.extrapol_r_ext   = extrapol->r( cs, SUBPOS_EXT );
    args.extrapol_z_ent   = extrapol->z( cs, SUBPOS_ENT );
    args.extrapol_z_ext   = extrapol->z( cs, SUBPOS_EXT );

    args.pdgId  = truth->pdgid();
    args.charge = HepPDT::ParticleID( args.pdgId ).charge();

    args.nhits  = nhit;
    args.rand   = 0;
    args.geo    = gld->get_geoPtr();
    args.rd4h   = simulstate.get_gpu_rand();
    args.ncells = gld->get_ncells();

    args.is_first = simulstate.get_es()->is_first;
    args.is_last  = simulstate.get_es()->is_last;

    ichn = 0;
    for ( auto hitsim : m_chain ) {

      //	std::string s= std::string(typeid( * hitsim ).name()) ;

      //	if(s.find("TFCSCenterPositionCalculation") != std::string::npos ) {
      if ( ichn == 0 ) {
        //         std::cout<<"---m_extrapWeight"<< ((TFCSCenterPositionCalculation *)hitsim)->getExtrapWeight()
        //         <<std::endl ;
        //  hitsim->Print();
        args.extrapWeight = ( (TFCSCenterPositionCalculation*)hitsim )->getExtrapWeight();
      }

      if ( ichn == 1 ) {

        //	TFCS2DFunctionHistogram h=((TFCSHistoLateralShapeParametrization *) hitsim)->histogram() ;
        //	std::cout << "size of hist: "<<h.get_HistoBordersx().size() <<", "<<h.get_HistoBordersy().size()
        //		<<"Pointer: " << &h <<std::endl ;
        auto tt1 = std::chrono::system_clock::now();

        ( (TFCSHistoLateralShapeParametrization*)hitsim )->LoadHistFuncs();

        auto tt2 = std::chrono::system_clock::now();
        TFCSShapeValidation::time_o1 += ( tt2 - tt1 );

        args.is_phi_symmetric = ( (TFCSHistoLateralShapeParametrization*)hitsim )->is_phi_symmetric();
        args.fh2d             = ( (TFCSHistoLateralShapeParametrization*)hitsim )->LdFH()->hf2d_d();

        // std::cout<<"Hitsim_ptr="<<hitsim<<", Ld_FH_ptr="<<  ((TFCSHistoLateralShapeParametrization *) hitsim)->LdFH()
        // <<",FH2d_ptr="<< args.fh2d <<std::endl ;

        args.fh2d_h = *( ( (TFCSHistoLateralShapeParametrization*)hitsim )->LdFH()->hf2d_h() );
      }
      if ( ichn == 2 ) {
        //if ( 0 ) {
        //  std::cout << "---NumberOfBins:" << ( (TFCSHitCellMappingWiggle*)hitsim )->get_number_of_bins() << std::endl;
        //  std::vector<const TFCS1DFunction*> funcs = ( (TFCSHitCellMappingWiggle*)hitsim )->get_functions();
        //  for ( auto it = funcs.begin(); it != funcs.end(); ++it ) {

        //    std::cout << "----+++type of funcs: " << typeid( *( *it ) ).name() << ", pointer: " << *it << std::endl;
        //  }
        //}
        auto tt1 = std::chrono::system_clock::now();
        ( (TFCSHitCellMappingWiggle*)hitsim )->LoadHistFuncs();
        auto tt2 = std::chrono::system_clock::now();
        TFCSShapeValidation::time_o1 += ( tt2 - tt1 );

        args.fhs = ( (TFCSHitCellMappingWiggle*)hitsim )->LdFH()->hf_d();
        args.fhs_h = *(( (TFCSHitCellMappingWiggle*)hitsim )->LdFH()->hf_h() );
      }

      ichn++;
    }

    //  auto t1 = std::chrono::system_clock::now();
    //  std::chrono::duration<double> diff = t1-start;
    //  std::cout <<  "Time before GPU simulate_hit :" << diff.count() <<" s" << std::endl ;

    CaloGpuGeneral::simulate_hits( Ehit, nhit, args );

    for ( unsigned int ii = 0; ii < args.ct; ++ii ) {
      // std::cout<<"celleleIndex="<< args.hitcells_h[ii]<<" " << args.hitcells_ct_h[ii]<<std::endl;
      const CaloDetDescrElement* cellele = gld->index2cell( args.hitcells_E_h[ii].cellid );
      simulstate.deposit( cellele, args.hitcells_E_h[ii].energy );
    }

    //  auto t2 = std::chrono::system_clock::now();
    //  diff = t2-t1;
    //  std::cout <<  "Time of GPU simulate_hit :" << diff.count() <<" s" <<" CT="<<args.ct<<  std::endl ;
    //  TFCSShapeValidation::time_g += (t2-start) ;
  } else {
#endif
   auto end_nhits = std::chrono::system_clock::now();
   TFCSShapeValidation::time_nhits += end_nhits - start;
  
    auto start_hit = std::chrono::system_clock::now();
  
    for ( int i = 0; i < nhit; ++i ) {
   
      auto start_mchain = std::chrono::system_clock::now();
      TFCSLateralShapeParametrizationHitBase::Hit hit;
      hit.E() = Ehit;
      
      for ( TFCSLateralShapeParametrizationHitBase* hitsim : m_chain ) {
      
        auto start_hitsim = std::chrono::system_clock::now();
        if ( debug ) {
          if ( i < 2 )
            hitsim->setLevel( MSG::DEBUG );
          else
            hitsim->setLevel( MSG::INFO );
        }

        for ( int i = 0; i <= FCS_RETRY_COUNT; i++ ) {
          if ( i > 0 )
            ATH_MSG_WARNING( "TFCSLateralShapeParametrizationHitChain::simulate(): Retry simulate_hit call "
                             << i << "/" << FCS_RETRY_COUNT );

          FCSReturnCode status = hitsim->simulate_hit( hit, simulstate, truth, extrapol );
    
          if ( status == FCSSuccess )
            break;
          else if ( status == FCSFatal )
            return FCSFatal;

          if ( i == FCS_RETRY_COUNT ) {
            ATH_MSG_ERROR( "TFCSLateralShapeParametrizationHitChain::simulate(): simulate_hit call failed after "
                           << FCS_RETRY_COUNT << "retries" );
          }
        }
        auto end_hitsim = std::chrono::system_clock::now();
        TFCSShapeValidation::time_hitsim += end_hitsim - start_hitsim;

      }
      auto end_mchain = std::chrono::system_clock::now();
      TFCSShapeValidation::time_mchain += end_mchain - start_mchain; 
    }
    
#if defined USE_GPU || defined USE_OMPGPU
  }

  auto t2 = std::chrono::system_clock::now();
  if ( nhit > MIN_GPU_HITS && our_chainA ) {
    TFCSShapeValidation::time_g1 += ( t2 - start );
  } else if ( nhit > MIN_GPU_HITS && our_chainB ) {
    TFCSShapeValidation::time_g2 += ( t2 - start );
  } else
#endif
  {
    auto t2 = std::chrono::system_clock::now();
    TFCSShapeValidation::time_h += ( t2 - start );
  }
  {
    auto t3 = std::chrono::system_clock::now();
    TFCSShapeValidation::time_o2 += ( t3 - start );
  }


  std::call_once(calledGetEnv, [](){
        if(const char* env_p = std::getenv("FCS_DUMP_HITCOUNT")) {
          if  (strcmp(env_p,"1") == 0) {
            FCS_dump_hitcount = true;
          }
        }
  });
  
  if (FCS_dump_hitcount) {
    printf(" HitCellCount: %3lu / %3lu   nhit: %4d%3s\n", simulstate.cells().size()-ss0,
           simulstate.cells().size(), nhit, (onGPU ? "  *" : "") );
  }
  
  return FCSSuccess;
}

void TFCSLateralShapeParametrizationHitChain::Print( Option_t* option ) const {
  TFCSLateralShapeParametrization::Print( option );
  TString opt( option );
  bool    shortprint = opt.Index( "short" ) >= 0;
  bool    longprint  = msgLvl( MSG::DEBUG ) || ( msgLvl( MSG::INFO ) && !shortprint );
  TString optprint   = opt;
  optprint.ReplaceAll( "short", "" );

  if ( m_number_of_hits_simul ) {
    if ( longprint ) ATH_MSG_INFO( optprint << "#:Number of hits simulation:" );
    m_number_of_hits_simul->Print( opt + "#:" );
  }
  if ( longprint ) ATH_MSG_INFO( optprint << "- Simulation chain:" );
  char count = 'A';
  for ( TFCSLateralShapeParametrizationHitBase* hitsim : m_chain ) {
    hitsim->Print( opt + count + " " );
    count++;
  }
}

#ifdef USE_GPU
void gpu_hit_chain() {}

#endif
