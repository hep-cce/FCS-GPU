/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/
#include "FastCaloSimAnalyzer/TFCSLateralShapeParametrizationHitChain.h"
#include "ISF_FastCaloSimEvent/FastCaloSim_CaloCell_ID.h"

#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"


#ifdef USE_GPU
#include <typeinfo>
#include "ISF_FastCaloSimEvent/TFCS1DFunction.h"
#include "ISF_FastCaloSimEvent/TFCSCenterPositionCalculation.h"
#include "FastCaloSimAnalyzer/TFCSHistoLateralShapeParametrization.h"
#include "FastCaloSimAnalyzer/TFCSHitCellMappingWiggle.h"
#include "ISF_FastCaloSimEvent/TFCSHitCellMapping.h"
#include "FastCaloSimAnalyzer/TFCSValidationHitSpy.h"
#include "FastCaloGpu/FastCaloGpu/CaloGpuGeneral.h"
#include "FastCaloGpu/FastCaloGpu/GeoLoadGpu.h"
#include "FastCaloGpu/FastCaloGpu/Args.h"
#include "HepPDT/ParticleData.hh"
#include "HepPDT/ParticleDataTable.hh"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"
#endif



#include <chrono>
#include "FastCaloSimAnalyzer/TFCSShapeValidation.h"


//=============================================
//======= TFCSLateralShapeParametrization =========
//=============================================

TFCSLateralShapeParametrizationHitChain::TFCSLateralShapeParametrizationHitChain(const char* name, const char* title):TFCSLateralShapeParametrization(name,title),m_number_of_hits_simul(nullptr)
{
}

TFCSLateralShapeParametrizationHitChain::TFCSLateralShapeParametrizationHitChain(TFCSLateralShapeParametrizationHitBase* hitsim):TFCSLateralShapeParametrization(TString("hit_chain_")+hitsim->GetName(),TString("hit chain for ")+hitsim->GetTitle()),m_number_of_hits_simul(nullptr)
{
  set_pdgid_Ekin_eta_Ekin_bin_calosample(*hitsim);
  
  m_chain.push_back(hitsim);
}

void TFCSLateralShapeParametrizationHitChain::set_geometry(ICaloGeometry* geo)
{
  TFCSLateralShapeParametrization::set_geometry(geo);
  if(m_number_of_hits_simul) m_number_of_hits_simul->set_geometry(geo);
}

int TFCSLateralShapeParametrizationHitChain::get_number_of_hits(TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol) const
{
  // TODO: should we still do it?
  if(m_number_of_hits_simul) {
    int n=m_number_of_hits_simul->get_number_of_hits(simulstate,truth,extrapol);
    if(n<1) n=1;
    return n;
  }
  for(TFCSLateralShapeParametrizationHitBase* hitsim : m_chain) {
    int n=hitsim->get_number_of_hits(simulstate,truth,extrapol);
    if(n>0) return n;
  } 
  return 1;
}

FCSReturnCode TFCSLateralShapeParametrizationHitChain::simulate(TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol)
{
  // Call get_number_of_hits() only once, as it could contain a random number
  int nhit = get_number_of_hits(simulstate, truth, extrapol);
  if (nhit <= 0) {
    ATH_MSG_ERROR("TFCSLateralShapeParametrizationHitChain::simulate(): number of hits could not be calculated");
    return FCSFatal;
  }

  float Ehit=simulstate.E(calosample())/nhit;

  bool debug = msgLvl(MSG::DEBUG);
  if (debug) {
    ATH_MSG_DEBUG("E("<<calosample()<<")="<<simulstate.E(calosample())<<" #hits="<<nhit);
  }

    auto start = std::chrono::system_clock::now();
#ifdef USE_GPU

  std::string sA[5]={"TFCSCenterPositionCalculation","TFCSValidationHitSpy","TFCSHistoLateralShapeParametrization",
	 "TFCSHitCellMappingWiggle", "TFCSValidationHitSpy" } ;
  std::string sB[3]={"TFCSCenterPositionCalculation","TFCSHistoLateralShapeParametrization",
	 "TFCSHitCellMappingWiggle" } ;
  std::string sC[3]={"TFCSCenterPositionCalculation","TFCSHistoLateralShapeParametrization",
	 "TFCSHitCellMapping" } ;
 
 if(debug) {
  std::cout<<"---xxx---nhits="<< nhit << ", " ;
  for(TFCSLateralShapeParametrizationHitBase* hitsim : m_chain)
      std::cout << "-----In TFCSLateralShapeParametizationHitChain:" << typeid( * hitsim ).name() << " "<< hitsim <<std::endl ;
  }
  int ichn=0 ;
  bool  our_chainA=true;
  bool  our_chainB=true;
  bool  our_chainC=true;
  for(TFCSLateralShapeParametrizationHitBase* hitsim : m_chain) {
      if (std::string(typeid( * hitsim ).name()).find(sA[ichn++]) == std::string::npos) 
	   { our_chainA= false ; break ; }
  } 
  ichn=0 ;
  for(TFCSLateralShapeParametrizationHitBase* hitsim : m_chain) {
      if (std::string(typeid( * hitsim ).name()).find(sB[ichn++]) == std::string::npos) 
	   { our_chainB= false ; break ; }
  } 
  ichn=0 ;
  for(TFCSLateralShapeParametrizationHitBase* hitsim : m_chain) {
      if (std::string(typeid( * hitsim ).name()).find(sC[ichn++]) == std::string::npos) 
	   { our_chainC= false ; break ; }
  } 
      
   
    //if ( (our_chainA || our_chainB ) ) {
    if ( (our_chainA || our_chainB || (our_chainC && nhit >1000 )) ) {
	  int cs = calosample();

         GeoLoadGpu * gld = (GeoLoadGpu *) simulstate.get_geold() ;	  

	TFCSSimulationState::EventStatus* es= simulstate.get_es() ;
	(*es).hits=nhit ;
	(*es).tot_hits+=nhit ;
	int n_simbins = (*es).n_simbins ;
	(*es).simbins[n_simbins]=(*es).tot_hits ;

	HitParams * htparams= (HitParams *) ((*es).hitparams) ;
	htparams[n_simbins].index= (*es).index ;
	htparams[n_simbins].cs= cs ;
	htparams[n_simbins].pdgId= truth->pdgid()  ;
	htparams[n_simbins].charge = HepPDT::ParticleID(htparams[n_simbins].pdgId).charge() ;
	htparams[n_simbins].E = Ehit ;
	htparams[n_simbins].nhits = nhit ;
        htparams[n_simbins].extrapol_eta_ent=extrapol->eta(cs, SUBPOS_ENT) ;
        htparams[n_simbins].extrapol_eta_ext=extrapol->eta(cs, SUBPOS_EXT) ;
        htparams[n_simbins].extrapol_phi_ent=extrapol->phi(cs, SUBPOS_ENT) ;
        htparams[n_simbins].extrapol_phi_ext=extrapol->phi(cs, SUBPOS_EXT) ;
        htparams[n_simbins].extrapol_r_ent=extrapol->r(cs, SUBPOS_ENT) ;
        htparams[n_simbins].extrapol_r_ext=extrapol->r(cs, SUBPOS_EXT) ;
        htparams[n_simbins].extrapol_z_ent=extrapol->z(cs, SUBPOS_ENT) ;
        htparams[n_simbins].extrapol_z_ext=extrapol->z(cs, SUBPOS_EXT) ;
	htparams[n_simbins].cmw = false;

	(*es).n_simbins += 1 ;
	(*es).gpu=true ;

 	ichn=0 ;
 	for( auto hitsim : m_chain ) {

	std::string s= std::string(typeid( * hitsim ).name()) ;
	
	if(s.find("TFCSCenterPositionCalculation") != std::string::npos ) {
        htparams[n_simbins].extrapWeight=((TFCSCenterPositionCalculation *)hitsim)->getExtrapWeight() ;

	}
/*
	if(s.find("TFCSValidationHitSpy") != std::string::npos ) {
	 args.spy=true ;
	  TFCSValidationHitSpy * hspy_ptr= (TFCSValidationHitSpy * ) hitsim ;
	 if(ichn==4) hsp2= hspy_ptr ;
	 else hsp1=  hspy_ptr  ;
	 if(0) { 
		std::cout<<"---m_previous"<< ((TFCSValidationHitSpy*)hitsim)->previous() << std::endl ;
		std::cout<<"---m_saved_hit"<< &(((TFCSValidationHitSpy*)hitsim)->saved_hit()) << std::endl ;
		std::cout<<"---m_saved_cellele"<< ((TFCSValidationHitSpy*)hitsim)->saved_cellele() << std::endl ;
		std::cout<<"---m_hist_hitgeo_dphi"<< ((TFCSValidationHitSpy*)hitsim)->hist_hitgeo_dphi() << std::endl ;
		std::cout<<"---m_hist_hitgeo_matchprevious_dphi"<< ((TFCSValidationHitSpy*)hitsim)->hist_hitgeo_matchprevious_dphi() << std::endl ;
		std::cout<<"---m_hist_hitenergy_r"<< ((TFCSValidationHitSpy*)hitsim)->hist_hitenergy_r() << std::endl ;
		std::cout<<"---m_hist_hitenergy_z"<< ((TFCSValidationHitSpy*)hitsim)->hist_hitenergy_z() << std::endl ;
		std::cout<<"---m_hist_hitenergy_weight"<< ((TFCSValidationHitSpy*)hitsim)->hist_hitenergy_weight() << std::endl ;
		std::cout<<"---m_hist_hitenergy_mean_r"<< ((TFCSValidationHitSpy*)hitsim)->hist_hitenergy_mean_r() << std::endl ;
		std::cout<<"---m_hist_hitenergy_mean_z"<< ((TFCSValidationHitSpy*)hitsim)->hist_hitenergy_mean_z() << std::endl ;
		std::cout<<"---m_hist_hitenergy_mean_weight"<< ((TFCSValidationHitSpy*)hitsim)->hist_hitenergy_mean_weight() << std::endl ;
		std::cout<<"---m_hist_hitenergy_alpha_radius"<< ((TFCSValidationHitSpy*)hitsim)->hist_hitenergy_alpha_radius() << std::endl ;
		std::cout<<"---m_hist_hitenergy_alpha_absPhi_radius"<< ((TFCSValidationHitSpy*)hitsim)->hist_hitenergy_alpha_absPhi_radius() << std::endl ;
		std::cout<<"---m_hist_deltaEta"<< ((TFCSValidationHitSpy*)hitsim)->hist_deltaEta() << std::endl ;
		std::cout<<"---m_hist_deltaPhi"<< ((TFCSValidationHitSpy*)hitsim)->hist_deltaPhi() << std::endl ;
		std::cout<<"---m_hist_deltaRt"<< ((TFCSValidationHitSpy*)hitsim)->hist_deltaRt() << std::endl ;
		std::cout<<"---m_hist_deltaZ"<< ((TFCSValidationHitSpy*)hitsim)->hist_deltaZ() << std::endl ;
		std::cout<<"---m_hist_total_dphi"<< ((TFCSValidationHitSpy*)hitsim)->hist_total_dphi() << std::endl ;
		std::cout<<"---m_hist_matched_dphi"<< ((TFCSValidationHitSpy*)hitsim)->hist_matched_dphi() << std::endl ;
		std::cout<<"---m_hist_total_dphi_etaboundary"<< ((TFCSValidationHitSpy*)hitsim)->hist_total_dphi_etaboundary() << std::endl ;
		std::cout<<"---m_hist_matched_dphi_etaboundary"<< ((TFCSValidationHitSpy*)hitsim)->hist_matched_dphi_etaboundary() << std::endl ;
		std::cout<<"---m_hist_Rz"<< ((TFCSValidationHitSpy*)hitsim)->hist_Rz() << std::endl ;
		std::cout<<"---m_hist_Rz_outOfRange"<< ((TFCSValidationHitSpy*)hitsim)->hist_Rz_outOfRange() << std::endl ;
		std::cout<<"---m_get_deta_hit_minus_extrapol_mm"<< ((TFCSValidationHitSpy*)hitsim)->get_deta_hit_minus_extrapol_mm() << std::endl ;
		std::cout<<"---m_get_dphi_hit_minus_extrapol_mm"<< ((TFCSValidationHitSpy*)hitsim)->get_dphi_hit_minus_extrapol_mm() << std::endl ;
		std::cout<<"---m_phi_granularity_change_at_eta"<< ((TFCSValidationHitSpy*)hitsim)->get_eta_boundary() << std::endl ;
	  }
	  int hs_i = (ichn==4) ? 1 : 0 ;
	  hs[hs_i].hist_hitgeo_dphi.nbin=  hspy_ptr ->hist_hitgeo_dphi()->GetNbinsX() ;
	  hs[hs_i].hist_hitgeo_matchprevious_dphi.nbin=  hspy_ptr ->hist_hitgeo_matchprevious_dphi()->GetNbinsX() ;
	  hs[hs_i].hist_hitgeo_dphi.low=  hspy_ptr ->hist_hitgeo_dphi()->GetXaxis()->GetXmin() ;
	  hs[hs_i].hist_hitgeo_matchprevious_dphi.low=  hspy_ptr ->hist_hitgeo_matchprevious_dphi()->GetXaxis()->GetXmin() ;
	  hs[hs_i].hist_hitgeo_dphi.up=  hspy_ptr ->hist_hitgeo_dphi()->GetXaxis()->GetXmax() ;
	  hs[hs_i].hist_hitgeo_matchprevious_dphi.up=  hspy_ptr ->hist_hitgeo_matchprevious_dphi()->GetXaxis()->GetXmax() ;
	  if(hs_i ==0 )args.hs1=hs[hs_i] ;
	  else args.hs2=hs[hs_i] ;
         if(0) {
	 std::cout << "hs["<<hs_i<<"].hist_hitgeo_dphi.nbin" <<hs[hs_i].hist_hitgeo_dphi.nbin <<std::endl ;
	 std::cout << "hs["<<hs_i<<"].hist_hitgeo_matchprevious_dphi.nbin" <<hs[hs_i].hist_hitgeo_matchprevious_dphi.nbin <<std::endl ;
	 std::cout << "hs["<<hs_i<<"].hist_hitgeo_matchprevious_dphi.low" <<hs[hs_i].hist_hitgeo_matchprevious_dphi.low <<std::endl ;
	 std::cout << "hs["<<hs_i<<"].hist_hitgeo_matchprevious_dphi.up" <<hs[hs_i].hist_hitgeo_matchprevious_dphi.up <<std::endl ;
	 std::cout << "hs["<<hs_i<<"].hist_hitgeo_dphi.low" <<hs[hs_i].hist_hitgeo_dphi.low <<std::endl ;
	 std::cout << "hs["<<hs_i<<"].hist_hitgeo_dphi.up" <<hs[hs_i].hist_hitgeo_dphi.up <<std::endl ;
	}
	      args.isBarrel = ((TFCSValidationHitSpy * ) hitsim )->get_geometry()->isCaloBarrel(cs) ;
	//	std::cout<<"isBarrel="<<  args.isBarrel << std::endl ; 
	}

*/
	if(s.find("TFCSHistoLateralShapeParametrization") != std::string::npos ) {
//  auto t3 = std::chrono::system_clock::now();

		((TFCSHistoLateralShapeParametrization *) hitsim)->LoadHistFuncs() ;
	//std::cout<<"2D funtion size:"<<" , ChainB:"<<our_chainB<<", "<<"nhits="<<nhit<<", "<< ((TFCSHistoLateralShapeParametrization *) hitsim)->LdFH()->hf2d_d()->nbinsx <<", " << ((TFCSHistoLateralShapeParametrization *) hitsim)->LdFH()->hf2d_d()->nbinsy <<std::endl ;
//  auto t4 = std::chrono::system_clock::now();
//    TFCSShapeValidation::time_h += (t4-t3) ;

		htparams[n_simbins].f2d = ((TFCSHistoLateralShapeParametrization *) hitsim)->LdFH()->d_hf2d() ;
		htparams[n_simbins].is_phi_symmetric=((TFCSHistoLateralShapeParametrization *) hitsim)->is_phi_symmetric() ;
	}
	if(s.find("TFCSHitCellMappingWiggle") != std::string::npos ) {
		((TFCSHitCellMappingWiggle * ) hitsim )->LoadHistFuncs() ;
	 	htparams[n_simbins].f1d = ((TFCSHitCellMappingWiggle * ) hitsim )->LdFH()->d_hf();
	 	htparams[n_simbins].cmw = true;
	} 
		
	ichn++ ;
	}

/*
	CaloGpuGeneral::simulate_hits(Ehit, nhit, args) ;
	
	for (int ii=0; ii<args.ct; ++ii) {
        //std::cout<<"celleleIndex="<< args.hitcells_h[ii]<<" " << args.hitcells_ct_h[ii]<<std::endl;
		
		const CaloDetDescrElement * cellele = gld->index2cell(args.hitcells_E_h[ii].cellid) ;
		simulstate.deposit(cellele ,args.hitcells_E_h[ii].energy) ;
	}

	if(args.spy && args.is_last) {
	//push back the Hitspy histograms

        	hsp1 ->hist_hitgeo_dphi()->SetError(args.hs1.hist_hitgeo_dphi.sumw2_array_h) ;
        	hsp1 ->hist_hitgeo_dphi()->SetEntries((double)args.hs1.hist_hitgeo_dphi.nentries) ;
        	hsp1 ->hist_hitgeo_dphi()->PutStats(&args.hs_sumwx_h[0]) ;
		
        	hsp2 ->hist_hitgeo_dphi()->SetContent(args.hs2.hist_hitgeo_dphi.ct_array_h) ;
        	hsp2 ->hist_hitgeo_dphi()->SetError(args.hs2.hist_hitgeo_dphi.sumw2_array_h) ;
        	hsp2 ->hist_hitgeo_dphi()->SetEntries((double)args.hs2.hist_hitgeo_dphi.nentries) ;
        	hsp2 ->hist_hitgeo_dphi()->PutStats(&args.hs_sumwx_h[4]) ;
		
        	hsp2 ->hist_hitgeo_matchprevious_dphi()->SetContent(args.hs2.hist_hitgeo_matchprevious_dphi.ct_array_h) ;
        	hsp2 ->hist_hitgeo_matchprevious_dphi()->SetError(args.hs2.hist_hitgeo_matchprevious_dphi.sumw2_array_h) ;
        	hsp2 ->hist_hitgeo_matchprevious_dphi()->SetEntries((double)args.hs2.hist_hitgeo_matchprevious_dphi.nentries) ;
        	hsp2 ->hist_hitgeo_matchprevious_dphi()->PutStats(&args.hs_sumwx_h[8]) ;

		
	}
*/  
   } else {
#endif
if(debug )std::cout<<"Host Nhits: "<<nhit << std::endl ;
  for (int i = 0; i < nhit; ++i) {
    TFCSLateralShapeParametrizationHitBase::Hit hit; 
    hit.E()=Ehit;
    for(TFCSLateralShapeParametrizationHitBase* hitsim : m_chain) {
      if (debug) {
        if (i < 2) hitsim->setLevel(MSG::DEBUG);
        else hitsim->setLevel(MSG::INFO);
      }

      for (int i = 0; i <= FCS_RETRY_COUNT; i++) {
        if (i > 0) ATH_MSG_WARNING("TFCSLateralShapeParametrizationHitChain::simulate(): Retry simulate_hit call " << i << "/" << FCS_RETRY_COUNT);
  
        FCSReturnCode status = hitsim->simulate_hit(hit, simulstate, truth, extrapol);

        if (status == FCSSuccess)
          break;
        else if (status == FCSFatal)
          return FCSFatal;

        if (i == FCS_RETRY_COUNT) {
          ATH_MSG_ERROR("TFCSLateralShapeParametrizationHitChain::simulate(): simulate_hit call failed after " << FCS_RETRY_COUNT << "retries");
        }
      }
    }
  }
#ifdef USE_GPU
  }
  
  auto t2 = std::chrono::system_clock::now();
    if ( our_chainA  ) {
    TFCSShapeValidation::time_g1 += (t2-start) ;
//   } else if ( our_chainB || our_chainC) {
   } else if ( our_chainB || (our_chainC && nhit>1000 ) ){
     TFCSShapeValidation::time_g2 += (t2-start) ;
   } else
#endif

  {
  auto t2 = std::chrono::system_clock::now();
    TFCSShapeValidation::time_h += (t2-start) ;
   }

  return FCSSuccess;
}

void TFCSLateralShapeParametrizationHitChain::Print(Option_t *option) const
{
  TFCSLateralShapeParametrization::Print(option);
  TString opt(option);
  bool shortprint=opt.Index("short")>=0;
  bool longprint=msgLvl(MSG::DEBUG) || (msgLvl(MSG::INFO) && !shortprint);
  TString optprint=opt;optprint.ReplaceAll("short","");

  if(m_number_of_hits_simul) {
    if(longprint) ATH_MSG_INFO(optprint <<"#:Number of hits simulation:");
    m_number_of_hits_simul->Print(opt+"#:");
  }
  if(longprint) ATH_MSG_INFO(optprint <<"- Simulation chain:");
  char count='A';
  for(TFCSLateralShapeParametrizationHitBase* hitsim : m_chain) {
    hitsim->Print(opt+count+" ");
    count++;
  } 
}


