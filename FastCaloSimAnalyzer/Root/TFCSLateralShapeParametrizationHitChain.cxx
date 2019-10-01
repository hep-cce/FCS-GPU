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

#include "FastCaloSimAnalyzer/TFCSShapeValidation.h"

#include "FastCaloGpu/FastCaloGpu/CaloGpuGeneral.h"
#include "FastCaloGpu/FastCaloGpu/GeoLoadGpu.h"
#include "FastCaloGpu/FastCaloGpu/Args.h"
#include "HepPDT/ParticleData.hh"
#include "HepPDT/ParticleDataTable.hh"
#include "ISF_FastCaloSimEvent/TFCSExtrapolationState.h"

#include <chrono>



#endif


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

#ifdef USE_GPU

    auto start = std::chrono::system_clock::now();
  std::string sA[5]={"TFCSCenterPositionCalculation","TFCSValidationHitSpy","TFCSHistoLateralShapeParametrization",
	 "TFCSHitCellMappingWiggle", "TFCSValidationHitSpy" } ;
  std::string sB[3]={"TFCSCenterPositionCalculation","TFCSHistoLateralShapeParametrization",
	 "TFCSHitCellMappingWiggle" } ;
 
 if(debug) {
  std::cout<<"---xxx---nhits="<< nhit << ", " ;
  for(TFCSLateralShapeParametrizationHitBase* hitsim : m_chain)
      std::cout << "-----In TFCSLateralShapeParametizationHitChain:" << typeid( * hitsim ).name() << " "<< hitsim <<std::endl ;
  std::cout << std::endl ;
  }
  int ichn=0 ;
  bool  our_chainA=true;
  bool  our_chainB=true;
  for(TFCSLateralShapeParametrizationHitBase* hitsim : m_chain) {
      if (std::string(typeid( * hitsim ).name()).find(sA[ichn++]) == std::string::npos) 
	   { our_chainA= false ; break ; }
  } 
  ichn=0 ;
  for(TFCSLateralShapeParametrizationHitBase* hitsim : m_chain) {
      if (std::string(typeid( * hitsim ).name()).find(sB[ichn++]) == std::string::npos) 
	   { our_chainB= false ; break ; }
  } 
      
   
    if ( nhit > MIN_GPU_HITS && (our_chainA || our_chainB) ) {
	  int cs = calosample();

         GeoLoadGpu * gld = (GeoLoadGpu *) simulstate.get_geold() ;	  

	Chain0_Args args ;
          
	  args.debug=debug ;
	  args.cs = cs ;
          args.extrapol_eta_ent=extrapol->eta(cs, SUBPOS_ENT) ;
          args.extrapol_eta_ext=extrapol->eta(cs, SUBPOS_EXT) ;
          args.extrapol_phi_ent=extrapol->phi(cs, SUBPOS_ENT) ;
          args.extrapol_phi_ext=extrapol->phi(cs, SUBPOS_EXT) ;
          args.extrapol_r_ent=extrapol->r(cs, SUBPOS_ENT) ;
          args.extrapol_r_ext=extrapol->r(cs, SUBPOS_EXT) ;
          args.extrapol_z_ent=extrapol->z(cs, SUBPOS_ENT) ;
          args.extrapol_z_ext=extrapol->z(cs, SUBPOS_EXT) ;

	  args.pdgId	= truth->pdgid();
	  args.charge   = HepPDT::ParticleID(args.pdgId).charge() ;
	  
	  args.nhits= nhit ;
	  args.rand =0 ;
	  args.geo = GeoLoadGpu::Geo_g ;
	  args.rd4h=simulstate.get_gpu_rand();
	  args.ncells=GeoLoadGpu::num_cells;
	  args.spy=false ;
	  args.isBarrel=false ;
	  Hitspy_Hist hs[2] ;
	  hs[0]= Hitspy_Hist() ;
          hs[1]= Hitspy_Hist() ;
	  
	  args.is_first=simulstate.get_es()->is_first ;
	  args.is_last=simulstate.get_es()->is_last ;

         TFCSValidationHitSpy* hsp1;
	TFCSValidationHitSpy* hsp2 ; 
	
 	ichn=0 ;
 	for( auto hitsim : m_chain ) {

	std::string s= std::string(typeid( * hitsim ).name()) ;
	
	if(s.find("TFCSCenterPositionCalculation") != std::string::npos ) {
  //         std::cout<<"---m_extrapWeight"<< ((TFCSCenterPositionCalculation *)hitsim)->getExtrapWeight() <<std::endl ;
         //  hitsim->Print();
           args.extrapWeight=((TFCSCenterPositionCalculation *)hitsim)->getExtrapWeight() ;

	}

//	if(ichn==1 || ichn==4 ) {
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

	//if(ichn==2) {
	if(s.find("TFCSHistoLateralShapeParametrization") != std::string::npos ) {

	//	TFCS2DFunctionHistogram h=((TFCSHistoLateralShapeParametrization *) hitsim)->histogram() ;
	//	std::cout << "size of hist: "<<h.get_HistoBordersx().size() <<", "<<h.get_HistoBordersy().size()  
	//		<<"Pointer: " << &h <<std::endl ;

		((TFCSHistoLateralShapeParametrization *) hitsim)->LoadHistFuncs() ;
	

		args.is_phi_symmetric=((TFCSHistoLateralShapeParametrization *) hitsim)->is_phi_symmetric() ;
		args.fh2d = ((TFCSHistoLateralShapeParametrization *) hitsim)->LdFH()->d_hf2d() ; 

	//std::cout<<"Hitsim_ptr="<<hitsim<<", Ld_FH_ptr="<<  ((TFCSHistoLateralShapeParametrization *) hitsim)->LdFH() <<",FH2d_ptr="<< args.fh2d <<std::endl ;
 
		args.fh2d_v = *(((TFCSHistoLateralShapeParametrization *) hitsim)->LdFH()->hf2d_d()) ;

	}
//	if(ichn==3) {
	if(s.find("TFCSHitCellMappingWiggle") != std::string::npos ) {
	if(0) {
		std::cout<< "---NumberOfBins:" << ((TFCSHitCellMappingWiggle * ) hitsim )->get_number_of_bins () << std::endl;
		std::vector< const TFCS1DFunction* > funcs=  ((TFCSHitCellMappingWiggle * ) hitsim )->get_functions() ;
		for ( auto it = funcs.begin(); it != funcs.end() ; ++it ) {
		
		std::cout<< "----+++type of funcs: " << typeid( *(*it)).name()<<", pointer: " << *it <<  std::endl ;  
		} 
		
	      }
	((TFCSHitCellMappingWiggle * ) hitsim )->LoadHistFuncs() ;

        args.fhs=((TFCSHitCellMappingWiggle * ) hitsim )->LdFH()->d_hf(); 

	}

	ichn++ ;

	}

//  auto t1 = std::chrono::system_clock::now();
//  std::chrono::duration<double> diff = t1-start;
//   std::cout <<  "Time before GPU simulate_hit :" << diff.count() <<" s" << std::endl ;

	CaloGpuGeneral::simulate_hits(Ehit, nhit, args) ;
	
	for (int ii=0; ii<args.ct; ++ii) {
        //std::cout<<"celleleIndex="<< args.hitcells_h[ii]<<" " << args.hitcells_ct_h[ii]<<std::endl;
		
		const CaloDetDescrElement * cellele = gld->index2cell(args.hitcells_h[ii]) ;
		simulstate.deposit(cellele ,Ehit*args.hitcells_ct_h[ii]) ;
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
  
auto t2 = std::chrono::system_clock::now();
//   diff = t2-t1;
 //  std::cout <<  "Time of GPU simulate_hit :" << diff.count() <<" s" <<" CT="<<args.ct<<  std::endl ;
 //   TFCSShapeValidation::time_g += (t2-start) ;
   } else {
#endif
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
    if ( nhit >MIN_GPU_HITS  && our_chainA ) {
    TFCSShapeValidation::time_g1 += (t2-start) ;
   } else if (nhit >MIN_GPU_HITS && our_chainB) {
     TFCSShapeValidation::time_g2 += (t2-start) ;
   } else  {
  auto t2 = std::chrono::system_clock::now();
    TFCSShapeValidation::time_h += (t2-start) ;
   }
#endif
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


#ifdef USE_GPU
void gpu_hit_chain() {

}

#endif
