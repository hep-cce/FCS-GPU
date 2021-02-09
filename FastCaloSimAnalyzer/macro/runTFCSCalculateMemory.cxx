/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "runTFCSMergeParamEtaSlices.cxx"

void runTFCSCalculateMemory( std::string file = "TFCSparam_v008.root", int int_Emin = 64, int int_Emax = 4194304,
                             double etamin = 0., double etamax = 5 ) {

  const float toMB         = 1.f / 1024.f;
  double      tot_res_mem  = 0.;
  double      tot_virt_mem = 0.;

  double tot_res_mem_before  = 0.;
  double tot_virt_mem_before = 0.;

  double tot_res_mem_after  = 0.;
  double tot_virt_mem_after = 0.;

  TFile* fullchainfile = TFile::Open( file.c_str() );

  static ProcInfo_t info;
  gSystem->GetProcInfo( &info );
  tot_res_mem_before += info.fMemResident * toMB;
  tot_virt_mem_before += info.fMemVirtual * toMB;

  cout << " -----------------------------------------" << endl;
  printf( " res  memory before  = %g Mbytes\n", tot_res_mem_before );
  printf( " vir  memory before = %g Mbytes\n", tot_virt_mem_before );
  cout << " -----------------------------------------" << endl;

  cout << "======================================" << endl;
  cout << "==== Now reading the full file ====" << endl;
  cout << "======================================" << endl;
  TFCSParametrizationBase* testread = (TFCSParametrizationBase*)fullchainfile->Get( "SelPDGID" );
  gSystem->GetProcInfo( &info );
  tot_res_mem_after += info.fMemResident * toMB;
  tot_virt_mem_after += info.fMemVirtual * toMB;

  cout << " -----------------------------------------" << endl;
  printf( " res  memory after = %g Mbytes\n", tot_res_mem_after );
  printf( " vir  memory after = %g Mbytes\n", tot_virt_mem_after );
  cout << " -----------------------------------------" << endl;

  tot_res_mem  = tot_res_mem_after - tot_res_mem_before;
  tot_virt_mem = tot_virt_mem_after - tot_virt_mem_before;

  // testread->Print("short");

  fullchainfile->Close();

  cout << " -----------------------------------------" << endl;
  printf( " res  memory  = %g Mbytes\n", tot_res_mem );
  printf( " vir  memory = %g Mbytes\n", tot_virt_mem );
  cout << " -----------------------------------------" << endl;
}
