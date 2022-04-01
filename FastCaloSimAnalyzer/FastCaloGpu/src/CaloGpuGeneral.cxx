/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "CaloGpuGeneral.h"
#include "CaloGpuGeneral_cu.h"
#include "CaloGpuGeneral_kk.h"

#include "Rand4Hits.h"
#include <chrono>
#include <iostream>

void* CaloGpuGeneral::Rand4Hits_init( long long maxhits, unsigned short maxbin, unsigned long long seed,
                                      bool /*hitspy*/ ) {

  auto       t0   = std::chrono::system_clock::now();
  Rand4Hits* rd4h = new Rand4Hits;
  auto       t1   = std::chrono::system_clock::now();

  // By default, generate random numbers on GPU, unless macro RNDGEN_CPU is set
  // This is controlled by cmake parameter -DRNDGEN_CPU
#ifndef RNDGEN_CPU
  constexpr bool genOnCPU{false};
#else
  constexpr bool genOnCPU{true};
#endif
  
  auto t2 = std::chrono::system_clock::now();
  // use CPU rand num gen to be able to compare GPU implementations
  rd4h->create_gen( seed, 3 * maxhits, genOnCPU );
  auto t3 = std::chrono::system_clock::now();
//  rd4h->set_t_a_hits( maxhits );
//  rd4h->set_c_hits( 0 );
  auto t4 = std::chrono::system_clock::now();

  rd4h->allocate_simulation( maxhits, maxbin, 2000, 200000 );
  auto t5 = std::chrono::system_clock::now();

  std::chrono::duration<double> diff1 = t1 - t0;
  std::chrono::duration<double> diff2 = t2 - t1;
  std::chrono::duration<double> diff3 = t3 - t2;
  std::chrono::duration<double> diff4 = t4 - t3;
  std::chrono::duration<double> diff5 = t5 - t4;
  std::cout << "Time of R4hit: " << diff1.count() << "," << diff2.count() << "," << diff3.count() << ","
            << diff4.count() << "," << diff5.count() << " s" << std::endl;

  return (void*)rd4h;
}

void CaloGpuGeneral::Rand4Hits_finish( void* rd4h ) {
  if ( (Rand4Hits*)rd4h ) delete (Rand4Hits*)rd4h;
}

void CaloGpuGeneral::simulate_hits( float E, int nhits, Chain0_Args& args ) {

//  Rand4Hits* rd4h = (Rand4Hits*)args.rd4h;
//
//  float* r = rd4h->rand_ptr( nhits );
//
//  rd4h->add_a_hits( nhits );
//  args.rand = r;
//
//  args.maxhitct = MAXHITCT;
//
//  args.cells_energy = rd4h->get_cells_energy(); // Hit cell energy map , size of ncells(~200k float)
//  args.hitcells_E   = rd4h->get_cell_e();       // Hit cell energy map, moved together
//  args.hitcells_E_h = rd4h->get_cell_e_h();     // Host array
//
//  args.hitcells_ct = rd4h->get_ct(); // single value, number of  uniq hit cells
//
//#ifndef USE_KOKKOS
//  CaloGpuGeneral_cu::simulate_hits( E, nhits, args );
//#else
//  CaloGpuGeneral_kk::simulate_hits( E, nhits, args );
//#endif
}
