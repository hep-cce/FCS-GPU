/*
  Copyright (C) 2002-2021 CERN for the benefit of the ATLAS collaboration
*/

#include "CaloGpuGeneral_al.h"
#include "AlpakaDefs.h"
#include <alpaka/alpaka.hpp>
#include <iostream>

#define BLOCK_SIZE 256

using namespace CaloGpuGeneral_fnc;

static CaloGpuGeneral::KernelTime timing;
static bool first{true};

namespace CaloGpuGeneral_al {


  void Rand4Hits_finish( void* rd4h ) {

    if ( (Rand4Hits*)rd4h ) delete (Rand4Hits*)rd4h;
    
    if (timing.count > 0) {
      std::cout << "kernel timing\n";
      std::cout << timing;
    } else {
      std::cout << "no kernel timing available" << std::endl;
    }

  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  struct SimulateAKernel
  {
    template<typename TAcc, typename Chain0_Args>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc
				  , float E
				  , int nhits
				  , Chain0_Args args
				  ) const -> void
    {
      auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      if(idx < nhits) {
	Hit hit;
	hit.E() = E;
	
	CenterPositionCalculation_d( hit, args );
	HistoLateralShapeParametrization_d( hit, idx, args );
	HitCellMappingWiggle_d( hit, args, idx );
      }
    }
  };

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  struct SimulateCtKernel
  {
    template<typename TAcc, typename Chain0_Args>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc
				  , Chain0_Args args
				  ) const -> void
    {
      auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      if(idx < args.ncells) {
	if ( args.cells_energy[idx] > 0 ) {
	  unsigned int ct = atomicAdd( args.hitcells_ct, 1 );
	  Cell_E ce;
	  ce.cellid           = idx;
	  ce.energy           = args.cells_energy[idx];
	  args.hitcells_E[ct] = ce;
	}
      }
    }
  };

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  struct SimulateCleanKernel
  {
    template<typename TAcc, typename Chain0_Args>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc
				  , Chain0_Args args
				  ) const -> void
    {
      auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      if(idx < args.ncells) {
	args.cells_energy[idx] = 0.0;
      }
      if(idx == 0) {
	args.hitcells_ct[0] = 0;
      }
    }
  };


  auto simulate_clean_alpaka(Chain0_Args& args, QueueAcc& queue) -> void {

    int  blocksize   = BLOCK_SIZE;
    int  threads_tot = args.ncells;
    int  nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;
    WorkDiv workdiv{static_cast<Idx>(nblocks)
	, static_cast<Idx>(blocksize)
	, static_cast<Idx>(1)};

    SimulateCleanKernel simulateClean;
    alpaka::exec<Acc>(queue
		      , workdiv
		      , simulateClean
		      , args);
    alpaka::wait(queue);
  }

  auto simulate_A_alpaka( float E, int nhits, Chain0_Args& args, QueueAcc& queue) -> void {

    int blocksize   = BLOCK_SIZE;
    int threads_tot = nhits;
    int nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;
    WorkDiv workdiv{static_cast<Idx>(nblocks)
	, static_cast<Idx>(blocksize)
	, static_cast<Idx>(1)};

    SimulateAKernel simulateA;
    alpaka::exec<Acc>(queue
                      , workdiv
                      , simulateA
		      , E
		      , nhits
                      , args);
    alpaka::wait(queue);
  }

  auto simulate_ct_alpaka(Chain0_Args& args, QueueAcc& queue) -> void {

    int  blocksize   = BLOCK_SIZE;
    int  threads_tot = args.ncells;
    int  nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;
    WorkDiv workdiv{static_cast<Idx>(nblocks)
	, static_cast<Idx>(blocksize)
	, static_cast<Idx>(1)};

    SimulateCtKernel simulateCt;
    alpaka::exec<Acc>(queue
                      , workdiv
                      , simulateCt
                      , args);
    alpaka::wait(queue);
  }

  void simulate_hits( float E, int nhits, Chain0_Args& args ) {

    QueueAcc queue(alpaka::getDevByIdx<Acc>(Idx{0}));

    auto          t0   = std::chrono::system_clock::now();
    simulate_clean_alpaka(args,queue);
    
    auto t1     = std::chrono::system_clock::now();
    simulate_A_alpaka(E, nhits, args,queue);

    auto t2 = std::chrono::system_clock::now();
    simulate_ct_alpaka(args,queue);

    int ct;
    auto t3 = std::chrono::system_clock::now();
    cudaMemcpy( &ct, args.hitcells_ct, sizeof( int ), cudaMemcpyDeviceToHost );
    cudaMemcpy( args.hitcells_E_h, args.hitcells_E, ct * sizeof( Cell_E ), cudaMemcpyDeviceToHost);

    auto t4 = std::chrono::system_clock::now();
    // pass result back
    args.ct = ct;

#ifdef DUMP_HITCELLS
    std::cout << "hitcells: " << args.ct << "  nhits: " << nhits << "  E: " << E << "\n";
    std::map<unsigned int,float> cm;
    for (int i=0; i<args.ct; ++i) {
      cm[args.hitcells_E_h[i].cellid] = args.hitcells_E_h[i].energy;
    }
    for (auto &em: cm) {
      std::cout << "  cell: " << em.first << "  " << em.second << std::endl;
    }
#endif
    
    CaloGpuGeneral::KernelTime kt( t1 - t0, t2 - t1, t3 - t2, t4 - t3 );
    if (first) {
      first = false;
    } else{ 
      timing += kt;
    }
  }
} // namespace CaloGpuGeneral_al
