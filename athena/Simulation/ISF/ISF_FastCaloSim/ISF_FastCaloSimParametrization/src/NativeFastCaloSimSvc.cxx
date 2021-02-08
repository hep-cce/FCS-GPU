/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

///////////////////////////////////////////////////////////////////
// FastCaloSimSvc.cxx, (c) ATLAS Detector software
///////////////////////////////////////////////////////////////////

// class header include
#include "ISF_FastCaloSimParametrization/NativeFastCaloSimSvc.h"

// StoreGate
#include "StoreGate/StoreGateSvc.h"
#include "ISF_Interfaces/IParticleBroker.h"

// ISF includes
#include "ISF_Event/ISFParticle.h"
#include "ISF_Event/ISFParticleContainer.h"

// HepMC include needed for FastCaloSim
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"
#include "HepMC/SimpleVector.h"
#include "CLHEP/Units/SystemOfUnits.h"
// McEventCollection
#include "GeneratorObjects/McEventCollection.h"
// Calo Cell includes
#include "CaloEvent/CaloCell.h"
#include "CaloEvent/CaloCellContainer.h"
#include "NavFourMom/INavigable4MomentumCollection.h"
// PunchThrough Tool
#include "ISF_FastCaloSimInterfaces/IPunchThroughTool.h"

/** Constructor **/
ISF::NativeFastCaloSimSvc::NativeFastCaloSimSvc(const std::string& name,ISvcLocator* svc) :
  BaseSimulationSvc(name, svc),
  m_ownPolicy(static_cast<int>(SG::VIEW_ELEMENTS)),
  m_batchProcessMcTruth(false),
  m_simulateUndefinedBCs(false),
  m_caloCellsOutputName("AllCalo"),
  m_caloCellHack(false),
  m_doPunchThrough(false),
  m_caloCellMakerTools_setup(),
  m_caloCellMakerTools_simulate(),
  m_caloCellMakerTools_release(),
  m_punchThroughTool(""),
  m_theContainer(0),
  m_particleBroker ("ISF_ParticleBroker",name)
{
  // where to go 
  declareProperty("OwnPolicy",                         m_ownPolicy) ;
  declareProperty("CaloCellMakerTools_setup"   ,       m_caloCellMakerTools_setup) ;
  declareProperty("CaloCellMakerTools_simulate",       m_caloCellMakerTools_simulate) ;
  declareProperty("CaloCellMakerTools_release" ,       m_caloCellMakerTools_release) ;
  declareProperty("PunchThroughTool",                  m_punchThroughTool);
  declareProperty("CaloCellsOutputName",               m_caloCellsOutputName) ;
  declareProperty("CaloCellHack",                      m_caloCellHack) ;
  declareProperty("DoPunchThroughSimulation", 	       m_doPunchThrough) ;
  declareProperty("SimulateUndefinedBarcodeParticles",
                  m_simulateUndefinedBCs,
                  "Whether or not to simulate paritcles with undefined barcode" );
  declareProperty("ParticleBroker",
                  m_particleBroker,
                  "ISF ParticleBroker Svc" );
  declareProperty("BatchProcessMcTruth",
                  m_batchProcessMcTruth=false,
                  "Run the FastShowerCellBuilders on the McTruth at the end of the event" );
}

ISF::NativeFastCaloSimSvc::~NativeFastCaloSimSvc() 
{}

/** framework methods */
StatusCode ISF::NativeFastCaloSimSvc::initialize()
{
   ATH_MSG_INFO ( m_screenOutputPrefix << "Initializing ...");

   // access tools and store them
   if ( retrieveTools<ICaloCellMakerTool>(m_caloCellMakerTools_setup).isFailure() ) 
        return StatusCode::FAILURE;
   if ( retrieveTools<ICaloCellMakerTool>(m_caloCellMakerTools_simulate).isFailure() ) 
        return StatusCode::FAILURE;
   if ( retrieveTools<ICaloCellMakerTool>(m_caloCellMakerTools_release).isFailure() ) 
        return StatusCode::FAILURE;

   if (m_doPunchThrough && m_punchThroughTool.retrieve().isFailure() ) 
   {
     ATH_MSG_ERROR (m_punchThroughTool.propertyName() << ": Failed to retrieve tool " << m_punchThroughTool.type());
     return StatusCode::FAILURE;
   } 
 
   ATH_MSG_DEBUG( m_screenOutputPrefix << " Output CaloCellContainer Name " << m_caloCellsOutputName );
   if (m_ownPolicy==SG::OWN_ELEMENTS){
       ATH_MSG_INFO( m_screenOutputPrefix << "...will OWN its cells." );
     }
     else
     {
       ATH_MSG_INFO( m_screenOutputPrefix << "...will VIEW its cells." );
     }

   if (m_caloCellHack){
     ATH_MSG_WARNING( m_screenOutputPrefix << " CaloCellContainer: " << m_caloCellsOutputName << "will be read in and modified !. To be used with care. " );
   }

  if ( m_particleBroker.retrieve().isFailure()) {
    ATH_MSG_FATAL ("Could not retrieve ISF ParticleBroker service " <<m_particleBroker);
    return StatusCode::FAILURE;
  }

   return StatusCode::SUCCESS;
}

/** framework methods */
StatusCode ISF::NativeFastCaloSimSvc::finalize()
{
    ATH_MSG_INFO ( m_screenOutputPrefix << "Finalizing ...");
    return StatusCode::SUCCESS;
}

StatusCode ISF::NativeFastCaloSimSvc::setupEvent()
{ 
  ATH_MSG_DEBUG ( m_screenOutputPrefix << "setup Event");
  
  if (!m_caloCellHack) {
    
    m_theContainer = new CaloCellContainer(static_cast<SG::OwnershipPolicy>(m_ownPolicy));

    StatusCode sc=StatusCode::SUCCESS;
    sc=evtStore()->record(m_theContainer,m_caloCellsOutputName);
     
    if (sc.isFailure())  {
      ATH_MSG_FATAL( m_screenOutputPrefix << "cannot record CaloCellContainer " << m_caloCellsOutputName );
      return StatusCode::FAILURE;
    }

    // also symLink as INavigable4MomentumCollection!
    INavigable4MomentumCollection* theNav4Coll = 0;
    sc = evtStore()->symLink(m_theContainer,theNav4Coll);

    if (sc.isFailure()) {
      ATH_MSG_WARNING( m_screenOutputPrefix << "Error symlinking CaloCellContainer to INavigable4MomentumCollection " );
      return StatusCode::FAILURE;
    }
  }
  else {
    // take CaloCellContainer from input and cast away constness
    const CaloCellContainer * theConstContainer ;
    
    StatusCode sc=StatusCode::SUCCESS;
    sc=evtStore()->retrieve(theConstContainer,m_caloCellsOutputName);
    if (sc.isFailure() || theConstContainer==0)
    {
      ATH_MSG_FATAL( m_screenOutputPrefix << "Could not retrieve CaloCellContainer " << m_caloCellsOutputName );
      return StatusCode::FAILURE;
    }
    m_theContainer = const_cast<CaloCellContainer *> (theConstContainer);
  }

  // loop on setup tools
  ToolHandleArray<ICaloCellMakerTool>::iterator itrTool=m_caloCellMakerTools_setup.begin();
  ToolHandleArray<ICaloCellMakerTool>::iterator endTool=m_caloCellMakerTools_setup.end();
  for (;itrTool!=endTool;++itrTool){
    ATH_MSG_DEBUG( m_screenOutputPrefix << "Calling tool " << itrTool->name() );   
    std::string chronoName=this->name()+"_"+ itrTool->name();
    
    if (m_chrono) m_chrono -> chronoStart( chronoName);
    StatusCode sc = (*itrTool)->process(m_theContainer);
    if (m_chrono) {
      m_chrono -> chronoStop( chronoName );
      ATH_MSG_DEBUG( m_screenOutputPrefix << "Chrono stop : delta " << m_chrono->chronoDelta (chronoName,IChronoStatSvc::USER ) * CLHEP::microsecond / CLHEP::second << " second " );
    }

    if (sc.isFailure()) {
      ATH_MSG_ERROR( m_screenOutputPrefix << "Error executing tool " << itrTool->name() );
    } 
  }

  /*
  // loop on simulate tools
  itrTool=m_caloCellMakerTools_simulate.begin();
  endTool=m_caloCellMakerTools_simulate.end();
  for (;itrTool!=endTool;++itrTool) {
    FastShowerCellBuilderTool* fcs=dynamic_cast< FastShowerCellBuilderTool* >(&(*(*itrTool)));
    if(fcs) {
      if(fcs->setupEvent().isFailure()) {
        ATH_MSG_ERROR( m_screenOutputPrefix << "Error executing tool " << itrTool->name() << " in setupEvent");
        return StatusCode::FAILURE;
      }  
    }
  }
  */

  return StatusCode::SUCCESS;
}

StatusCode ISF::NativeFastCaloSimSvc::releaseEvent()
{ 
  ATH_MSG_DEBUG ( m_screenOutputPrefix << "release Event");

  // the return value
  StatusCode sc = StatusCode::SUCCESS;

  // (1.) batch processing of all particles from McTruth
  //       (for MC12 parametrization)
  //
  if ( m_batchProcessMcTruth) {
    // -> run the FastShowerCellBuilder tools
    //        (in Python they should be configured to pick up the modified truth collection)
    
    //ZH commented out these to avoid warnings at the moment
    //ToolHandleArray<ICaloCellMakerTool>::iterator itrTool=m_caloCellMakerTools_simulate.begin();
    //ToolHandleArray<ICaloCellMakerTool>::iterator endTool=m_caloCellMakerTools_simulate.end();
    /*
    for (;itrTool!=endTool;++itrTool) {
      FastShowerCellBuilderTool* fcs=dynamic_cast< FastShowerCellBuilderTool* >(&(*(*itrTool)));
      if(!fcs) {
        ATH_MSG_WARNING( m_screenOutputPrefix << "tool " << itrTool->name()<< "is not a FastShowerCellBuilderTool" );   
        continue;
      }
      
      ATH_MSG_VERBOSE( m_screenOutputPrefix << "Calling tool " << itrTool->name() );   

      if( fcs->process(m_theContainer).isFailure()) {
        ATH_MSG_WARNING( m_screenOutputPrefix << "batch simulation of FastCaloSim particles failed" );   
        sc = StatusCode::FAILURE;
      }
    }
    */
  } // <-- end of batch particle processing

  // (2.) finalize simulation tools in a loop
  //
  ToolHandleArray<ICaloCellMakerTool>::iterator itrTool=m_caloCellMakerTools_simulate.begin();
  ToolHandleArray<ICaloCellMakerTool>::iterator endTool=m_caloCellMakerTools_simulate.end();
  /*
  for (;itrTool!=endTool;++itrTool) {
    FastShowerCellBuilderTool* fcs=dynamic_cast< FastShowerCellBuilderTool* >(&(*(*itrTool)));
    if(fcs) {
      if(fcs->releaseEvent().isFailure()) {
        ATH_MSG_ERROR( m_screenOutputPrefix << "Error executing tool " << itrTool->name() << " in releaseEvent");
        return StatusCode::FAILURE;
      }
    }
  }
  */
  
  // (3.) run release tools in a loop
  //
  itrTool=m_caloCellMakerTools_release.begin();
  endTool=m_caloCellMakerTools_release.end();
  for (;itrTool!=endTool;++itrTool){
    ATH_MSG_DEBUG( m_screenOutputPrefix << "Calling tool " << itrTool->name() );   
    std::string chronoName=this->name()+"_"+ itrTool->name();
    
    if (m_chrono) m_chrono -> chronoStart( chronoName);
    sc = (*itrTool)->process(m_theContainer);
    if (m_chrono) {
      m_chrono -> chronoStop( chronoName );
      ATH_MSG_DEBUG( m_screenOutputPrefix << "Chrono stop : delta " << m_chrono->chronoDelta (chronoName,IChronoStatSvc::USER ) * CLHEP::microsecond / CLHEP::second << " second " );
    }

    if (sc.isFailure()) {
      ATH_MSG_ERROR( m_screenOutputPrefix << "Error executing tool " << itrTool->name() );
    }
  }

  return StatusCode::SUCCESS; 
}


/** Simulation Call */
StatusCode ISF::NativeFastCaloSimSvc::simulate(const ISF::ISFParticle& isfp)
{
  // read the particle's barcode
  Barcode::ParticleBarcode bc = isfp.barcode();
//lets do punch-through here
//----------------------------------------------------------

  // punch-through simulation

  if (m_doPunchThrough) {
    // call punch-through simulation
    const ISF::ISFParticleContainer* isfpVec = m_punchThroughTool->computePunchThroughParticles(isfp);

    // add punch-through particles to the ISF particle broker
    if (isfpVec) {
      ISF::ISFParticleContainer::const_iterator partIt    = isfpVec->begin();
      ISF::ISFParticleContainer::const_iterator partItEnd = isfpVec->end();
      for ( ; partIt!=partItEnd; ++partIt) {
        m_particleBroker->push( *partIt, &isfp);
      }
    }
  }

  // (a.) batch process mode, ignore the incoming particle for now
  if ( m_batchProcessMcTruth) {
    ATH_MSG_DEBUG("particle is ignored now, will run Calo simulation using ID McTruth at the end of the event");
    return StatusCode::SUCCESS;
  }
  // (b.) throw away particles with undefined Barcode if m_simulateUndefinedBCs==False
  else if ( (!m_simulateUndefinedBCs) && (bc == Barcode::fUndefinedBarcode)) {
    ATH_MSG_DEBUG("particle has undefined barcode, will not simulate it");
    return StatusCode::SUCCESS;
  }
  // (c.) individual particle processing
  ATH_MSG_DEBUG("particle is simulated individually");
  return processOneParticle( isfp);
}


StatusCode ISF::NativeFastCaloSimSvc::processOneParticle( const ISF::ISFParticle& isfp) {
  ATH_MSG_VERBOSE ( m_screenOutputPrefix << "Simulating pdgid = "<< isfp.pdgCode());

  StatusCode sc(StatusCode::SUCCESS);
  //ZH commented out these to avoid warnings at the moment
  //ToolHandleArray<ICaloCellMakerTool>::iterator itrTool=m_caloCellMakerTools_simulate.begin();
  //ToolHandleArray<ICaloCellMakerTool>::iterator endTool=m_caloCellMakerTools_simulate.end();

  /*
  // loop on tools
  for (;itrTool!=endTool;++itrTool) {
    FastShowerCellBuilderTool* fcs=dynamic_cast< FastShowerCellBuilderTool* >(&(*(*itrTool)));
    if(!fcs) {
      ATH_MSG_WARNING( m_screenOutputPrefix << "tool " << itrTool->name()<< "is not a FastShowerCellBuilderTool" );   
      continue;
    }
    
    ATH_MSG_VERBOSE( m_screenOutputPrefix << "Calling tool " << itrTool->name() );   
    std::string chronoName=this->name()+"_"+ itrTool->name();
    
    if (m_chrono) m_chrono->chronoStart( chronoName);

    HepMC::FourVector momentum(isfp.momentum().x(),isfp.momentum().y(),isfp.momentum().z(), sqrt( isfp.mass()*isfp.mass() + isfp.momentum().mag2()) );
    HepMC::GenParticle* part=new HepMC::GenParticle(momentum,isfp.pdgCode());
    HepMC::FourVector position(isfp.position().x(),isfp.position().y(),isfp.position().z(),isfp.timeStamp());
    HepMC::GenVertex vertex(position);
    vertex.add_particle_out( part ); // HepMC::GenVertex destructor will delete the particle

    //sc = (*itrTool)->process(m_theContainer);
    if(fcs->process_particle(m_theContainer,part,0).isFailure()) {
      ATH_MSG_WARNING( m_screenOutputPrefix << "simulation of particle pdgid=" << isfp.pdgCode()<< " failed" );   
      sc = StatusCode::FAILURE;
    }

    if (m_chrono) m_chrono->chronoStop( chronoName );
    //ATH_MSG_DEBUG( m_screenOutputPrefix << "Chrono stop : delta " << m_chrono->chronoDelta (chronoName,IChronoStatSvc::USER ) * CLHEP::microsecond / CLHEP::second << " second " );
  } //end of for-loop
  */

  //  ATH_MSG_VERBOSE ( m_screenOutputPrefix << "kill the particle in the end");
  return sc;
}

