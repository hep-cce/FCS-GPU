/*
  Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
*/

#ifndef ISF_FASTCALOSIMEVENT_TFCSParametrizationBase_h
#define ISF_FASTCALOSIMEVENT_TFCSParametrizationBase_h

#include <TNamed.h>
#include <set>

class ICaloGeometry;
class TFCSSimulationState;
class TFCSTruthState;
class TFCSExtrapolationState;

// Define Athena like message macro's such that they work stand alone and inside athena
#if defined(__FastCaloSimStandAlone__) || defined(__FastCaloSimStandAloneDict__)
  #include <iostream>
  #include <iomanip>
  typedef std::ostream MsgStream;
  #define endmsg std::endl
  
  namespace MSG {
    enum Level {
      NIL = 0,
      VERBOSE,
      DEBUG,
      INFO,
      WARNING,
      ERROR,
      FATAL,
      ALWAYS,
      NUM_LEVELS
    }; // enum Level
    __attribute__ ((unused)) static const char* LevelNames[NUM_LEVELS]={"NIL","VERBOSE","DEBUG","INFO","WARNING","ERROR","FATAL","ALWAYS"};
  }  // end namespace MSG  
  // Needs a check despite the name, as stand alone mode is not using MsgStream, but just cout internally
  #define ATH_MSG_LVL_NOCHK(lvl, x)               \
    do {                                          \
      if(this->msgLvl(lvl)) this->msg(lvl) << std::setw(45) << std::left << this->GetName() << " " << MSG::LevelNames[lvl] << " " << x << endmsg; \
    } while (0)

  #define ATH_MSG_LVL(lvl, x)                     \
    do {                                          \
      if (this->msgLvl(lvl)) ATH_MSG_LVL_NOCHK(lvl, x);                \
    } while (0)

  #define ATH_MSG_VERBOSE(x) ATH_MSG_LVL(MSG::VERBOSE, x)
  #define ATH_MSG_DEBUG(x)   ATH_MSG_LVL(MSG::DEBUG,   x)
  // note that we are using the _NOCHK variant here
  #define ATH_MSG_INFO(x)    ATH_MSG_LVL_NOCHK(MSG::INFO,    x)
  #define ATH_MSG_WARNING(x) ATH_MSG_LVL_NOCHK(MSG::WARNING, x)
  #define ATH_MSG_ERROR(x)   ATH_MSG_LVL_NOCHK(MSG::ERROR,   x)
  #define ATH_MSG_FATAL(x)   ATH_MSG_LVL_NOCHK(MSG::FATAL,   x)

  // can be used like so: ATH_MSG(INFO) << "hello" << endmsg;
  #define ATH_MSG(lvl) \
    if (this->msgLvl(MSG::lvl)) this->msg(MSG::lvl) << std::setw(45) << std::left << this->GetName() << " " << MSG::LevelNames[MSG::lvl] << " " 

#else
  #include "AthenaKernel/MsgStreamMember.h"
  #include "AthenaBaseComps/AthMsgStreamMacros.h"
#endif

/** Base class for all FastCaloSim parametrizations
Functionality in derivde classes is  provided through the simulate method. The simulate method takes a TFCSTruthState and a TFCSExtrapolationState object as input and provides output in a TFCSSimulationState.
Parametrizations contain information on the pdgid, range in Ekin and range in eta of particles 
to which they can be applied.
Several basic types of parametrization exists:
- classes derived from TFCSEnergyParametrization simulate energy information which is written into TFCSSimulationState
- classes derived from TFCSLateralShapeParametrization simulate cell level information for specific calorimeter layers and bins "Ebin" in the energy parametrization
- classes derived from TFCSParametrizationChain call other parametrization. Depending on the derived class, these other parametrization are only called under special conditions
- a special case of TFCSLateralShapeParametrization is TFCSLateralShapeParametrizationHitBase for hit level shape simulation through the simulate_hit method. Hit level simulation is controlled through the special chain TFCSLateralShapeParametrizationHitChain.
*/

///Return codes for the simulate function
enum FCSReturnCode {
  FCSFatal = 0,
  FCSSuccess = 1,
  FCSRetry = 2
};

#define FCS_RETRY_COUNT 3

class TFCSParametrizationBase:public TNamed {
public:
  TFCSParametrizationBase(const char* name=nullptr, const char* title=nullptr);

  ///Status bit for FCS needs
  enum FCSStatusBits {
     kMatchAllPDGID = BIT(14) ///< Set this bit in the TObject bit field if valid for all PDGID
  };

  virtual bool is_match_pdgid(int /*id*/) const {return TestBit(kMatchAllPDGID);};
  virtual bool is_match_Ekin(float /*Ekin*/) const {return false;};
  virtual bool is_match_eta(float /*eta*/) const {return false;};

  virtual bool is_match_Ekin_bin(int /*Ekin_bin*/) const {return false;};
  virtual bool is_match_calosample(int /*calosample*/) const {return false;};

  virtual bool is_match_all_pdgid() const {return TestBit(kMatchAllPDGID);};
  virtual bool is_match_all_Ekin() const {return false;};
  virtual bool is_match_all_eta() const {return false;};
  virtual bool is_match_all_Ekin_bin() const {return false;};
  virtual bool is_match_all_calosample() const {return false;};

  virtual const std::set< int > &pdgid() const {return s_no_pdgid;};
  virtual double Ekin_nominal() const {return init_Ekin_nominal;};
  virtual double Ekin_min() const {return init_Ekin_min;};
  virtual double Ekin_max() const {return init_Ekin_max;};
  virtual double eta_nominal() const {return init_eta_nominal;};
  virtual double eta_min() const {return init_eta_min;};
  virtual double eta_max() const {return init_eta_max;};

  virtual void set_match_all_pdgid() {SetBit(kMatchAllPDGID);};
  virtual void reset_match_all_pdgid() {ResetBit(kMatchAllPDGID);};

  ///Method to set the geometry access pointer. Loops over daughter objects if present
  virtual void set_geometry(ICaloGeometry* geo);
  
  ///Some derived classes have daughter instances of TFCSParametrizationBase objects
  ///The size() and operator[] methods give general access to these daughters
  virtual unsigned int size() const {return 0;};
  
  ///Some derived classes have daughter instances of TFCSParametrizationBase objects
  ///The size() and operator[] methods give general access to these daughters
  virtual const TFCSParametrizationBase* operator[](unsigned int /*ind*/) const {return nullptr;};
  
  ///Some derived classes have daughter instances of TFCSParametrizationBase objects
  ///The size() and operator[] methods give general access to these daughters
  virtual TFCSParametrizationBase* operator[](unsigned int /*ind*/) {return nullptr;};

  ///Method in all derived classes to do some simulation
  virtual FCSReturnCode simulate(TFCSSimulationState& simulstate,const TFCSTruthState* truth, const TFCSExtrapolationState* extrapol);

  ///Print object information. 
  void Print(Option_t *option = "") const;
  
  ///Deletes all objects from the s_cleanup_list. 
  ///This list can get filled during streaming operations, where an immediate delete is not possible
  static void DoCleanup();

protected:
  const double init_Ekin_nominal=0;
  const double init_Ekin_min=0;
  const double init_Ekin_max=14000000;
  const double init_eta_nominal=0;
  const double init_eta_min=-100;
  const double init_eta_max=100;

  static std::vector< TFCSParametrizationBase* > s_cleanup_list;

#if defined(__FastCaloSimStandAlone__) || defined(__FastCaloSimStandAloneDict__)
public:
  /// Update outputlevel
  virtual void setLevel(int level,bool recursive=false) {
    level = (level >= MSG::NUM_LEVELS) ?
      MSG::ALWAYS : (level<MSG::NIL) ? MSG::NIL : level;
    m_level = MSG::Level(level);
    if(recursive) for(unsigned int i=0;i<size();++i) (*this)[i]->setLevel(m_level,recursive);
  }
  /// Retrieve output level
  MSG::Level level() const {return m_level;}

  /// Log a message using cout; a check of MSG::Level lvl is not possible!
  MsgStream& msg() const {return *m_msg;}
  MsgStream& msg( const MSG::Level ) const {return *m_msg;}  
  /// Check whether the logging system is active at the provided verbosity level
  bool msgLvl( const MSG::Level lvl ) const {return m_level<=lvl;}
private:
  MSG::Level m_level;//! Do not persistify!
  
  MsgStream* m_msg;//! Do not persistify!
#else
public:
  /// Update outputlevel
  void setLevel(int level) {s_msg->get().setLevel(level);}

  /// Retrieve output level
  MSG::Level level() const {return s_msg->get().level();}

  /// Log a message using the Athena controlled logging system
  MsgStream& msg() const { return s_msg->get(); }

  /// Log a message using the Athena controlled logging system
  MsgStream& msg( MSG::Level lvl ) const { return *s_msg << lvl; }

  /// Check whether the logging system is active at the provided verbosity level
  bool msgLvl( MSG::Level lvl ) const { return s_msg->get().level() <= lvl; }
  
private:
  /// Static private message stream member. We don't want this to take memory for every instance of this object created
  static Athena::MsgStreamMember* s_msg;//! Do not persistify!
#endif  
  
private:
  static std::set< int > s_no_pdgid;

  ClassDef(TFCSParametrizationBase,1)  //TFCSParametrizationBase
};

#if defined(__ROOTCLING__) && defined(__FastCaloSimStandAlone__)
#pragma link C++ class TFCSParametrizationBase+;
#endif

#endif
