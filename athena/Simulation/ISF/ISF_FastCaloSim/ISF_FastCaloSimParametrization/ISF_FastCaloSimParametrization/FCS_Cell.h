/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#ifndef FCS_Cell
#define FCS_Cell
#include <vector>
//#include <stdint.h>
#include <Rtypes.h>
#include <TLorentzVector.h>
//#include <iostream>
/******************************************
This contains structure definition
All structures are relatively simple
each matched cell remembers - cell properties + vector of g4hits in this cell + vector of FCS hits in this cell

Technicalities - needs a Linkdef.h file + makefile to create the dictionary for ROOT
then the last class could be saved in to the TTree

 ******************************************/

struct FCS_cell
{
  Long64_t cell_identifier;
  int   sampling;
  float energy;
  float center_x;
  float center_y;
  float center_z; //to be updated later      
  bool operator<(const FCS_cell &rhs) const { return energy > rhs.energy;};                                                                   
};

struct FCS_hit //this is the FCS detailed hit
{
  Long64_t identifier; //hit in the same tile cell can have two identifiers (for two PMTs)
  Long64_t cell_identifier;
  int    sampling; //calorimeter layer
  float  hit_energy; //energy is already scaled for the sampling fraction
  float  hit_time;
  float  hit_x;
  float  hit_y;
  float  hit_z;
  bool operator<(const FCS_hit &rhs) const { return hit_energy > rhs.hit_energy;};
  //float  hit_sampfrac;
};

struct FCS_g4hit //this is the standard G4Hit
{
  Long64_t identifier;
  Long64_t cell_identifier;
  int    sampling;
  float  hit_energy;
  float  hit_time;
  //float  hit_sampfrac;
  bool operator<(const FCS_g4hit &rhs) const { return hit_energy > rhs.hit_energy;};
};

struct FCS_matchedcell //this is the matched structure for a single cell
{
  FCS_cell cell;
  std::vector<FCS_g4hit> g4hit;
  std::vector<FCS_hit> hit;
  inline void clear() {g4hit.clear(); hit.clear();};
  inline float scalingfactor(){float hitsum =0.; for (unsigned int i=0; i<hit.size(); i++){hitsum+=hit[i].hit_energy;}; return cell.energy/hitsum;}; //doesn't check for 0!
  bool operator<(const FCS_matchedcell &rhs) const { return cell.energy > rhs.cell.energy;};
  inline void sorthit() { std::sort(hit.begin(), hit.end());};
  inline void sortg4hit() { std::sort(g4hit.begin(), g4hit.end());};
  inline void sort() { sorthit(); sortg4hit();};
  inline void time_trim(float timing_cut) { /*std::cout <<"Cutting: "<<timing_cut<<" from: "<<hit.size()<<" "<<g4hit.size()<<std::endl;*/hit.erase(std::remove_if(hit.begin(), hit.end(), [&timing_cut](const FCS_hit &rhs) { return rhs.hit_time>timing_cut;}), hit.end()); g4hit.erase(std::remove_if(g4hit.begin(), g4hit.end(), [&timing_cut](const FCS_g4hit &rhs) { return rhs.hit_time>timing_cut;}),g4hit.end());/*std::cout <<"remaining: "<<hit.size()<<" "<<g4hit.size()<<std::endl;*/};
};

struct FCS_matchedcellvector //this is the matched structure for the whole event (or single layer) - vector of FCS_matchedcell 
{
  //Note that struct can have methods
  //Note the overloaded operator(s) to access the underlying vector
  std::vector<FCS_matchedcell> m_vector;
  inline std::vector<FCS_matchedcell> GetLayer(int layer){std::vector<FCS_matchedcell> ret; for (unsigned i=0; i<m_vector.size(); i++) {if (m_vector[i].cell.sampling == layer) ret.push_back(m_vector[i]);}; return ret;};
  inline FCS_matchedcell operator[](unsigned int place) { return m_vector[place];};
  inline unsigned int size() {return m_vector.size();};
  inline void push_back(FCS_matchedcell cell) { m_vector.push_back(cell);};
  inline void sort_cells() { std::sort(m_vector.begin(), m_vector.end());};
  inline void sort() { std::sort(m_vector.begin(), m_vector.end()); for (unsigned int i=0; i<m_vector.size(); i++) { m_vector[i].sort();};};
  inline void time_trim(float timing_cut) 
  { for (unsigned int i=0; i< m_vector.size(); i++) { m_vector[i].time_trim(timing_cut); }; m_vector.erase(std::remove_if(m_vector.begin(), m_vector.end(), [] (const FCS_matchedcell &rhs) { return (rhs.hit.size()==0 && rhs.g4hit.size() ==0 && fabs(rhs.cell.energy)<1e-3);}), m_vector.end());};
  inline float scalingfactor(){float cellsum=0.; float hitsum=0.; for (unsigned int i=0; i<m_vector.size(); i++){cellsum+=m_vector[i].cell.energy;for (unsigned int j=0; j<m_vector[i].hit.size(); j++){hitsum+=m_vector[i].hit[j].hit_energy;};}; return cellsum/hitsum;}; //doesn't check for 0!
};


#endif

