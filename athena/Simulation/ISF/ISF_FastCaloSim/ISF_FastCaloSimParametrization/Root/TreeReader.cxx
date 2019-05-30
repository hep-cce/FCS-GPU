/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TTreeFormula.h"
#include "TTreeFormulaManager.h"
#include "TString.h"
#include <iostream>
#include <TChain.h>
#include <vector>
#include "ISF_FastCaloSimParametrization/TreeReader.h"

//////////////////////////////////////////////////
//
//		Class TreeReader
//		TreeReader.cpp
//
//////////////////////////////////////////////////

//______________________________________________________________________________
/*
Class for Tree reading through TFormula.
______________________________________________________________________________*/


TreeReader::TreeReader()
{
  // Default constructor.
  m_isChain = false;
  m_currentTree = -1;
  m_tree = 0;
  m_currentEntry = -1;
  m_entries = -1;
}

TreeReader::~TreeReader()
{
  m_formulae.clear();
}


//============================================================
TreeReader::TreeReader(TTree* n)
  //============================================================
{
  // Constructor.
  m_tree = 0;
  m_entries = -1;
  SetTree(n);
}

//============================================================
void TreeReader::SetTree(TTree* n)
  //============================================================
{
  // check for null pointer BEFORE trying to use it
  if(!n) return;
  // Set tree.
  m_tree = n;
  m_currentEntry = -1;
  m_formulae.clear();
  m_formulae["__DUMMY__"] = new TTreeFormula("__DUMMY__","0",m_tree);  
  m_isChain = (n->IsA() == TClass::GetClass("TChain"));
  m_currentTree = 0;
  m_entries = (int) m_tree->GetEntries();
}

//=============================================================
double TreeReader::GetVariable(const char* c, int entry)
  //============================================================
{
  // Get vaviable.
  // Return variable for a given entry (<0 -> current entry).
if(entry>=0 && entry!=m_currentEntry) this->GetEntry(entry);
std::string s = c;
TTreeFormula *f = m_formulae[s]; 
if(!f)
  {
  f = new TTreeFormula(c,c,m_tree);
  f->SetQuickLoad(kTRUE);
//   fManager->Add(f);
//   fManager->Sync();  
  if(f->GetNdim()!=1)  //invalid fomula
    {
    delete f;
    f = m_formulae["__DUMMY__"];
    std::cout << "in [TreeReader] : " << s << " is not valid -> return 0" << std::endl;
    }
//  else {f->Notify();}
  m_formulae[s] = f;     
  }
if(f == m_formulae["__DUMMY__"]) return 0; 
int valid = f->GetNdata()  ;
if(!valid) return 0; 
// std::cout << "Evaluating formula : " << s << std::flush;
// std::cout << "  " << f->EvalInstance(0) << std::endl;
return f->EvalInstance(0);
}


//============================================================
int TreeReader::GetEntry(int entry)
  //============================================================
{
  // Read a given entry in the buffer (-1 -> next entry).
  // Return kFALSE if not found.
//   entry += 1;
  if(m_entries==0) return 0;
  if(entry==-1) entry = m_currentEntry+1;
  if(entry<m_entries)
    {
    int entryNumber = m_tree->GetEntryNumber(entry);
    if (entryNumber < 0) return 0;
    Long64_t localEntry = m_tree->LoadTree(entryNumber);   
    if (localEntry < 0) return 0;    
    m_currentEntry = entry;
      if(m_isChain) // check file change in chain
        {
        int I = static_cast<TChain*>(m_tree)->GetTreeNumber();   
        if(I!=m_currentTree) 
          {
          m_currentTree = I;
          //fManager->Clear();
          std::map<std::string, TTreeFormula*>::iterator itr = m_formulae.begin();
          std::map<std::string, TTreeFormula*>::iterator itrE= m_formulae.end();  
          TTreeFormula* dummy = m_formulae["__DUMMY__"];     
          for(;itr!=itrE;itr++) 
            { 
            if(itr->second!=dummy) itr->second->Notify(); //itr->second->UpdateFormulaLeaves();
   	        }
          }	
        }     
      return 1;
    }
  return 0;
}



//ClassImp(TreeReader) // Integrate this class into ROOT

