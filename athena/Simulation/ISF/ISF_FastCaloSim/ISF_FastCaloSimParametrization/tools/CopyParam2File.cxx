/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

/**
  * Prerequisites:
  *
  * -) 
  *
  * $ root -l -b -q 'CopyParam2File.cxx("shape.pdgid_211.en_50000.eta_020.root", "EnergyParam", 211, 50000, 0.20)'
  * 
 */
#include "TROOT.h"
#include "TRint.h"
#include "TKey.h"
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"
#include <string>

// build string sparam.<pdgid>.<energy>.<eta>.<pca#>.<layer#>
std::string doSParam(std::string pstr, std::string kstr) {
  std::string delim1 = "layer";
  std::string delim2 = "_";
  std::string delim3 = "pca";
  size_t l_startpos = kstr.find(delim1)+delim1.length();
  size_t l_endpos = kstr.find_last_of(delim2)-(kstr.find(delim1)+delim1.length());
  size_t p_startpos = kstr.find(delim3)+delim3.length();
  std::string lNum = kstr.substr(l_startpos, l_endpos);
  std::string pNum = kstr.substr(p_startpos,std::string::npos);
  return pstr.append(".pca_"+pNum+".layer_"+lNum);
}

// build string eparam.<pdgid>.<energy>.<eta>.<pca#>.<
std::string doEParam(std::string pstr, std::string kstr, std::string dirstr) {
  std::string lNum;
  std::string retstr = "";
  // bin/pca common amongst all
  std::string delim1 = "bin";
  std::string delim2 = "/";
  size_t p_startpos = dirstr.find(delim1)+delim1.length();
  size_t p_endpos = dirstr.find_last_of(delim2)-(dirstr.find(delim1)+delim1.length());
  std::string pNum = dirstr.substr(p_startpos, p_endpos);
  pstr.append(".pca_"+pNum);
  // check if we are in a pca dir
  if (dirstr.find("pca") != std::string::npos) {
    retstr = pstr+"."+kstr;
  }
  // ...in a layer dir?
  else if (dirstr.find("layer") != std::string::npos) {
    size_t l_startpos = dirstr.find("layer")+std::string("layer").length();
    retstr = pstr+".layer_"+dirstr.substr(l_startpos, std::string::npos);
  }
  // ...must be a `totalE`!
  else if (dirstr.find("totalE") != std::string::npos) {
    retstr = pstr+".totalE";
  }
  // Nope -- we don't know where we are!
  else {
    std::cout << "ERROR: where are we? (Not a `layer` or `pca`, that's for sure!)" << std::endl;
    exit(1);
  }

  std::cout << "\tretstr: " << retstr << std::endl;
  return retstr;
}

void copyDir(TDirectory* src, std::string param_str) {
  // Copy src dir to dest
  src->ls();
  TDirectory* savdir = gDirectory;
  TDirectory* adir = savdir;
  adir->cd();
  // loop on all entries in src
  TKey* key;
  TIter nextkey(src->GetListOfKeys());
  while ((key = (TKey*) nextkey())) {
    const char *classname = key->GetClassName();
    TClass *cl = gROOT->GetClass(classname);
    if (!cl) continue;
    if (cl->InheritsFrom("TDirectory")) {
//      adir = savdir->mkdir(key->GetName());
      src->cd(key->GetName());
      TDirectory *subdir = gDirectory;
      adir->cd();
      copyDir(subdir, param_str);
      adir->cd();
    } else if (cl->InheritsFrom("TTree")) {
      TTree *T = (TTree*)src->Get(key->GetName());
      adir->cd();
      TTree *newT = (TTree*)T->CloneTree();
      newT->Write();
    } else {
      src->cd();
      TObject *obj = key->ReadObj();
      adir->cd();
      std::string complete_param_str;
      if (param_str.substr(0,1) == "s") complete_param_str = doSParam(param_str, std::string(key->GetName()));
      else if (param_str.substr(0,1) == "e") complete_param_str = doEParam(param_str, std::string(key->GetName()), std::string(src->GetPath()));
      obj->Write(complete_param_str.c_str(),TObject::kWriteDelete);
      delete obj;
    }
  }
  adir->SaveSelf(kTRUE);
  savdir->cd();
}

void copyFile(const char* fname, std::string param_str) {
  TDirectory* target = gDirectory;
  TFile* fsrc = TFile::Open(fname);
  if (!fsrc || fsrc->IsZombie()) {
    printf("Cannot open source file: %s", fname);
    target->cd();
    return;
  }
  // Copy dir from src to dest
  target->cd("/");
  copyDir(fsrc, param_str);
  delete fsrc;
  target->cd();
}

void 
CopyParam2File(std::string src_str, std::string param_type, unsigned int pdgid, unsigned int energy, float eta, std::string dest_str = "FCSParams.root") {
  // ROOT-aware
  TLorentzVector* t;
  gROOT->LoadMacro("../20.20.X-VAL/Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/IntArray.cxx+");
  gROOT->LoadMacro("../20.20.X-VAL/Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCS1DFunction.cxx+");
  gROOT->LoadMacro("../20.20.X-VAL/Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCS1DFunctionRegression.cxx+");
  gROOT->LoadMacro("../20.20.X-VAL/Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCS1DFunctionRegressionTF.cxx+");
  gROOT->LoadMacro("../20.20.X-VAL/Simulation/ISF/ISF_FastCaloSim/ISF_FastCaloSimEvent/src/TFCS1DFunctionHistogram.cxx+");

  // Destination (where to write to)
  TFile* fdest = TFile::Open(dest_str.c_str(), "UPDATE");
  if (!fdest || fdest->IsZombie()) {
    printf("Cannot open destination file: %s (it doesn't exist?)", dest_str.c_str());
    printf("Creating file...");
    fdest = TFile::Open(dest_str.c_str(), "CREATE");
    // Check if the new file creation was a success; if not, exit with an error.
    if (!fdest || fdest->IsZombie()) {
      printf("Cannot create destination file: %s (something went wrong)", dest_str.c_str());
      return;
    }
  }

  std::string s_param_prefix;
  char param_prefix[100];

  if (param_type == "EnergyParam") {
    s_param_prefix = "eparam";
  }
  else if (param_type == "ShapeParam") {
    s_param_prefix = "sparam";
  }
  else {
    std::cout << "Invalid param_type. Valid param_type arguments are EnergyParam and ShapeParam." << std::endl;
    exit(1);
  }

  // Get rid of the '.' in `eta`
  std::ostringstream ss_eta;
  ss_eta << fixed << setprecision(2) << eta;
  std::string s_eta(ss_eta.str());
  s_eta.erase(1,1);

  // Construct complete param prefix
  s_param_prefix.append(".pdgid_"+std::to_string(pdgid)+".en_"+std::to_string(energy)+".eta_"+s_eta);

  // Copy src
  copyFile(src_str.c_str(), s_param_prefix);

  // cleanup
  fdest->Close();
  delete fdest;
  return;
}

