
#include "TMatrixD.h"
#include "TVectorD.h"
#include "TH1D.h"
#include "TH1I.h"
#include "TFile.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TApplication.h"
#include "TTree.h"
#include "TSystem.h"
#include "TH2D.h"
#include "TPrincipal.h"
#include "TMath.h"
#include "TBrowser.h"
#include "TFCSApplyFirstPCA.h"
#include "TFCSMakeFirstPCA.h"
#include "TreeReader.h"
#include "TLorentzVector.h"
#include "TChain.h"
#include "ISF_FastCaloSimEvent/TFCSSimulationState.h"

#include <iostream>
using namespace std;

TFCSApplyFirstPCA::TFCSApplyFirstPCA(string MakeFirstPCA_rootfilename) {
  // default parameters:
  m_nbins1 = 5;
  m_nbins2 = 1;
  m_dorescale = 1;
  m_infilename = MakeFirstPCA_rootfilename;
}

void TFCSApplyFirstPCA::init() {

  cout << "TFCSApplyFirstPCA::init" << endl;

  TFile* inputfile = TFile::Open(m_infilename.c_str());
  if (inputfile->IsZombie()) {
    cout << "Error: problem with loading PCA file" << endl;
    exit(-1);
  }
  m_principal = (TPrincipal*)inputfile->Get("principal");
  TTree* T_Gauss = (TTree*)inputfile->Get("T_Gauss");

  TH1I* h_layer_input = (TH1I*)inputfile->Get("h_layer_input");
  for (int b = 1; b <= h_layer_input->GetNbinsX(); b++) {
    if (h_layer_input->GetBinContent(b)) {
      m_layer_number.push_back(b - 1);
      m_layer_totE_name.push_back(Form("layer%i", b - 1));
    }
  }

  m_layer_totE_name.push_back("totalE");

  cout << "check layer" << endl;
  for (unsigned int l = 0; l < m_layer_number.size(); l++)
    cout << "l " << l << " number " << m_layer_number[l] << endl;

  // binning:

  TreeReader* tree_Gauss = new TreeReader();
  tree_Gauss->SetTree(T_Gauss);

  TH1D* hPCA_first_component =
      new TH1D("hPCA_first_component", "hPCA_first_component", 100000, -20, 20);

  double* data_PCA = new double[m_layer_totE_name.size()];
  double* input_data = new double[m_layer_totE_name.size()];
  for (int event = 0; event < tree_Gauss->GetEntries(); event++) {
    tree_Gauss->GetEntry(event);
    for (unsigned int l = 0; l < m_layer_totE_name.size(); l++)
      input_data[l] = tree_Gauss->GetVariable(
          Form("data_Gauss_%s", m_layer_totE_name[l].c_str()));

    m_principal->X2P(input_data, data_PCA);
    hPCA_first_component->Fill(data_PCA[0]);
  }

  double* xq = new double[m_nbins1];
  double* yq = new double[m_nbins1];

  quantiles(hPCA_first_component, m_nbins1, xq, yq);

  for (int i = 0; i < m_nbins1; i++) m_yq.push_back(yq[i]);

  std::vector<TH1D*> h_compo1;
  for (int m = 0; m < m_nbins1; m++)
    h_compo1.push_back(new TH1D(Form("h_compo1_bin%i", m),
                                Form("h_compo1_bin%i", m), 20000, -20, 20));

  for (int event = 0; event < tree_Gauss->GetEntries(); event++) {
    tree_Gauss->GetEntry(event);

    for (unsigned int l = 0; l < m_layer_totE_name.size(); l++)
      input_data[l] = tree_Gauss->GetVariable(
          Form("data_Gauss_%s", m_layer_totE_name[l].c_str()));

    m_principal->X2P(input_data, data_PCA);

    int firstbin = -42;
    // Binning 1st PC
    for (int m = 0; m < m_nbins1; m++) {
      if (m == 0 && data_PCA[0] <= yq[0]) firstbin = 0;
      if (m > 0 && data_PCA[0] > yq[m - 1] && data_PCA[0] <= yq[m])
        firstbin = m;
    }

    if (firstbin >= 0) h_compo1[firstbin]->Fill(data_PCA[1]);
  }

  std::vector<std::vector<double> > yq2d(m_nbins1,
                                         std::vector<double>(m_nbins2));

  for (int m = 0; m < m_nbins1; m++) {
    cout << "m " << m << " m_nbins1 " << m_nbins1 << endl;

    double* xq2 = new double[m_nbins2];
    double* yq2 = new double[m_nbins2];

    quantiles(h_compo1[m], m_nbins2, xq2, yq2);

    for (int u = 0; u < m_nbins2; u++) yq2d[m][u] = yq2[u];

    delete[] xq2;
    delete[] yq2;
  }

  for (int i = 0; i < m_nbins1; i++) {
    vector<double> this_yq;
    for (int j = 0; j < m_nbins2; j++) this_yq.push_back(yq2d[i][j]);
    m_yq2d.push_back(this_yq);
  }

  // cleanup
  delete hPCA_first_component;
  for (auto it = h_compo1.begin(); it != h_compo1.end(); ++it) delete *it;
  h_compo1.clear();
  delete tree_Gauss;
  delete T_Gauss;

  delete[] xq;
  delete[] yq;
}

int TFCSApplyFirstPCA::get_PCAbin_from_simstate(TFCSSimulationState& simstate) {

  int firstPCAbin = -1;

  // check for all-zero event:
  if (fabs(simstate.E()) < 0.0001) return 0;

  vector<double> PCA_transformed_data = get_PCAdata_from_simstate(simstate);

  // Apply Binning to 1st and 2nd PC
  int Bin_1stPC1 = 0;
  int Bin_1stPC2 = 0;

  for (int m = 0; m < m_nbins1; m++) {
    if (m == 0 && PCA_transformed_data[0] <= m_yq[0]) {
      Bin_1stPC1 = 0;
      for (int u = 0; u < m_nbins2; u++) {
        if (u == 0 && PCA_transformed_data[1] <= m_yq2d[0][0]) Bin_1stPC2 = 0;
        if (u > 0 && PCA_transformed_data[1] > m_yq2d[0][u - 1] &&
            PCA_transformed_data[1] <= m_yq2d[0][u])
          Bin_1stPC2 = u;
      }
    }
    if (m > 0 && PCA_transformed_data[0] > m_yq[m - 1] &&
        PCA_transformed_data[0] <= m_yq[m]) {
      Bin_1stPC1 = m;
      for (int u = 0; u < m_nbins2; u++) {
        if (u == 0 && PCA_transformed_data[1] <= m_yq2d[m][0]) Bin_1stPC2 = 0;
        if (u > 0 && PCA_transformed_data[1] > m_yq2d[m][u - 1] &&
            PCA_transformed_data[1] <= m_yq2d[m][u])
          Bin_1stPC2 = u;
      }
    }
  }

  firstPCAbin = Bin_1stPC1 + m_nbins1 * Bin_1stPC2 + 1;

  return firstPCAbin;
}

vector<double> TFCSApplyFirstPCA::get_PCAdata_from_simstate(
    TFCSSimulationState& simstate) {

  vector<double> PCA_transformed_data;

  double* input_data = new double[m_layer_totE_name.size()];
  for (unsigned int l = 0; l < m_layer_number.size(); l++)
    input_data[l] = simstate.Efrac(m_layer_number[l]);
  input_data[m_layer_totE_name.size() - 1] = simstate.E();
  double* data_PCA = new double[m_layer_totE_name.size()];
  double* transformed_input_data = new double[m_layer_totE_name.size()];

  for (unsigned int l = 0; l < m_layer_totE_name.size(); l++) {
    // double cumulant =
    // TFCSMakeFirstPCA::get_cumulant_random(input_data[l],m_cumulative_energies[l]);
    double cumulant = TFCSMakeFirstPCA::get_cumulant_random(
        simstate.randomEngine(), input_data[l], m_cumulative_energies[l]);
    double maxErfInvArgRange = 0.99999999;
    double arg = 2.0 * cumulant - 1.0;
    arg = TMath::Min(+maxErfInvArgRange, arg);
    arg = TMath::Max(-maxErfInvArgRange, arg);
    transformed_input_data[l] = TMath::Pi() / 2.0 * TMath::ErfInverse(arg);
  }

  m_principal->X2P(transformed_input_data,
                   data_PCA);  // data_PCA is the output (PCA transformed data)

  for (unsigned int i = 0; i < m_layer_totE_name.size(); i++)
    PCA_transformed_data.push_back(data_PCA[i]);

  delete[] data_PCA;
  delete[] input_data;
  delete[] transformed_input_data;

  return PCA_transformed_data;
}

// void TFCSApplyFirstPCA::run_over_chain(TChain* inputchain, string
// outfilename)
void TFCSApplyFirstPCA::run_over_chain(CLHEP::HepRandomEngine* randEngine,
                                       TChain* inputchain, string outfilename) {

  TreeReader* read_inputTree = new TreeReader();
  read_inputTree->SetTree(inputchain);
  vector<TH1D*> histos_data =
      TFCSMakeFirstPCA::get_G4_histos_from_tree(m_layer_number, read_inputTree);
  vector<TH1D*> cumul_data =
      TFCSMakeFirstPCA::get_cumul_histos(m_layer_totE_name, histos_data);

  set_cumulative_energy_histos(cumul_data);

  double* input_data = new double[m_layer_totE_name.size()];
  double* transformed_input_data = new double[m_layer_totE_name.size()];
  int firstPCAbin;

  cout << "--- Fill a tree that has the energy and bin information" << endl;

  TTree* tree_1stPCA = new TTree(Form("tree_1stPCA"), Form("tree_1stPCA"));
  tree_1stPCA->SetDirectory(0);

  tree_1stPCA->Branch("firstPCAbin", &firstPCAbin, "firstPCAbin/I");
  for (unsigned int l = 0; l < m_layer_totE_name.size(); l++)
    tree_1stPCA->Branch(Form("energy_%s", m_layer_totE_name[l].c_str()),
                        &input_data[l],
                        Form("energy_%s/D", m_layer_totE_name[l].c_str()));

  for (int event = 0; event < read_inputTree->GetEntries(); event++) {
    read_inputTree->GetEntry(event);

    double total_e = read_inputTree->GetVariable("total_cell_energy");

    // if(fabs(total_e)<0.0001) continue;

    vector<double> e_layer;
    for (unsigned int l = 0; l < m_layer_number.size(); l++) {
      e_layer.push_back(read_inputTree->GetVariable(
          Form("cell_energy[%d]", m_layer_number[l])));
      // cout<<"event "<<event<<" l "<<l<<" energy "<<e_layer[l]<<endl;
    }

    /*
    int all_bad=1;
    for(unsigned int l=0;l<m_layer_number.size();l++)
    {
     if(fabs(e_layer[l])>0.0001) all_bad=0;
    }
    if(all_bad) continue;
    */

    // rescale to get rid of negative fractions:
    if (m_dorescale) {
      double e_positive_sum = 0.0;
      for (unsigned int e = 0; e < m_layer_number.size(); e++) {
        if (e_layer[e] > 0)
          e_positive_sum += e_layer[e];
        else
          e_layer[e] = 0.0;
      }

      double rescale = 1.0;
      if (e_positive_sum > 0) rescale = total_e / e_positive_sum;

      for (unsigned int e = 0; e < e_layer.size(); e++) e_layer[e] *= rescale;
    }

    for (unsigned int l = 0; l < m_layer_totE_name.size(); l++) {
      if (l == m_layer_totE_name.size() - 1)
        input_data[l] = total_e;
      else  // layer
      {
        if (fabs(total_e) > 0.0001)
          input_data[l] = e_layer[l] / total_e;
        else
          input_data[l] = 0;
      }

      // cout<<"l "<<l<<" input_data[l] "<<input_data[l]<<" total_e
      // "<<total_e<<endl;
    }

    // TFCSSimulationState simstate;
    TFCSSimulationState simstate(randEngine);

    for (int s = 0; s < CaloCell_ID_FCS::MaxSample; s++) {
      double energyfrac = 0.0;
      for (unsigned int l = 0; l < m_layer_number.size(); l++) {
        if (m_layer_number[l] == s) energyfrac = input_data[l];
      }
      simstate.set_Efrac(s, energyfrac);
      simstate.set_E(s, energyfrac * total_e);
      simstate.set_E(total_e);
    }

    firstPCAbin = get_PCAbin_from_simstate(simstate);

    tree_1stPCA->Fill();

  }  // for events in gauss

  delete[] input_data;
  delete[] transformed_input_data;

  // add a histogram that holds the relevant layer:
  int totalbins = m_nbins1 * m_nbins2;

  TH2I* h_layer = new TH2I("h_layer", "h_layer", totalbins, 0.5,
                           totalbins + 0.5, 25, -0.5, 24.5);
  h_layer->GetXaxis()->SetTitle("PCA bin");
  h_layer->GetYaxis()->SetTitle("Layer");
  for (int b = 0; b < totalbins; b++) {
    for (int l = 0; l < 25; l++) {
      int is_relevant = 0;
      for (unsigned int i = 0; i < m_layer_number.size(); i++) {
        if (l == m_layer_number[i]) is_relevant = 1;
      }
      h_layer->SetBinContent(b + 1, l + 1, is_relevant);
    }
  }

  TFile* output = TFile::Open(outfilename.c_str(), "RECREATE");
  output->Add(h_layer);
  output->Add(tree_1stPCA);
  output->Write();

  cout << "1st PCA is done. Output file: " << outfilename << endl;

  delete read_inputTree;

}  // run

void TFCSApplyFirstPCA::quantiles(TH1D* h, int nq, double* xq, double* yq) {

  // Function for quantiles
  // h Input histo
  // nq number of quantiles
  // xq position where to compute the quantiles in [0,1]
  // yq array to contain the quantiles

  for (int i = 0; i < nq; i++) {
    xq[i] = double(i + 1) / nq;
    h->GetQuantiles(nq, yq, xq);
  }
}

void TFCSApplyFirstPCA::print_binning() {

  cout << "binning of the first component" << endl;
  for (int m = 0; m < m_nbins1; m++) {
    cout << "bin nr " << m << " cut at " << m_yq[m] << endl;
    cout << "   binning of the second component" << endl;
    for (int n = 0; n < m_nbins2; n++) {
      cout << "   bin nr " << n << " cut at " << m_yq2d[m][n] << endl;
    }
  }
}

void TFCSApplyFirstPCA::set_pcabinning(int bin1, int bin2) {
  m_nbins1 = bin1;
  m_nbins2 = bin2;
}

void TFCSApplyFirstPCA::set_cumulative_energy_histos(
    vector<TH1D*> cumul_inputdata) {

  // copy the stuff into the member variable:
  for (unsigned int i = 0; i < cumul_inputdata.size(); i++)
    m_cumulative_energies.push_back(cumul_inputdata[i]);
}
