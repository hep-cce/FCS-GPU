#include <sstream>

#include "FastCaloSimAnalyzer/TFCSVertexZPositionStudies.h"

#include "TKey.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLatex.h"
#include "TStyle.h"

void printATLASlabel(float size, float x, float y,
                     string s = "Simulation Internal");
void WriteInfo(string info, float size, float x, float y, int color = 1);

void removeNegatives(TH1* hist) {
  for (int i = 1; i < hist->GetNbinsX() + 1; ++i) {
    if (hist->GetBinContent(i) < 0.001) hist->SetBinContent(i, 0.001);
  }
}

TFCSVertexZPositionStudies::TFCSVertexZPositionStudies() {

  m_histos.clear();
  m_hist_nominal = nullptr;
}

TFCSVertexZPositionStudies::~TFCSVertexZPositionStudies() {
  for (TH1F* h : m_histos) {
    if (h) delete h;
  }
}

void TFCSVertexZPositionStudies::loadFiles(string dirname,
                                           string filename_nominal,
                                           vector<string>& filenames_shifted) {

  m_dirname = dirname;
  m_file_nominal = TFile::Open((m_dirname + "/" + filename_nominal).c_str());
  if (m_file_nominal->IsZombie()) {
    cout << "Error: File " << m_dirname + "/" + filename_nominal
         << " is zombie!" << endl;
  }

  this->setParticleInfo(filename_nominal);

  m_nshifted = filenames_shifted.size();
  m_files_shifted.resize(m_nshifted);
  m_legendNames.resize(m_nshifted);
  m_vertexZPositions.resize(m_nshifted);

  for (int i = 0; i < m_nshifted; i++) {
    string full_filename = m_dirname + "/" + filenames_shifted[i];
    cout << full_filename << endl;
    m_files_shifted[i] = TFile::Open(full_filename.c_str());
    if (m_files_shifted[i]->IsZombie()) {
      cout << "Error: File " << full_filename << " is zombie!" << endl;
    }

    string shortname = filenames_shifted[i];
    std::size_t found = shortname.find("zv_");
    shortname.erase(0, found);
    found = shortname.find(".");
    shortname.erase(found);

    m_legendNames[i] = shortname;

    shortname.erase(0, 3);
    cout << shortname << endl;
    m_vertexZPositions[i] = shortname;
  }

  // m_files_shifted;
}

void TFCSVertexZPositionStudies::setParticleInfo(
    string filename) {  // Function will attempt to determine particle info from
                        // filename

  cout << filename << endl;

  vector<string> substrings;

  std::string substring;
  std::istringstream keystream(filename);
  while (std::getline(keystream, substring, '_')) {
    substrings.push_back(substring);
  }

  string pid = substrings[2].substr(3);
  string particleType;

  if (pid == "22")
    particleType = "Photons";
  else if (pid == "11")
    particleType = "Electrons";
  else if (pid == "211")
    particleType = "Pions";
  else
    particleType = "";

  double energy = std::stod(substrings[3].substr(1)) / 1000;

  string eta("");

  if (std::find(substrings.begin(), substrings.end(), "disj") !=
      substrings.end()) {
    cout << substrings[8] << " " << substrings[9] << endl;
    double eta_low = std::stod(substrings[8]) / 100;
    double eta_up = std::stod(substrings[9]) / 100;

    if (eta_low == 0.)
      eta = Form("%1.2f #leq #eta #leq %1.2f", -eta_up, eta_up);
    else
      eta = Form("%1.2f #leq #left|#eta#right| #leq %1.2f", eta_low, eta_up);

  } else {
    double eta_low = std::stod(substrings[5]) / 100;
    double eta_up = std::stod(substrings[6]) / 100;
    if (eta_low == 0.)
      eta = Form("%1.0f #leq #eta #leq %1.2f", eta_low, eta_up);
    else
      eta = Form("%1.2f #leq #eta #leq %1.2f", eta_low, eta_up);
  }

  m_particle_energy = Form("%.0f GeV", energy);
  m_particle_eta = eta;
  m_particle_type = particleType;

  // cout << pid << " " << m_particle_info << endl;
}

void TFCSVertexZPositionStudies::initializeLayersAndPCAs() {

  TIter nextkey(m_file_nominal->GetListOfKeys());
  TKey* key = 0;
  while ((key = (TKey*)nextkey())) {

    string keyname = key->GetName();

    vector<string> substrings;

    std::string substring;
    std::istringstream keystream(keyname);
    while (std::getline(keystream, substring, '_')) {
      substrings.push_back(substring);
    }

    substrings[0].erase(0, 2);
    substrings[1].erase(0, 3);

    int layer = atoi(substrings[0].c_str());
    int pca = atoi(substrings[1].c_str());

    if (find(m_layers.begin(), m_layers.end(), layer) == m_layers.end())
      m_layers.push_back(layer);
    if (find(m_pcas.begin(), m_pcas.end(), pca) == m_pcas.end())
      m_pcas.push_back(pca);
  }
}

void TFCSVertexZPositionStudies::loadHistogramsInLayerAndPCA(int layer, int pca,
                                                             string histname) {

  m_name = Form("cs%i_pca%i_", layer, pca);
  m_name += histname;
  m_currentPCA = pca;
  m_currentLayer = layer;

  cout << m_name << endl;

  // m_histos.resize(m_nshifted);
  // m_ratios.resize(m_nshifted);

  m_hist_nominal = (TH1F*)m_file_nominal->Get(m_name.c_str());
  removeNegatives(m_hist_nominal);

  // cout << m_hist_nominal->GetName() << endl;

  m_histos.resize(m_nshifted);
  m_ratios.resize(m_nshifted);

  for (int i = 0; i < m_nshifted; i++) {

    m_histos[i] = (TH1F*)m_files_shifted[i]->Get(m_name.c_str());
    removeNegatives(m_histos[i]);
    // cout <<  m_histos[i]->GetName() << endl;

    m_ratios[i] = (TH1F*)m_histos[i]->Clone();
    m_ratios[i]->SetName((TString)m_histos[i]->GetName() + "_ratio");
    m_ratios[i]->Divide(m_hist_nominal);
  }

  m_info = Form("Layer %i, pca %i", m_currentLayer, m_currentPCA);
}

void TFCSVertexZPositionStudies::loadHistogramsAllPCAs(int layer,
                                                       string histname) {

  m_currentPCA = 0;
  m_currentLayer = layer;

  int pca = m_pcas[0];

  m_name = Form("cs%i_pca%i_", layer, pca);
  m_name += histname;

  m_hist_nominal = (TH1F*)m_file_nominal->Get(m_name.c_str());
  m_histos.resize(m_nshifted);
  m_ratios.resize(m_nshifted);

  for (int i = 0; i < m_nshifted; i++) {
    m_histos[i] = (TH1F*)m_files_shifted[i]->Get(m_name.c_str());
  }

  TH1F* h_temp = nullptr;
  for (int pca : m_pcas) {
    m_name = Form("cs%i_pca%i_", layer, pca);
    m_name += histname;

    h_temp = (TH1F*)m_file_nominal->Get(m_name.c_str());
    m_hist_nominal->Add(h_temp);
    delete h_temp;

    for (int i = 0; i < m_nshifted; i++) {
      h_temp = (TH1F*)m_files_shifted[i]->Get(m_name.c_str());
      m_histos[i]->Add(h_temp);
      delete h_temp;
    }
  }

  removeNegatives(m_hist_nominal);

  for (int i = 0; i < m_nshifted; i++) {
    removeNegatives(m_histos[i]);

    m_ratios[i] = (TH1F*)m_histos[i]->Clone();
    m_ratios[i]->SetName((TString)m_histos[i]->GetName() + "_ratio");
    m_ratios[i]->Divide(m_hist_nominal);
  }

  m_name = Form("cs%i_pca%i_", layer, m_currentPCA);
  m_name += histname;

  m_info = Form("Layer %i, All PCAs", m_currentLayer);
}

void TFCSVertexZPositionStudies::loadMeanEnergyHistogram(string histname) {
  // const int npcas=m_pcas.size();
  // vector<vector<TH1F*> > histos(m_nshifted);
  // for(int i=0;i<m_nshifted;i++)histos[i].resize(npcas);

  // for(int ishifted=0;ishifted<m_nshifted;ishifted++){
  // for(int ipca=0;ipca<npcas;ipca++){
  // int pca=m_pcas[ipca];
  // m_name=Form("cs%i_pca%i_",layer,pca);
  // m_name+=histname;
  // m_histos[ishifted][ipca] = (TH1F*)m_files_shifted[i]->Get(m_name.c_str());

  //}

  //}

  m_name = histname;

  const int npcas = m_pcas.size();
  const int nlayers = m_layers.size();
  // const int nlayers = 1;

  m_hist_nominal = new TH1F((m_name + "_nominal").c_str(), "", npcas, 0, npcas);
  m_hist_nominal->Sumw2();
  for (int i = 0; i < npcas; i++)
    m_hist_nominal->GetXaxis()->SetBinLabel(i + 1, Form("PCA %i", m_pcas[i]));

  m_histos.resize(m_nshifted);
  m_ratios.resize(m_nshifted);

  for (int i = 0; i < m_nshifted; i++) {
    m_histos[i] = (TH1F*)m_hist_nominal->Clone();
    m_ratios[i] = (TH1F*)m_hist_nominal->Clone();
  }
  TH1F* h_temp = nullptr;
  for (int ipca = 0; ipca < npcas; ipca++) {
    int pca = m_pcas[ipca];
    for (int ilayer = 0; ilayer < nlayers; ilayer++) {
      int layer = m_layers[ilayer];
      string name = Form("cs%i_pca%i_", layer, pca) + histname;
      h_temp = (TH1F*)m_file_nominal->Get(name.c_str());
      m_hist_nominal->AddBinContent(ipca + 1, h_temp->Integral());
      m_hist_nominal->SetBinError(
          ipca + 1, sqrt(pow(m_hist_nominal->GetBinError(ipca), 2) +
                         h_temp->GetStdDev()));
      delete h_temp;
      for (int ishifted = 0; ishifted < m_nshifted; ishifted++) {
        h_temp = (TH1F*)m_files_shifted[ishifted]->Get(name.c_str());
        m_histos[ishifted]->AddBinContent(ipca + 1, h_temp->Integral());
        m_histos[ishifted]->SetBinError(
            ipca + 1, sqrt(pow(m_histos[ishifted]->GetBinError(ipca), 2) +
                           h_temp->GetStdDev()));
        delete h_temp;
      }
    }
  }
  for (int i = 0; i < m_nshifted; i++) {
    m_ratios[i] = (TH1F*)m_histos[i]->Clone();
    m_ratios[i]->SetName((TString)m_histos[i]->GetName() + "_ratio");
    m_ratios[i]->Divide(m_hist_nominal);
  }

  m_name = histname + "_AllLayers";
}

void TFCSVertexZPositionStudies::loadMeanEnergyHistogram(int pca,
                                                         string histname) {
  m_name = Form("pca%i_", pca);
  m_name += histname;

  m_currentPCA = pca;

  const int nlayers = m_layers.size();
  m_hist_nominal =
      new TH1F((m_name + "_nominal").c_str(), "", nlayers, 0, nlayers);
  m_hist_nominal->Sumw2();
  m_histos.resize(m_nshifted);
  m_ratios.resize(m_nshifted);

  for (int i = 0; i < nlayers; i++) {
    m_hist_nominal->GetXaxis()->SetBinLabel(i + 1,
                                            Form("Layer %i", m_layers[i]));
  }
  for (int i = 0; i < m_nshifted; i++) {
    m_histos[i] = (TH1F*)m_hist_nominal->Clone();
    m_ratios[i] = (TH1F*)m_hist_nominal->Clone();
  }

  TH1F* h_temp = nullptr;
  for (int i = 0; i < nlayers; i++) {
    int layer = m_layers[i];
    string name = Form("cs%i_", layer) + m_name;
    h_temp = (TH1F*)m_file_nominal->Get(name.c_str());
    m_hist_nominal->SetBinContent(i + 1, h_temp->GetBinContent(1));
    m_hist_nominal->SetBinError(i + 1, h_temp->GetBinError(1));
    delete h_temp;
    for (int ishifted = 0; ishifted < m_nshifted; ishifted++) {
      h_temp = (TH1F*)m_files_shifted[ishifted]->Get(name.c_str());
      m_histos[ishifted]->SetBinContent(i + 1, h_temp->GetBinContent(1));
      m_histos[ishifted]->SetBinError(i + 1, h_temp->GetBinError(1));
      delete h_temp;
    }
  }

  for (int i = 0; i < m_nshifted; i++) {
    m_ratios[i] = (TH1F*)m_histos[i]->Clone();
    m_ratios[i]->SetName((TString)m_histos[i]->GetName() + "_ratio");
    m_ratios[i]->Divide(m_hist_nominal);
  }

  m_info = Form("pca %i", m_currentPCA);
}

void TFCSVertexZPositionStudies::loadHistogramsForFixedZVAndLayer(
    int zv_index, int layer, string histname) {
  m_name = Form("zv_%s_cs%i_", m_vertexZPositions[zv_index].c_str(), layer);
  m_name += histname;

  const unsigned int npcas = m_pcas.size();
  m_histos.resize(npcas);
  m_ratios.resize(npcas);
  m_legendNames.resize(npcas);

  for (unsigned int i = 0; i < npcas; i++) {
    m_legendNames[i] = Form("pca %i", m_pcas[i]);
    string name = Form("cs%i_pca%i_", layer, m_pcas[i]);
    name += histname;
    if (i == 0)
      m_hist_nominal = (TH1F*)m_files_shifted[zv_index]->Get(name.c_str());
    m_histos[i] = (TH1F*)m_files_shifted[zv_index]->Get(name.c_str());
    m_ratios[i] = (TH1F*)m_histos[i]->Clone();
    m_ratios[i]->Divide(m_hist_nominal);
  }
  m_currentPCA = 0;
  m_currentLayer = layer;
  m_info = Form("Vertex z %s, Layer %i", m_vertexZPositions[zv_index].c_str(),
                m_currentLayer);
}

void TFCSVertexZPositionStudies::normalizeHistograms() {
  const int nhistos = m_histos.size();

  m_hist_nominal->Scale(1. / m_hist_nominal->Integral());
  for (int i = 0; i < nhistos; i++) {
    m_histos[i]->Scale(1. / m_histos[i]->Integral());
    delete m_ratios[i];
    m_ratios[i] = (TH1F*)m_histos[i]->Clone();
    m_ratios[i]->Divide(m_hist_nominal);
  }
}

void TFCSVertexZPositionStudies::deleteHistograms() {
  const int nhistos = m_histos.size();
  for (int i = 0; i < nhistos; i++) {
    delete m_histos[i];
    delete m_ratios[i];
  }
  m_histos.clear();
  m_ratios.clear();

  delete m_hist_nominal;
}

void TFCSVertexZPositionStudies::findBinning(bool useMMbinning, double factor,
                                             double quantile) {

  int nbins = m_hist_nominal->GetNbinsX();

  vector<double> prob = {quantile, 1 - quantile};

  vector<double> binning_orig(nbins + 1);

  const int nhistos = m_histos.size();
  TH1F* h1 = (TH1F*)m_histos[0]->Clone();
  for (int i = 1; i < nhistos; i++) h1->Add(m_histos[i]);

  vector<double> quantiles(prob.size());
  h1->GetQuantiles(prob.size(), &quantiles[0], &prob[0]);

  for (int i = 0; i < (int)prob.size(); i++) cout << quantiles[i] << std::endl;

  binning_orig[0] = h1->GetXaxis()->GetBinLowEdge(1);
  for (int ibin = 1; ibin <= nbins; ibin++) {
    binning_orig[ibin] = h1->GetXaxis()->GetBinUpEdge(ibin);
  }

  auto it_low =
      lower_bound(binning_orig.begin(), binning_orig.end(), quantiles[0]);
  auto it_up =
      upper_bound(binning_orig.begin(), binning_orig.end(), quantiles[1]);
  // auto it_low=binning_orig.begin();
  // auto it_up=binning_orig.end();

  cout << *it_low << " " << *it_up << endl;

  m_binning.clear();
  m_binning = vector<double>(it_low, it_up);
  cout << m_binning.size() << " " << m_binning[0] << " " << m_binning.back()
       << endl;

  TH1F* h_temp = (TH1F*)h1->Rebin(
      m_binning.size() - 1, (TString)h1->GetName() + "_temp", &m_binning[0]);
  double stats[4];
  h_temp->GetStats(stats);
  double Neff = sqrt(stats[0] * stats[0] / stats[1]) / nhistos;

  double bin_width_opt = 0.;

  if (useMMbinning)
    bin_width_opt = factor;
  else {
    prob = {0.25, 0.75};
    h_temp->GetQuantiles(prob.size(), &quantiles[0], &prob[0]);

    bin_width_opt = 2. * (quantiles[1] - quantiles[0]) * pow(Neff, -1. / 3);
    // double sigma = h_temp->GetMeanError();
    // double bin_width_opt = 3.49*sigma*pow(Neff,-1./3);

    bin_width_opt *= factor;  // Require finer binning than optimal
  }

  int Nbins_final =
      (m_binning[m_binning.size() - 1] - m_binning[0]) / bin_width_opt;

  Nbins_final = max(3, Nbins_final);
  Nbins_final = min((int)m_binning.size() - 1, Nbins_final);

  int res = (m_binning.size() - 1) % Nbins_final;
  if (res % 2 == 0) {
    m_binning.erase(m_binning.begin(), m_binning.begin() + res / 2);
    m_binning.erase(m_binning.end() - res / 2, m_binning.end());
  } else {
    m_binning.erase(m_binning.begin(), m_binning.begin() + res / 2);
    m_binning.erase(m_binning.end() - res / 2 - 1, m_binning.end());
  }

  vector<double> binning(Nbins_final + 1);

  binning[0] = m_binning[0];
  for (int i = 1; i <= Nbins_final; i++) {
    binning[i] = m_binning[i * (m_binning.size() - 1) / Nbins_final];
  }
  binning[Nbins_final] = m_binning.back();
  m_binning.clear();

  m_binning = binning;

  cout << Neff << " "
       << (m_binning[m_binning.size() - 1] - m_binning[0]) /
              (2. * (quantiles[1] - quantiles[0]) * pow(Neff, -1. / 3)) << " "
       << Nbins_final << " " << m_binning.size() << " "
       << (m_binning.back() - m_binning[0]) / Nbins_final << endl;

  // for(int i=1;i<=Nbins_final;i++)cout << m_binning[i] << " ";
  // cout << endl;

  delete h_temp;
}

void TFCSVertexZPositionStudies::rebinHistos() {
  TH1F* h_temp = (TH1F*)m_hist_nominal->Rebin(
      m_binning.size() - 1, m_hist_nominal->GetName(), &m_binning[0]);
  delete m_hist_nominal;
  m_hist_nominal = h_temp;
  const int nhistos = m_histos.size();

  for (int i = 0; i < nhistos; i++) {
    h_temp = (TH1F*)m_histos[i]->Rebin(
        m_binning.size() - 1, m_hist_nominal->GetName(), &m_binning[0]);
    delete m_histos[i];
    m_histos[i] = h_temp;

    delete m_ratios[i];
    m_ratios[i] = (TH1F*)m_histos[i]->Clone();
    m_ratios[i]->Divide(m_hist_nominal);
  }
}

void TFCSVertexZPositionStudies::printMeanValues() {}

void TFCSVertexZPositionStudies::plotHistograms(string outputDir,
                                                bool ratio_plots,
                                                bool drawErrorBars,
                                                bool useLogScale) {

  if (ratio_plots)
    plotHistograms(m_ratios, outputDir, drawErrorBars, useLogScale);
  else
    plotHistograms(m_histos, outputDir, drawErrorBars, useLogScale);
}

void TFCSVertexZPositionStudies::plotHistograms(vector<TH1F*>& histos,
                                                string outputDir,
                                                bool drawErrorBars,
                                                bool useLogScale) {

  unique_ptr<TCanvas> c = make_unique<TCanvas>("c1", "", 900, 800);
  unique_ptr<TLegend> leg = make_unique<TLegend>(0.7, 0.7, 0.9, 0.9);

  float max = FLT_MIN, min = FLT_MAX;

  const int nhistos = m_histos.size();

  if (useLogScale) {

    for (int i = 0; i < nhistos; i++)
      for (int ibin = 1; ibin < histos[i]->GetNbinsX(); ibin++) {
        if (histos[i]->GetBinContent(ibin) < 0) {
          histos[i]->SetBinContent(ibin, 0.);
          histos[i]->SetBinError(ibin, 0.);
        }
      }
  }

  for (int i = 0; i < nhistos; i++) {

    max = histos[i]->GetMaximum() > max ? histos[i]->GetMaximum() : max;
    min = histos[i]->GetMinimum(0.) < min ? histos[i]->GetMinimum(0.) : min;

    histos[i]->GetYaxis()->SetTitle(m_ytitle);
    histos[i]->GetYaxis()->SetTitleOffset(1.4);
    histos[i]->GetXaxis()->SetTitle(m_xtitle);
  }

  TString drawOption = "HIST";
  if (drawErrorBars) {
    drawOption = "HIST E1";
    gStyle->SetEndErrorSize(10);
  }
  for (int i = 0; i < nhistos; i++) {
    histos[i]->SetTitle("");
    histos[i]->SetLineColor(i + 1);
    histos[i]->SetMarkerColor(i + 1);
    histos[i]->SetLineWidth(2);
    leg->AddEntry(histos[i], m_legendNames[i].c_str());

    if (i == 0) {
      if (useLogScale)
        histos[i]->GetYaxis()->SetRangeUser(min / 2, max * 1000);
      else
        histos[i]->GetYaxis()->SetRangeUser(0.5 * min, max * 2);
      histos[i]->Draw(drawOption);
    }
    histos[i]->Draw(drawOption + "SAME");
  }
  leg->Draw();

  printATLASlabel(0.04, 0.2, 0.9, "Simulation Internal");

  WriteInfo(m_info, 0.04, 0.2, 0.85);
  WriteInfo((m_particle_type + ", " + m_particle_energy).c_str(), 0.04, 0.2,
            0.8);
  WriteInfo((m_particle_eta).c_str(), 0.04, 0.2, 0.75);

  // TString
  // figurename=outDir+Form("Ratio_layer%i_pca%i_overlay",layers[ilayer],pca);
  if (useLogScale) c->SetLogy();
  c->SaveAs((outputDir + "/" + m_name + ".png").c_str());
}

void printATLASlabel(float size, float x, float y, string text) {

  TLatex l;
  if (size > 0) l.SetTextSize(size);
  l.SetNDC();
  l.SetTextFont(72);
  l.SetTextColor(1);
  l.DrawLatex(x, y, "ATLAS");

  TString label = text.c_str();

  float shift = 0.13;

  TString ToWrite = "";
  l.SetNDC();
  l.SetTextFont(42);
  l.SetTextSize(size);
  l.SetTextColor(1);

  l.DrawLatex(x + shift, y, text.c_str());
}

void WriteInfo(string info, float size, float x, float y, int color) {
  TLatex l;
  l.SetNDC();
  l.SetTextFont(42);
  l.SetTextSize(size);
  l.SetTextColor(color);
  l.DrawLatex(x, y, info.c_str());
}
