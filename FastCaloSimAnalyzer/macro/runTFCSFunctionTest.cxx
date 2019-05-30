#include "ISF_FastCaloSimEvent/TFCS1DFunctionTemplateHistogram.h"
#include "TH1.h"
#include "TCanvas.h"

void runTFCSFunctionTest()
{
  int binscale=4;
  const int ntest=3;
  TH1* inputhist[ntest];
  TString prefix[ntest]={"Triangle","Gauss4bin","Gauss16bin"};
  
  std::vector< const TFCS1DFunction* > functions;
  std::vector< float > bin_low_edges;
  
  TH1* histfine;

  histfine=new TH1D("Triangle","Triangle",64,0,1);
  for(int i=1;i<=histfine->GetNbinsX();++i) {
    double y;
    y=histfine->GetNbinsX()-i+4;
    if(i<=histfine->GetNbinsX()/2) y=i+4;
     else y=histfine->GetNbinsX()+1-i+4;
    if(y<0) y=0;
    histfine->SetBinContent(i,y);
    histfine->SetBinError(i,0.001);
  }  
  inputhist[0]=histfine;
  inputhist[1]=TFCS1DFunction::generate_histogram_random_gauss(4*binscale,1000000,1,4);
  inputhist[2]=TFCS1DFunction::generate_histogram_random_gauss(16*binscale,4000000,1,4);
  
  for(int itest=0;itest<ntest;++itest) {
    histfine=inputhist[itest];
    TH1* hist;
    if(itest>0) {
      hist=(TH1*)histfine->Clone(TString(histfine->GetName())+"_coarse");
      hist->Rebin(binscale);
      hist->Scale(1.0/binscale);
    } else {
      vector<double> xbins(histfine->GetNbinsX());
      int ibin=0;
      for(int i=1;i<=histfine->GetNbinsX();i+=binscale) {
        double xlow=histfine->GetBinLowEdge(i);
        cout<<"bin "<<i<<" -> "<<ibin+1<<" xlow="<<xlow<<endl;
        xbins[ibin]=xlow;
        if(i!=1+histfine->GetNbinsX()/2+3*binscale) ++ibin;
      }
      xbins[ibin]=histfine->GetXaxis()->GetXmax();
      hist=new TH1D(TString(histfine->GetName())+"_coarse",TString(histfine->GetTitle())+" coarse",ibin,xbins.data());
      for(int i=1;i<=histfine->GetNbinsX();++i) {
        hist->Fill(histfine->GetBinCenter(i),histfine->GetBinContent(i)/binscale);
      }
    }  
    hist->SetLineWidth(3);

    hist->SetName(prefix[itest]+"Int8Int8");
    histfine->SetName(TString(hist->GetName())+"_fine");
    TFCS1DFunction* funcInt8Int8=new TFCS1DFunctionInt8Int8Histogram(hist);
    TFCS1DFunction::unit_test(hist,funcInt8Int8,10000000,histfine);
    functions.push_back(funcInt8Int8);
    bin_low_edges.push_back(itest*5+0);

    hist->SetName(prefix[itest]+"InterpolInt8Int8");
    histfine->SetName(TString(hist->GetName())+"_fine");
    TFCS1DFunction* funcInterInt8Int8=new TFCS1DFunctionInt8Int8InterpolationHistogram(hist);
    TFCS1DFunction::unit_test(hist,funcInterInt8Int8,10000000,histfine);
    functions.push_back(funcInterInt8Int8);
    bin_low_edges.push_back(itest*5+1);
    
    hist->SetName(prefix[itest]+"Int8Int16");
    histfine->SetName(TString(hist->GetName())+"_fine");
    TFCS1DFunction* funcInt8Int16=new TFCS1DFunctionInt8Int16Histogram(hist);
    TFCS1DFunction::unit_test(hist,funcInt8Int16,10000000,histfine);
    functions.push_back(funcInt8Int16);
    bin_low_edges.push_back(itest*5+2);

    hist->SetName(prefix[itest]+"InterpolInt8Int16");
    histfine->SetName(TString(hist->GetName())+"_fine");
    TFCS1DFunction* funcInterInt8Int16=new TFCS1DFunctionInt8Int16InterpolationHistogram(hist);
    TFCS1DFunction::unit_test(hist,funcInterInt8Int16,10000000,histfine);
    functions.push_back(funcInterInt8Int16);
    bin_low_edges.push_back(itest*5+3);
    
    break;
  }
  bin_low_edges.push_back(100);
  TFCSHitCellMappingWiggle* wiggle_test=new TFCSHitCellMappingWiggle("WiggleTest","WiggleTest");
  wiggle_test->initialize(functions,bin_low_edges);
  TFile* f=TFile::Open("WiggleTest","Recreate");
  wiggle_test->Write();
  f->ls();
  f->Close();
  
  TFile* ftest=TFile::Open("WiggleTest");
  if(ftest) {
    ftest->ls();
    TFCSHitCellMappingWiggle* wiggle_test=(TFCSHitCellMappingWiggle*)ftest->Get("WiggleTest");
    if(wiggle_test) {
      cout<<"nbin="<<wiggle_test->get_number_of_bins()<<endl;
      for(int i=0;i<wiggle_test->get_number_of_bins();++i) {
        const TFCS1DFunction* func=wiggle_test->get_function(i);
        cout<<wiggle_test->get_bin_low_edge(i)<<" <= eta < "<<wiggle_test->get_bin_up_edge(i)<<" func="<<func<<endl;
        TH1* histfine=new TH1D(Form("Test%d",i),Form("Test %d",i),128,0,1);
        for(int irnd=0;irnd<10000000;++irnd) {
          double rnd=gRandom->Rndm();
          histfine->Fill(func->rnd_to_fct(rnd));
        }
        new TCanvas(Form("Test%d",i),Form("Test %d",i));
        histfine->Draw();
      }
    }
  }
  
  return;
  
  /*
  hist->SetName("Int8Int32Gauss");histfine->SetName(TString(hist->GetName())+"_fine");
  TFCS1DFunction* funcInt8Int32Gauss=new TFCS1DFunctionInt8Int32Histogram(hist);
  TFCS1DFunction::unit_test(hist,funcInt8Int32Gauss,10000000,histfine);
  */
  /*
  hist->SetName("InterInt8Int8Gauss");histfine->SetName(TString(hist->GetName())+"_fine");
  TFCS1DFunction* funcInterInt8Int8Gauss=new TFCS1DFunctionInt8Int8InterpolationHistogram(hist);
  TFCS1DFunction::unit_test(hist,funcInterInt8Int8Gauss,10000000,histfine);
  */
  
  /*
  hist->SetName("InterInt8Int16Gauss");histfine->SetName(TString(hist->GetName())+"_fine");
  TFCS1DFunction* funcInterInt8Int16Gauss=new TFCS1DFunctionInt8Int16InterpolationHistogram(hist);
  for(float p=0;p<1;p+=0.125) {
    float x=funcInterInt8Int16Gauss->rnd_to_fct(p);
    cout<<p<<": func="<<x<<endl;
  } 
  TFCS1DFunction::unit_test(hist,funcInterInt8Int16Gauss,10000000,histfine);
  */  

  return;  
  
  histfine=TFCS1DFunction::generate_histogram_random_gauss(64,1000000,1,4);

  histfine->SetName("Int8Int8Gaussfine");
  TFCS1DFunction* funcInt8Int8Gaussfine=new TFCS1DFunctionInt8Int8Histogram(histfine);
  TFCS1DFunction::unit_test(histfine,funcInt8Int8Gaussfine,10000000);
  
  histfine->SetName("Int8Int16Gaussfine");
  TFCS1DFunction* funcInt8Int16Gaussfine=new TFCS1DFunctionInt8Int16Histogram(histfine);
  TFCS1DFunction::unit_test(histfine,funcInt8Int16Gaussfine,10000000);

/*
  histfine->SetName("Int8Int32Gaussfine");
  TFCS1DFunction* funcInt8Int32Gaussfine=new TFCS1DFunctionInt8Int32Histogram(histfine);
  TFCS1DFunction::unit_test(histfine,funcInt8Int32Gaussfine,10000000);
*/  
}

