/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

// root -l -q -b tools/testSimpleLatShape.Croot -l tools/testSimpleLatShape.C

void testSimpleLatShape(){
    gStyle->SetOptStat(0);

    
    //gSystem->Load("libISF_FastCaloSimParametrizationLib.so"); // if compiled by cmt else use below
    
    gSystem->AddIncludePath(" -I../../../../Calorimeter/CaloGeoHelpers ");
    gROOT->LoadMacro("Root/TFCSExtrapolationState.cxx+");
    gROOT->LoadMacro("Root/TFCSTruthState.cxx+");
    gROOT->LoadMacro("Root/TFCSSimulationState.cxx+");
    gROOT->LoadMacro("Root/TFCSParametrizationBase.cxx+");
    gROOT->LoadMacro("Root/TFCSParametrization.cxx+");
    gROOT->LoadMacro("Root/TFCSLateralShapeParametrization.cxx+");
    gROOT->LoadMacro("Root/TFCSSimpleLateralShapeParametrization.cxx+");


    // setup inputs
    //TString filename = "share/Tree_layer1_pi_PCAbin0_Alphabin8.root";
    TString filename = "share/Tree_layer2_e_PCAbin0_Alphabin8.root";

    TFCSSimpleLateralShapeParametrization get_sigmas;
    get_sigmas.Initialize(filename, "cumulative_histogram");

    TFCSSimpleLateralShapeParametrization param;
    param.Initialize(get_sigmas.getSigma_x(), get_sigmas.getSigma_y());



    const TFCSTruthState* truth=new TFCSTruthState();
    const TFCSExtrapolationState* extrapol=new TFCSExtrapolationState();
    TFCSSimulationState simulstate;
    param.simulate(simulstate, truth, extrapol);



    // grab original histogram
    TFile *f = new TFile(filename);
    TH2D *orig = ((TH2D*)f->Get("cumulative_histogram"));

    TFile *nFile = new TFile("pdfs/outputs_test.root","recreate");
    orig->SetDirectory(nFile);

    float x_bound=15, y_bound=15;
    //Double_t levels[] = {0, 0.004,0.008,0.012,0.016,0.02,0.024,0.028,0.032,0.036,0.04};
    Double_t levels[] = {0, 0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.02};

    TCanvas *c_orig = new TCanvas("orig_pol_c","orig_pol_c");
    TH2D *h2_orig_pol = new TH2D("orig_pol","orig_pol",100,-1*x_bound,x_bound,100,-1*y_bound,y_bound);
    h2_orig_pol->Draw();


    orig->Scale( 1./ orig->Integral(0,100000));
    orig->SetContour(10, levels);
    orig->GetZaxis()->SetRangeUser(0,0.02);

    orig->Draw("pol colz same");
    //h2_orig_pol->GetXaxis()->SetRangeUser(-0.02,0.02);
    //h2_orig_pol->GetYaxis()->SetRangeUser(-0.02,0.02);
    c_orig->SaveAs("pdfs/orig_pol.pdf");


    // Fill new hist based on parametrization
    TH2D *new_pol = (TH2D*)orig->Clone();
    new_pol->Reset();
    new_pol->SetName("new_pol");

    double x,y;
    int nEvents = 100000;

    for (int i = 0; i < nEvents; i++)
    {
        param.getHitXY(x,y); // get x and y hit positions from fit
        //std::cout<<"x: "<<x<<" y: "<<y<<std::endl;

        double r = TMath::Sqrt(x*x + y*y);
        double phi = TMath::ATan2( y, x) ;
        phi = (phi > 0 ? phi : 2*TMath::Pi() + phi) + TMath::Pi()/8;

        new_pol->Fill(phi, r);

    }

    TH2D *h2_new_pol = new TH2D("new_pol_template","new_pol_template",100,-1*x_bound,x_bound,100,-1*y_bound,y_bound);
    h2_new_pol->Draw();

    new_pol->Scale(1./new_pol->Integral());
    new_pol->SetContour(10, levels);
    new_pol->GetZaxis()->SetRangeUser(0,0.02);

    new_pol->Draw("pol colz same");
    c_orig->SaveAs("pdfs/new_pol.pdf");








    nFile->Write();
    nFile->Close();
}
