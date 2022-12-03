/*
  Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
*/

TH1D* get_cumul(TH1D*);
double get_inverse1(double rnd);
double get_inverse2(double rnd);
double get_inverse3(double rnd);
double linear1(double y1,double y2,double x1,double x2,double y);
double linear2(double y1,double y2,double x1,double x2,double y, double w);
double linear3(double y1,double y2,double x1,double x2,double y);
vector<float> m_HistoBorders;
vector<float> m_HistoContents;
double get_maxdev(TH1D* h_in, TH1D* h_out);
double get_change(TH1D* histo);
TH1D* smart_rebin(TH1D* h_input);
TH1D* rebin(TH1D* hist, double);

void test_inverse()
{
 
 int inverse_type=2;
 cout<<"inverse_type (1 or 2 or 3) "<<endl;
 cin>>inverse_type;
 
 double cut_maxdev=0.0;  //1 (=1%) is more or less the standard value
 cout<<"cut maxdev "<<endl;
 cin>>cut_maxdev;
 
 //test the inverse on the total E
 int dsid=431004;
 
 TFile* file1=TFile::Open(Form("output/ds%i.firstPCA.ver01.root",dsid));
 
 double minE=55000;  double maxE=70000;
 TH1D* h_g4 =new TH1D("h_g4","h_g4",100,minE,maxE);
 TH1D* h_sim=new TH1D("h_sim","h_sim",100,minE,maxE);
 
 //get the g4 from the tree:
 
 TTree* tree = (TTree*)file1->Get("tree_1stPCA");
 double e;
 tree->SetBranchAddress("energy_totalE",&e);
 for(int event=0;event<tree->GetEntries();event++)
 {
  tree->GetEvent(event);
  h_g4->Fill(e);
 } //loop over g4
 
 //convert to cumul [0,1]
 TH1D* h_g4_cumul1=get_cumul(h_g4);
 h_g4_cumul1->SetName("h_g4_cumul1");
 
 //rebin
 TH1D* h_g4_cumul=rebin(h_g4_cumul1,cut_maxdev);
 
 //convert to vectors
 for(int b=1;b<=h_g4_cumul->GetNbinsX();b++)
  m_HistoBorders.push_back((float)h_g4_cumul->GetBinLowEdge(b));
 m_HistoBorders.push_back((float)h_g4_cumul->GetXaxis()->GetXmax());
 
 for(int b=1;b<h_g4_cumul->GetNbinsX();b++)
  //m_HistoContents.push_back(h_g4_cumul->GetBinContent(b)+(h_g4_cumul->GetBinContent(b+1)-h_g4_cumul->GetBinContent(b))/2.0);
  m_HistoContents.push_back(h_g4_cumul->GetBinContent(b));
 m_HistoContents.push_back(1);
 
 TRandom3* ran=new TRandom3(0);
 int ntoys=h_g4->GetEntries();
 cout<<"ntoys "<<ntoys<<endl;
 for(int i=1;i<=ntoys;i++)
 {
 	if(i%100000==0) cout<<i<<" done"<<endl;
  double rnd=ran->Uniform();
  double val=0.5;
  if(inverse_type==1) val=get_inverse1(rnd);
  if(inverse_type==2) val=get_inverse2(rnd);
  if(inverse_type==3) val=get_inverse3(rnd);
  h_sim->Fill(val);
 }//loop over toys
 
 h_sim->Scale(h_g4->Integral()/h_sim->Integral());
 
 double max=h_g4->GetBinContent(h_g4->GetMaximumBin());
 if(h_sim->GetBinContent(h_sim->GetMaximumBin())>max) max=h_sim->GetBinContent(h_sim->GetMaximumBin());
 
 h_sim->SetLineWidth(1);
 h_g4->SetLineWidth(1);
 h_g4_cumul->SetLineWidth(1);
 h_g4_cumul1->SetLineWidth(1);
 
 
 TCanvas* can=new TCanvas("can","can",0,0,1400,600);
 can->Divide(2);
 can->cd(1);
 h_g4->Draw("hist");
 h_g4->GetYaxis()->SetRangeUser(0.1,max*1.2);
 h_sim->SetLineStyle(2);
 h_sim->SetLineColor(2);
 h_sim->Draw("histsame");
 TLegend* leg=new TLegend(0.18,0.5,0.55,0.8);
 leg->SetFillStyle(0);
 leg->SetBorderSize(0);
 leg->AddEntry(h_g4,"Original","l");
 if(cut_maxdev>0.00001) leg->AddEntry(h_sim,Form("cumul->rebin->inverse%i",inverse_type),"l");
 else leg->AddEntry(h_sim,Form("cumul->inverse%i",inverse_type),"l");
 leg->Draw();
 can->cd(2);
 h_g4->Draw("hist");
 h_sim->Draw("histsame");
 can->cd(2)->SetLogy();
 can->Print(Form("inverse%i_maxdev%.1f.pdf",inverse_type,cut_maxdev));
 
 TCanvas* can2=new TCanvas("can2","can2",0,0,1400,600);
 can2->Divide(2);
 h_g4_cumul->SetLineColor(3); h_g4_cumul->SetLineStyle(2);
 can2->cd(1);
 h_g4_cumul1->Draw("hist");
 h_g4_cumul->Draw("histsame");
 TLegend* leg2=new TLegend(0.2,0.5,0.5,0.8);
 leg2->SetFillStyle(0);
 leg2->SetBorderSize(0);
 leg2->AddEntry(h_g4_cumul1,"Cumul original","l");
 leg2->AddEntry(h_g4_cumul,"Cumul rebinned","l");
 leg2->Draw();
 can2->cd(1)->SetLogy();
 can2->cd(2);
 TH1D* h_g4_cumul1_zoom=(TH1D*)h_g4_cumul1->Clone("h_g4_cumul1_zoom");
 TH1D* h_g4_cumul_zoom=(TH1D*)h_g4_cumul->Clone("h_g4_cumul_zoom");
 h_g4_cumul1_zoom->Draw("hist");
 h_g4_cumul1_zoom->GetXaxis()->SetRangeUser(64000,70000);
 h_g4_cumul1_zoom->GetYaxis()->SetRangeUser(0.9,1.02);
 h_g4_cumul_zoom->Draw("histsame");
 can2->Print(Form("cumul_maxdev%.1f.pdf",cut_maxdev));
 
 
}

TH1D* rebin(TH1D* hist, double cut_maxdev)
{
 
  double change=get_change(hist)*1.000001;  //increase slighlty for comparison of floats
  
  double maxdev=-1;
  
  TH1D* h_input=(TH1D*)hist->Clone("h_input");
  TH1D* h_output=0;
  
  int i=0;
  while(1)
  {
    
    TH1D* h_out;
    if(i==0)
    {
     h_out=(TH1D*)h_input->Clone("h_out");
    }
    else
    {
     h_out=smart_rebin(h_input); h_out->SetName("h_out");
    }
    
    maxdev=get_maxdev(hist,h_out);
    maxdev*=100.0;
    
    //if(i%100==0)
     cout<<"Iteration nr. "<<i<<" -----> change "<<change<<" bins "<<h_out->GetNbinsX()<<" -> maxdev="<<maxdev<<endl;
    
    if(maxdev<cut_maxdev && h_out->GetNbinsX()>5 && i<1000)
    {
      delete h_input;
      h_input=(TH1D*)h_out->Clone("h_input");
      change=get_change(h_input)*1.000001;
      delete h_out;
      i++;
    }
    else
    {
      h_output=(TH1D*)h_input->Clone("h_output");
      delete h_out;
      break;
    }
    
  }
  
  cout<<"Info: Rebinned histogram has "<<h_output->GetNbinsX()<<" bins."<<endl;
  
  return h_output;
  
}

TH1D* get_cumul(TH1D* hist)
{
  TH1D* h_cumul=(TH1D*)hist->Clone("h_cumul");
  double sum=0;
  for(int b=1;b<=h_cumul->GetNbinsX();b++)
  {
    sum+=hist->GetBinContent(b);
    h_cumul->SetBinContent(b,sum);
  }
  h_cumul->Scale(1.0/h_cumul->GetBinContent(h_cumul->GetNbinsX()));
  return h_cumul; 
}

double get_inverse1(double rnd)
{
  
  double value = 0.;
  
  if(rnd<m_HistoContents[0])
  {
   double x1=m_HistoBorders[0];
   double x2=m_HistoBorders[1];
   double y1=0;
   double y2=m_HistoContents[0];
   double x=linear1(y1,y2,x1,x2,rnd);
   value=x;
  }
  else
  {
   //find the first HistoContent element that is larger than rnd:
   vector<float>::iterator larger_element = std::upper_bound(m_HistoContents.begin(), m_HistoContents.end(), rnd);
   int index=larger_element-m_HistoContents.begin();
   double y=m_HistoContents[index];
   double x1=m_HistoBorders[index];
   double x2=m_HistoBorders[index+1];
   double y1=m_HistoContents[index-1];
   double y2=y;
   if((index+1)==((int)m_HistoContents.size()-1))
   {
    x2=m_HistoBorders[m_HistoBorders.size()-1];
    y2=1;
   }
   double x=linear1(y1,y2,x1,x2,rnd);
   value=x;
  }
  
  return value;
  
}

double linear1(double y1,double y2,double x1,double x2,double y)
{
  double x=-1;
  double eps=0.0000000001;
  if((y2-y1)<eps) x=x1;
  else
  {
    double m=(y2-y1)/(x2-x1);
    double n=y1-m*x1;
    x=(y-n)/m;
  }
  return x;
}

double get_inverse3(double rnd)
{
  double value = 0.;
  
  if(rnd<m_HistoContents[0])
  {
   double x1=m_HistoBorders[0];
   double x2=m_HistoBorders[1];
   double y1=0;
   double y2=m_HistoContents[0];
   double x=linear3(y1,y2,x1,x2,rnd);
   value=x;
  }
  else
  {
   //find the first HistoContent element that is larger than rnd:
   vector<float>::iterator larger_element = std::upper_bound(m_HistoContents.begin(), m_HistoContents.end(), rnd);
   int index=larger_element-m_HistoContents.begin();
   double y=m_HistoContents[index];
   double x1=m_HistoBorders[index];
   double x2=m_HistoBorders[index+1];
   double y1=m_HistoContents[index-1];
   double y2=y;
   if((index+1)==((int)m_HistoContents.size()-1))
   {
    x2=m_HistoBorders[m_HistoBorders.size()-1];
    y2=1;
   }
   double x =linear3(y1,y2,x1,x2,rnd);
   double x_old=linear1(y1,y2,x1,x2,rnd);
   //cout<<" x1 "<<x1<<" x2 "<<x2<<" y1 "<<y1<<" y2 "<<y2<<" rnd "<<rnd<<" x "<<x<<" x_old "<<x_old<<endl;
   value=x;
  }
  
  return value;
  
}


double linear3(double y1,double y2,double x1,double x2,double y)
{
  
  double x=-1;
  double eps=0.0000000001;
  if((y2-y1)<eps) x=x1;
  else
  {
    double b=(x1-x2)/(sqrt(y1)-sqrt(y2));
    double a=x1-b*sqrt(y1);
    x=a+b*sqrt(y);
  }
  return x;
}



double get_inverse2(double rnd)
{
  double value = 0.;
  
  
  if(rnd<m_HistoContents[0])
  {
   double x1=m_HistoBorders[0];
   double x2=m_HistoBorders[1];
   double y1=0;
   double y2=m_HistoContents[0];
   double w=m_HistoBorders[1]-m_HistoBorders[0];
   double x=linear2(y1,y2,x1,x2,rnd,w);
   value=x;
  }
  else
  {
   //find the first HistoContent element that is larger than rnd:
   vector<float>::iterator larger_element = std::upper_bound(m_HistoContents.begin(), m_HistoContents.end(), rnd);
   int index=larger_element-m_HistoContents.begin();
   double y=m_HistoContents[index];
   double x1=m_HistoBorders[index];
   double x2=m_HistoBorders[index+1];
   double y1=m_HistoContents[index-1];
   double y2=y;
   double w=m_HistoBorders[index+1]-m_HistoBorders[index];
   if((index+1)==((int)m_HistoContents.size()-1))
   {
    x2=m_HistoBorders[m_HistoBorders.size()-1];
    y2=1;
   }
   double x =linear2(y1,y2,x1,x2,rnd,w);
   value=x;
  }
  
  return value;
  
}


double linear2(double y1,double y2,double x1,double x2,double y,double w)
{
  
  double x=-1;
  double eps=0.0000000001;
  if((y2-y1)<eps) x=x1;
  else
  {
    /*
    double a=(y2-y1)/pow(x2-x1,2);
    x=sqrt(y-y1)/sqrt(a)+x1;
    */
    
    double a=(y2-y1)/pow(x2-x1,2);
    double c=(w*(y2-y1)-a/3*(pow(x2,3)-pow(x1,3)))/(x2-x1);
    x=sqrt(y-y1-c)/sqrt(a)+x1;
    cout<<"rnd "<<y<<" --> x1 "<<x1<<" x2 "<<x2<<" y1 "<<y1<<" y2 "<<y2<<" a "<<a<<" c "<<c<<" x "<<x<<endl;
  }
  
  return x;
}

double get_maxdev(TH1D* h_in, TH1D* h_out)
{
 
 double maxdev=0;
 for(int i=1;i<=h_in->GetNbinsX();i++)
 {
  int bin=h_out->FindBin(h_in->GetBinCenter(i));
  double val=fabs(h_out->GetBinContent(bin)-h_in->GetBinContent(i));
  if(val>maxdev) maxdev=val;
 }
 return maxdev;
 
}

double get_change(TH1D* histo)
{
 //return the smallest change between 2 bin contents, but don't check the last bin, because that one never gets merged
 double minchange=100.0;
 for(int b=2;b<histo->GetNbinsX();b++)
 {
  double diff=histo->GetBinContent(b)-histo->GetBinContent(b-1);
  if(diff<minchange && diff>0) minchange=diff;
 }
 
 return minchange;
 
}

TH1D* smart_rebin(TH1D* h_input)
{
  
  TH1D* h_out1=(TH1D*)h_input->Clone("h_out1");
  
  //get the smallest change
  double change=get_change(h_out1)*1.00001;
  
  vector<double> content;
  vector<double> binborder;
  
  binborder.push_back(h_out1->GetXaxis()->GetXmin());
  
  int merged=0;
  int skip=0;
  int secondlastbin_merge=0;
  for(int b=1;b<h_out1->GetNbinsX()-1;b++)  //never touch the last bin
  {
   double thisBin=h_out1->GetBinContent(b);
   double nextBin=h_out1->GetBinContent(b+1);
   double width  =h_out1->GetBinWidth(b);
   double nextwidth=h_out1->GetBinWidth(b+1);
   double diff=fabs(nextBin-thisBin);
   if(!skip && (diff>change || merged))
   {
    binborder.push_back(h_out1->GetBinLowEdge(b+1));
    content.push_back(thisBin);
   }
   skip=0;
   if(diff<=change && !merged)
   {
    double sum=thisBin*width+nextBin*nextwidth;
    double sumwidth=width+nextwidth;
    binborder.push_back(h_out1->GetBinLowEdge(b+2));
    content.push_back(sum/sumwidth);
    merged=1;
    skip=1;
    if(b==(h_out1->GetNbinsX()-2))
     secondlastbin_merge=1;
   }
  } //for b
  if(!secondlastbin_merge)
  {
   binborder.push_back(h_out1->GetBinLowEdge(h_out1->GetNbinsX()));
   content.push_back(h_out1->GetBinContent(h_out1->GetNbinsX()-1));
  }
  binborder.push_back(h_out1->GetXaxis()->GetXmax());
  content.push_back(h_out1->GetBinContent(h_out1->GetNbinsX()));
  
  double* bins=new double[content.size()+1];
  for(unsigned int i=0;i<binborder.size();i++)
   bins[i]=binborder[i];
  
  TH1D* h_out2=new TH1D("h_out2","h_out2",content.size(),bins);
  for(unsigned int b=1;b<=content.size();b++)
   h_out2->SetBinContent(b,content[b-1]);
  
  delete[] bins;
  delete h_out1;
  
  return h_out2;
  
}
