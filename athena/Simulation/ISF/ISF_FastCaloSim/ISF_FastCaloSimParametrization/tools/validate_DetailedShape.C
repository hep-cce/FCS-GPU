/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

void validate_DetailedShape(){

  gSystem->AddIncludePath(" -I.. ");

  int Ntoys=10000;
  TRandom1* myRandom =new TRandom1(); myRandom->SetSeed(0);
  TRandom1* myRandom2=new TRandom1(); myRandom2->SetSeed(1000);

  const TFCSTruthState* truth=new TFCSTruthState();
  const TFCSExtrapolationState* extrapol=new TFCSExtrapolationState();

  for(int i=0;i<Ntoys;i++)
    {
      double random  = myRandom->Uniform(0,1);
      double random2 = myRandom2->Uniform(0,1);

      if(i%100==0)
        cout<<"Now run simulation for Toy "<<i<<endl;

      TFCSNNLateralShapeParametrization* shapetest=new TFCSNNLateralShapeParametrization("shapetest","shapetest");

      TFCSSimulationState simulstate;
      simulstate.set_Ebin(randombin);
      shapetest->simulate(simulstate, truth, extrapol);

    }
  
  std::cout << "END " <<  std::endl;

}

