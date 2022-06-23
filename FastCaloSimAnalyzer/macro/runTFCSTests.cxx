#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "FastCaloSimAnalyzer/TFCSShapeValidation.h"
#include <chrono>
#include <omp.h>
#include <iostream>

#define MAXHITS 200000
#define MAXBINS 1024

void Draw_1Dhist( TH1* hist1, double ymin = 0, double ymax = 0, bool logy = false, TString name = "",
                  TString title = "", TCanvas* c = 0, bool png = false ) {

  double min1, max1, rmin1, rmax1;
  TFCSAnalyzerBase::autozoom( hist1, min1, max1, rmin1, rmax1 );

  return;
}

//int main ( int argc, char* argv[] ) {
//  
//  double a = 1.0;
//  std::cout << "Executed" << std::endl;
//  return 0;
//}

//unsigned int Factorial( unsigned int number ) {
//  return number > 1 ? Factorial(number-1)*number : 1;
//}
//
//TEST_CASE( "Factorials are computed" ) {
//
//  //std::cout << "Start test" << std::endl;
//  //REQUIRE( 1 == 1 );
//    REQUIRE( Factorial(1) == 1 );
//    REQUIRE( Factorial(2) == 2 );
//    REQUIRE( Factorial(3) == 6 );
//    REQUIRE( Factorial(10) == 3628800 );
//
//    long seed = 42;
//
//    void*    m_rd4h;
//    m_rd4h = CaloGpuGeneral::Rand4Hits_init( MAXHITS, MAXBINS, seed, true );
//
//}
//
