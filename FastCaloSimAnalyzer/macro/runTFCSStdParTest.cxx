#include <cstdio>
#include <map>
#include <iostream>
#include "citer.h"
#include <docopt/docopt.h>

#include "FastCaloSimAnalyzer/TFCSStdParTest.h"

static const char* USAGE =
    R"(Run test for stdpar

Usage:
  runTFCSStdParTest [--doAtomicTest] [--doVectorTest] [-n <size> | --num <size>]
  runTFCSStdParTest (-h | --help)

Options:
  -h --help                    Show help screen.
  --doAtomicTest               Do test for atomic increments [default: false].
  --doVectorTest               Do test for allocating/accessing vector<float> [default: false].
  -n <size> --num <size>       Size of array to allocate for tests [default: 10].
)";


int main( int argc, char** argv ) {

  std::map<std::string, docopt::value> args = docopt::docopt( USAGE, {argv + 1, argv + argc}, true );

  bool doAtomic = args["--doAtomicTest"].asBool();
  bool doVec    = args["--doVectorTest"].asBool();
  int  num      = args["--num"].asLong();
  
  TFCSStdParTest test;

  test.test(doAtomic,doVec,num);

  return 0;
}
