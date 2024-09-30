#include "FastCaloSimAnalyzer/TFCSStdParTest.h"
#include "FastCaloGpu/TestStdPar.h"

#include <iostream>

void TFCSStdParTest::test(bool doAtomic, bool doVector, unsigned long num) {

  TestStdPar tst;

  if (doAtomic) {
    tst.test_atomicAdd_int(num);
    tst.test_atomicAdd_float(num);
  }

  if (doVector) {
      tst.test_floatArray(num);        
      tst.test_vecFloat(num);
      tst.test_vecInt(num);
  }
  
}
