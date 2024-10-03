#ifndef FCS_TEST_STDPAR
#define FCS_TEST_STDPAR 1

class TestStdPar {

public:

  void testAll(unsigned long);

  void test_floatArray(unsigned long);
  void test_vecInt(unsigned long);
  void test_vecFloat(unsigned long);
  void test_atomicAdd_int(unsigned long);
  void test_atomicAdd_float(unsigned long);
  
};

#endif
