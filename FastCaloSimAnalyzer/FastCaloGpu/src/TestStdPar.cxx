#include "TestStdPar.h"
#include "CountingIterator.h"

#include <algorithm>
#include <execution>
#include <iostream>
#include <vector>
#include <atomic>

void TestStdPar::testAll(unsigned long num) {
  test_vecInt(num);
  test_vecFloat(num);
  test_atomicAdd_int(num);
  test_atomicAdd_float(num);
}

void TestStdPar::test_floatArray(unsigned long num) {

    std::cout << "---------- test_floatArray( " << num << " ) -------------\n";

    double sum(0.);
    float *pf = new float[num];
    for (int i=0; i<num; ++i) {
      pf[i] = float(i) / 100.;
      if (i<10) { printf("float test CPU: %d %f\n", i, pf[i]); }
      sum += pf[i] + 1.;
    }

    std::for_each_n(std::execution::par_unseq, counting_iterator(0), num,
                    [=](int i) {
                      if (i<10) { printf(" float test GPU: %d %f\n", i, pf[i]); }
                      pf[i] += 1.;
                    } );


    double sum2(0.);
    for (int i=0; i<num; ++i) {
      sum2 += pf[i];
    }
    

    if (float(sum) == float(sum2)) {
      std::cout << "test OK: " << sum << " == " << sum2 << std::endl;
    } else {
      std::cout << "test FAIL: " << sum << " != " << sum2 << std::endl;
    }

    delete pf;
    
    std::cout << "------- done test_floatArray -------------\n";
    
  }
    


void TestStdPar::test_vecFloat(unsigned long num) {

    std::cout << "---------- test_vec<float>( " << num << " ) -------------\n";

    double sum(0.);
    std::vector<float>* pvec = new std::vector<float>;
    pvec->resize(num);

    float* pdat = pvec->data();

    for (int i=0; i<num; ++i) {
      pdat[i] = float(i) / 100.;
      if (i<10) { printf("vec test CPU: %d %f\n", i, pdat[i]); }
      sum += pdat[i] + 1.;
    }

    std::for_each_n(std::execution::par_unseq, counting_iterator(0), num,
                    [=](int i) {
                      if (i<10) { printf(" vec test GPU: %d %f\n", i, pdat[i]); }
                      pdat[i] += 1.;
                    } );

    double sum2(0.);
    for (int i=0; i<num; ++i) {
      sum2 += pdat[i];
    }
    

    if (float(sum) == float(sum2)) {
      std::cout << "test OK: " << sum << " == " << sum2 << std::endl;
    } else {
      std::cout << "test FAIL: " << sum << " != " << sum2 << std::endl;
    }

    delete pvec;
    
    std::cout << "------- done test_vec(num) -------------\n";
    
  }

void TestStdPar::test_vecInt(unsigned long num) {

    std::cout << "---------- test_vec<int>( " << num << " ) -------------\n";

    long int sum(0);
    std::vector<int>* pvec = new std::vector<int>;
    pvec->resize(num);

    int* pdat = pvec->data();

    for (int i=0; i<num; ++i) {
      pdat[i] = i;
      if (i<10) { printf("vec test CPU: %d %d\n", i, pdat[i]); }
      sum += pdat[i] + 1000;
    }

    std::for_each_n(std::execution::par_unseq, counting_iterator(0), num,
                    [=](int i) {
                      if (i<10) { printf(" vec test GPU: %d %d\n", i, pdat[i]); }
                      pdat[i] += 1000;
                    } );

    long int sum2(0);
    for (int i=0; i<num; ++i) {
      sum2 += pdat[i];
    }
    

    if (sum == sum2) {
      std::cout << "test OK: " << sum << " == " << sum2 << std::endl;
    } else {
      std::cout << "test FAIL: " << sum << " != " << sum2 << std::endl;
    }

    delete pvec;
    
    std::cout << "------- done test_vec<int>(num) -------------\n";
    
  }


  
void TestStdPar::test_atomicAdd_int(unsigned long num) {
    std::cout << "---------- test_atomic<int>_add -------------\n";
    std::atomic<int> *ii = new std::atomic<int>{0};
    std::for_each_n(std::execution::par_unseq, counting_iterator(0), num,
                    [=](int i) {
                      int j = (*ii)++;
                      printf("%d %d\n",i,j);
                    } );
    std::cout << "   after loop: " << *ii << " (should be " << num << ")" <<std::endl;
    std::cout << "---------- done test_atomic<int>_add -------------\n\n";
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void TestStdPar::test_atomicAdd_float(unsigned long N) {
    std::cout << "---------- test_atomicAdd_float -------------\n";
    
    float ta[N]{0.}, tc[N]{0.};
    for (int i=0; i<N; ++i) {
      ta[i%2] += i;
      tc[i] += i;
    }

    
    float *fa = new float[N];
    float *fb = new float[N];
    float *fc = new float[N];
    std::for_each_n(std::execution::par_unseq, counting_iterator(0), N,
                    [=] (int i) {
                      fb[i%2] += i;
#if defined ( _NVHPC_STDPAR_NONE ) || defined ( _NVHPC_STDPAR_MULTICORE )
                      fa[i % 2] += i;
                      fc[i] += i;
#else
                      atomicAdd(&fa[i%2],float(i));
                      atomicAdd(&fc[i],float(i));
#endif
                    });
    for (int i=0; i<N; ++i) {
      printf("%d : %2g [%2g] %g  %g [%g]\n",i, fa[i], ta[i], fb[i], fc[i], tc[i]);
    }
    std::cout << "---------- done test_atomicAdd_float -------------\n\n";
  }
