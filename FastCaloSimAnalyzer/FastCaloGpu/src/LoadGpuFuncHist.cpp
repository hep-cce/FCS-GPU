#include "LoadGpuFuncHist.h"
#include "CU_BigMem.h"

CU_BigMem *CU_BigMem::bm_ptr;

LoadGpuFuncHist::~LoadGpuFuncHist() {
  free(m_hf);
  /*
    hipFree((*m_hf_d).low_edge);
    hipFree((*m_hf_d).h_szs);
    for(unsigned int i=0 ; i< (*m_d_hf).nhist ; ++i ){
      hipFree((*m_hf_d).h_contents[i]);
      hipFree((*m_hf_d).h_borders[i]);
    }
  */
  free(m_hf_d);
  //  hipFree(m_d_hf) ;

  free(m_hf2d);
  /*
   hipFree((*m_hf2d_d).h_bordersx);
   hipFree((*m_hf2d_d).h_bordersy);
   hipFree((*m_hf2d_d).h_contents);
  */
  free(m_hf2d_d);
  // hipFree(m_d_hf2d) ;
}

void LoadGpuFuncHist::LD2D() {
  if (!m_hf2d) {
    std::cout << "Error Load 2DFunctionHisto " << std::endl;
    return;
  }

  FH2D *hf_ptr = new FH2D;
  FH2D hf = {0, 0, 0, 0, 0};

  hf.nbinsx = (*m_hf2d).nbinsx;
  hf.nbinsy = (*m_hf2d).nbinsy;

  CU_BigMem *p = CU_BigMem::bm_ptr;

  hf.h_bordersx = p->cu_bm_alloc<float>(hf.nbinsx + 1);
  hf.h_bordersy = p->cu_bm_alloc<float>(hf.nbinsy + 1);
  hf.h_contents = p->cu_bm_alloc<float>(hf.nbinsy * hf.nbinsx);

  gpuQ(hipMemcpy(hf.h_bordersx, (*m_hf2d).h_bordersx,
                 (hf.nbinsx + 1) * sizeof(float), hipMemcpyHostToDevice));
  gpuQ(hipMemcpy(hf.h_bordersy, (*m_hf2d).h_bordersy,
                 (hf.nbinsy + 1) * sizeof(float), hipMemcpyHostToDevice));
  gpuQ(hipMemcpy(hf.h_contents, (*m_hf2d).h_contents,
                 (hf.nbinsx * hf.nbinsy) * sizeof(float),
                 hipMemcpyHostToDevice));

  *(hf_ptr) = hf;
  m_hf2d_d = hf_ptr;

  m_d_hf2d = p->cu_bm_alloc<FH2D>(1);
  gpuQ(hipMemcpy(m_d_hf2d, m_hf2d_d, sizeof(FH2D), hipMemcpyHostToDevice));
}

void LoadGpuFuncHist::LD() {
  // this call  assume  already have Histofuncs set in m_hf
  // this function allocate memory of GPU and deep copy m_hf to m_d_hf

  if (!m_hf) {
    std::cout << "Error Load WiggleHistoFunctions " << std::endl;
    return;
  }

  FHs hf = {0, 0, 0, 0, 0, 0};
  hf.s_MaxValue = (*m_hf).s_MaxValue;
  hf.nhist = (*m_hf).nhist;
  unsigned int *h_szs = (*m_hf).h_szs; // already allocateded on host ;

  CU_BigMem *p = CU_BigMem::bm_ptr;

  hf.low_edge = p->cu_bm_alloc<float>(hf.nhist + 1);
  gpuQ(hipMemcpy(hf.low_edge, (*m_hf).low_edge, (hf.nhist + 1) * sizeof(float),
                 hipMemcpyHostToDevice));

  hf.h_szs = p->cu_bm_alloc<unsigned int>(hf.nhist);
  gpuQ(hipMemcpy(hf.h_szs, (*m_hf).h_szs, hf.nhist * sizeof(unsigned int),
                 hipMemcpyHostToDevice));

  hf.h_contents = p->cu_bm_alloc<uint32_t *>(hf.nhist);
  hf.h_borders = p->cu_bm_alloc<float *>(hf.nhist);

  uint32_t **contents_ptr = (uint32_t **)malloc(hf.nhist * sizeof(uint32_t *));
  float **borders_ptr = (float **)malloc(hf.nhist * sizeof(float *));

  for (unsigned int i = 0; i < hf.nhist; ++i) {

    contents_ptr[i] = p->cu_bm_alloc<uint32_t>(h_szs[i]);
    borders_ptr[i] = p->cu_bm_alloc<float>(h_szs[i] + 1);

    gpuQ(hipMemcpy(contents_ptr[i], (*m_hf).h_contents[i],
                   h_szs[i] * sizeof(uint32_t), hipMemcpyHostToDevice));
    gpuQ(hipMemcpy(borders_ptr[i], (*m_hf).h_borders[i],
                   (h_szs[i] + 1) * sizeof(float), hipMemcpyHostToDevice));
  }

  gpuQ(hipMemcpy(hf.h_contents, contents_ptr, hf.nhist * sizeof(uint32_t *),
                 hipMemcpyHostToDevice));
  gpuQ(hipMemcpy(hf.h_borders, borders_ptr, hf.nhist * sizeof(float *),
                 hipMemcpyHostToDevice));

  m_d_hf = p->cu_bm_alloc<FHs>(1);
  gpuQ(hipMemcpy(m_d_hf, &hf, sizeof(FHs), hipMemcpyHostToDevice));

  free(contents_ptr);
  free(borders_ptr);

  m_hf_d = &hf;
}
