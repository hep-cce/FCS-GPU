#ifndef GeoRegion_G_H
#define GeoRegion_G_H


#include "CaloDetDescrElement_g.h"

class GeoRegion {
    public:
       __host__ __device__ GeoRegion() {
	m_all_cells = 0 ;
	m_xy_grid_adjustment_factor=0 ;
	m_index = 0 ;
	m_cell_grid_eta=0;
	m_cell_grid_phi=0;
	m_mineta=0;
	m_maxeta=0;
	m_minphi=0;
	m_maxphi=0;
	m_mineta_raw=0;
	m_maxeta_raw=0;
	m_minphi_raw=0;
	m_maxphi_raw=0;
	m_mineta_correction=0;
	m_maxeta_correction=0;
	m_minphi_correction=0;
	m_maxphi_correction=0;
	m_deta=0;
	m_dphi=0;
	m_eta_correction=0;
	m_phi_correction=0;
	m_dphi_double=0 ;
	m_deta_double=0 ;
	m_cells=0 ;
	m_cells_g=0;
	} ;

       __host__ __device__ ~GeoRegion() { free( m_cells) ;} ;


    __host__ __device__ void set_all_cells(CaloDetDescrElement * c) {m_all_cells = c ; };
    __host__ __device__ void set_xy_grid_adjustment_factor(float f) {m_xy_grid_adjustment_factor=f ; };
    __host__ __device__ void set_index(int i ) {m_index=i ; };
    __host__ __device__ void set_cell_grid_eta(int i) {m_cell_grid_eta=i ; };
    __host__ __device__ void set_cell_grid_phi(int i) {m_cell_grid_phi=i ; };
    __host__ __device__ void set_mineta(float f) {m_mineta=f ; };
    __host__ __device__ void set_maxeta(float f) {m_maxeta=f ; };
    __host__ __device__ void set_minphi(float f) {m_minphi=f ; };
    __host__ __device__ void set_maxphi(float f) {m_maxphi=f ; };
    __host__ __device__ void set_minphi_raw(float f) {m_minphi_raw=f ; };
    __host__ __device__ void set_maxphi_raw(float f) {m_maxphi_raw=f ; };
    __host__ __device__ void set_mineta_raw(float f) {m_mineta_raw=f ; };
    __host__ __device__ void set_maxeta_raw(float f) {m_maxeta_raw=f ; };
    __host__ __device__ void set_mineta_correction(float f) {m_mineta_correction=f ; };
    __host__ __device__ void set_maxeta_correction(float f) {m_maxeta_correction=f ; };
    __host__ __device__ void set_minphi_correction(float f) {m_minphi_correction=f ; };
    __host__ __device__ void set_maxphi_correction(float f) {m_maxphi_correction=f ; };
    __host__ __device__ void set_eta_correction(float f) {m_eta_correction=f ; };
    __host__ __device__ void set_phi_correction(float f) {m_phi_correction=f ; };
    __host__ __device__ void set_deta(float f) {m_deta=f ; };
    __host__ __device__ void set_dphi(float f) {m_dphi=f ; };
    __host__ __device__ void set_deta_double(float f) {m_deta_double=f ; };
    __host__ __device__ void set_dphi_double(float f) {m_dphi_double=f ; };
    __host__ __device__ void set_cell_grid( long long * cells ) {m_cells= cells ; };
    __host__ __device__ void set_cell_grid_g( long long * cells ) {m_cells_g = cells ; };


    __host__ __device__ long long * cell_grid( ) { return m_cells ; };
    __host__ __device__ long long * cell_grid_g( ) { return m_cells_g ; };
    __host__ __device__ int cell_grid_eta( ) { return m_cell_grid_eta ; };
    __host__ __device__ int cell_grid_phi( ) { return m_cell_grid_phi ; };
    __host__ __device__ int index() {return m_index ; };
    __host__ __device__ float mineta_raw( ) {return m_mineta_raw ; };
    __host__ __device__ float minphi_raw( ) {return m_minphi_raw ; };
    __host__ __device__  CaloDetDescrElement *all_cells( ) {return m_all_cells ; };
    __host__ __device__ float maxeta() {return m_maxeta ; };
    __host__ __device__ float mineta() {return m_mineta ; };
    __host__ __device__ float maxphi() {return m_maxphi ; };
    __host__ __device__ float minphi() {return m_minphi ; };




/*
    float mineta() const {return m_mineta;};
    float maxeta() const {return m_maxeta;};
    float minphi() const {return m_minphi;};
    float maxphi() const {return m_maxphi;};

    float mineta_raw() const {return m_mineta_raw;};
    float maxeta_raw() const {return m_maxeta_raw;};
    float minphi_raw() const {return m_minphi_raw;};
    float maxphi_raw() const {return m_maxphi_raw;};

    float minx() const {return m_mineta;};
    float maxx() const {return m_maxeta;};
    float miny() const {return m_minphi;};
    float maxy() const {return m_maxphi;};

    float minx_raw() const {return m_mineta_raw;};
    float maxx_raw() const {return m_maxeta_raw;};
    float miny_raw() const {return m_minphi_raw;};
    float maxy_raw() const {return m_maxphi_raw;};

    float deta() {return m_deta;};
    float dphi() {return m_dphi;};
    float mindeta() {return m_mindeta;};
    float mindphi() {return m_mindphi;};
    float dx() {return m_deta;};
    float dy() {return m_dphi;};
    float mindx() {return m_mindeta;};
    float mindy() {return m_mindphi;};

    float eta_correction() {return m_eta_correction;};
    float phi_correction() {return m_phi_correction;};
    float x_correction() {return m_eta_correction;};
    float y_correction() {return m_phi_correction;};

    int cell_grid_eta() const {return m_cell_grid_eta;};
    int cell_grid_phi() const {return m_cell_grid_phi;};
    void set_xy_grid_adjustment_factor(float factor) {m_xy_grid_adjustment_factor=factor;};
*/

    __host__ __device__ int raw_eta_position_to_index(float eta_raw) const {return floor((eta_raw-m_mineta_raw)/m_deta_double);};
    __host__ __device__ int raw_phi_position_to_index(float phi_raw) const {return floor((phi_raw-m_minphi_raw)/m_dphi_double);};

    __host__ __device__  bool index_range_adjust(int& ieta,int& iphi) ;
    __host__ __device__  float calculate_distance_eta_phi(const long long DDE,float eta,float phi,float& dist_eta0,float& dist_phi0) ;

    __host__ __device__  long long  getDDE(float eta,float phi,float* distance=0,int* steps=0) ;


    protected: 


      long long * m_cells  ;  // my cells array in the region HOST ptr
      long long * m_cells_g  ;  // my cells array in the region gpu ptr
    CaloDetDescrElement * m_all_cells ;   // all cells in GPU, stored in array.


    float m_xy_grid_adjustment_factor;
    int m_index ;
    int m_cell_grid_eta,m_cell_grid_phi; 
    float m_mineta,m_maxeta,m_minphi,m_maxphi;
    float m_mineta_raw,m_maxeta_raw,m_minphi_raw,m_maxphi_raw;
    float m_mineta_correction,m_maxeta_correction,m_minphi_correction,m_maxphi_correction;
    float  m_deta,m_dphi,m_eta_correction,m_phi_correction;
    float  m_dphi_double, m_deta_double ;

};
#endif
