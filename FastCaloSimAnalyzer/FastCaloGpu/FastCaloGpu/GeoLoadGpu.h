#ifndef GeoLoadGpu_H
#define GeoLoadGpu_H

//This header can be use both gcc and nvcc host part

#include <map>

#include "GeoGpu_structs.h"

typedef std::map< Identifier , const CaloDetDescrElement* > t_cellmap;

class GeoLoadGpu 
{
public :
    GeoLoadGpu(){
	m_cells=0 ;  m_cells_g=0 ;
	m_ncells=0; m_nregions=0; m_regions=0 ; 
	m_regions_g=0; Geo_g=0 ; } ;

    ~GeoLoadGpu() { delete m_cellid_array ; } ;

    static  struct GeoGpu *  Geo_g ; 
    static  unsigned long num_cells;
   
    void set_ncells(unsigned long  nc) { m_ncells = nc ; };
    void set_nregions(unsigned int  nr) { m_nregions = nr;  };
    void set_cellmap( t_cellmap *  cm) { m_cells = cm; };
    void set_regions( GeoRegion *  r) { m_regions = r ; };
    void set_g_regions( GeoRegion *  gr) { m_regions_g = gr ; };
    void set_cells_g( CaloDetDescrElement *  gc) { m_cells_g = gc ; };
    void set_max_sample( int s) { m_max_sample = s ; };
    void set_sample_index_h( Rg_Sample_Index * s) { m_sample_index_h = s ; };
    const CaloDetDescrElement* index2cell(unsigned long index) { return (*m_cells)[ m_cellid_array[index]] ; };  

    bool LoadGpu() ;
    //bool LoadGpu_Region(GeoRegion * ) ;



protected :
    unsigned long m_ncells;  //number of cells
    unsigned int  m_nregions;  // number of regions
    t_cellmap *   m_cells ;  // from Geometry class
    GeoRegion*  m_regions  ;  //array of regions on host
    GeoRegion * m_regions_g  ;  //array of region on GPU
    CaloDetDescrElement* m_cells_g ;   //Cells in GPU
    Identifier *  m_cellid_array ;  //cell id to Indentifier lookup table
    int m_max_sample ;     //Max number of samples
     Rg_Sample_Index *  m_sample_index_h ;   //index for flatout of  GeoLookup over sample 

};
#endif
