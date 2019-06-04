#ifndef GeoLoadGpu_H
#define GeoLoadGpu_H

//This header can be use both gcc and nvcc host part

#include <map>

#include "GeoRegion.h"

struct Rg_Sample_Index {
        int size ;
        int index ;
};


struct GeoGpu {
	unsigned long ncells ;
	CaloDetDescrElement* cells; 
	unsigned int nregions ;
	GeoRegion* regions ;
	int max_sample ;
	Rg_Sample_Index * sample_index ;
};	

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
   
    void set_ncells(unsigned long  nc) { m_ncells = nc ; };
    void set_nregions(unsigned int  nr) { m_nregions = nr;  };
    void set_cellmap( t_cellmap *  cm) { m_cells = cm; };
    void set_regions( GeoRegion *  r) { m_regions = r ; };
    void set_g_regions( GeoRegion *  gr) { m_regions_g = gr ; };
    void set_cells_g( CaloDetDescrElement *  gc) { m_cells_g = gc ; };
    void set_max_sample( int s) { m_max_sample = s ; };
    void set_sample_index_h( Rg_Sample_Index * s) { m_sample_index_h = s ; };

    bool LoadGpu() ;
    //bool LoadGpu_Region(GeoRegion * ) ;



protected :
    unsigned long m_ncells;
    unsigned int  m_nregions;
    t_cellmap *   m_cells ;
    GeoRegion*  m_regions  ;
    GeoRegion * m_regions_g  ;
    CaloDetDescrElement* m_cells_g ;
    Identifier *  m_cellid_array ;
    int m_max_sample ;
     Rg_Sample_Index *  m_sample_index_h ;

};
#endif
