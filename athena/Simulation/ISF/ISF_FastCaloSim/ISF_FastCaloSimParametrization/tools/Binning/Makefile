Target  = PlotHits
Objects = 

ROOTFLAGS = $(shell root-config --cflags)
ROOTLIBS = $(shell root-config --libs) -lTMVA -lMinuit 
INCOURS = -I./ISF_FastCaloSimParametrization-00-00-28-02/
INCTOOL = -I./ISF_FastCaloSimParametrization/tools

all:$(Target)

PlotHits: PlotHits.cxx $(Objects)
	g++  $(INCOURS) $(INCTOOL) -o $@ PlotHits.cxx $(Objects) $(ROOTFLAGS) $(ROOTLIBS) 
