OBJ = TFCS2DParametrizationAnalysisBase.o TFCS2DEventAnalyzer.o MakeTFCS2DEventAnalyzer.o
INC_DIR = -I ../
CC = g++
DEBUG = -g
CFLAGS = -std=c++11 -Wall -c $(DEBUG)
LFLAGS = -Wall
CXX=`root-config --cxx`
CXXFLAGS=`root-config --cflags --libs`
LDFLAGS=`root-config --ldflags`
LDLIBS=`root-config --glibs`
ROOTLIBS='-lRooFit -lHtml -lMinuit -lRooFitCore -lRooStats -lHistFactory'



MakeTFCS2DEventAnalyzer: $(OBJ)
	$(CC) $(LFLAGS) $(CXXFLAGS) $(OBJ) -o MakeTFCS2DEventAnalyzer

# CaloGeometryFromFile.o: CaloGeometryFromFile.h
# 	$(CC) $(INC_DIR) $(CFLAGS) $(CXXFLAGS) CaloGeometryFromFile.cxx

TFCS2DParametrizationAnalysisBase.o: TFCS2DParametrizationAnalysisBase.h
	$(CC) $(INC_DIR) $(CFLAGS) $(CXXFLAGS) TFCS2DParametrizationAnalysisBase.cxx

TFCS2DEventAnalyzer.o: TFCS2DEventAnalyzer.h TFCS2DParametrizationAnalysisBase.h
	$(CC) $(CFLAGS) $(CXXFLAGS) TFCS2DEventAnalyzer.cxx

MakeTFCS2DEventAnalyzer.o: TFCS2DEventAnalyzer.h
	$(CC) $(CFLAGS) $(CXXFLAGS) MakeTFCS2DEventAnalyzer.cxx



clean:
	\rm *.o *~ MakeTFCS2DEventAnalyzer

