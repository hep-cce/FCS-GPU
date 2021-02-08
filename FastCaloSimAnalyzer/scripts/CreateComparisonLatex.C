/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TROOT.h"
#include "TFile.h"
#include "TH2.h"
#include "TString.h"
#include "TLatex.h"

#include <iostream>


void CreateComparisonLatex() {


	std::vector<int> layer;

	layer.push_back(0);
	layer.push_back(1);
	layer.push_back(2);
	layer.push_back(3);
	// layer.push_back(12);
	// layer.push_back(13);
	// layer.push_back(14);



	std::string topDir = "/Users/hasib/Documents/Analysis/FastCaloSim/ISF_FastCaloSim/run/output/shape_para/comparison_1mm_opt/";

	std::string file = "/Users/hasib/Documents/Analysis/FastCaloSim/ISF_FastCaloSim/run/output/shape_para/comparison_1mm_opt/plot_book/ElectronComparisonPlotbook.tex";

	ofstream mylatex;
	mylatex.open(file.c_str());

	if (mylatex.is_open())
	{
		mylatex << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;
		mylatex << "\\documentclass[english,professionalfonts]{beamer}" << endl;
		mylatex << "\\usefonttheme{serif}" << endl;
		mylatex << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;
		mylatex << "\\usepackage{beamerthemesplit}" << endl;
		mylatex << "\\usepackage{multicol}" << endl;
		mylatex << "\\usepackage{amsmath}" << endl;
		mylatex << "\\usepackage{amssymb}" << endl;
		mylatex << "\\usepackage{array}" << endl;
		mylatex << "\\usepackage{graphicx}" << endl;
		mylatex << "\\usepackage{multimedia}" << endl;
		mylatex << "\\usepackage{hyperref}" << endl;
		mylatex << "\\usepackage{url}" << endl;
		mylatex << "%% Define a new 'leo' style for the package that will use a smaller font." << endl;
		mylatex << "\\makeatletter" << endl;
		mylatex << "\\def\\url@leostyle{\\@ifundefined{selectfont}{\\def\\UrlFont{\\sf}}{\\def\\UrlFont{\\small\\ttfamily}}}" << endl;
		mylatex << "\\makeatother" << endl;
		mylatex << "%% Now actually use the newly defined style." << endl;
		mylatex << "\\urlstyle{leo}" << endl;
		mylatex << "\\usepackage{cancel}" << endl;
		mylatex << "\\usepackage{color}" << endl;
		mylatex << "\\usepackage{verbatim}" << endl;
		mylatex << "\\usepackage{epsfig}" << endl;
		mylatex << "\\usepackage{fancybox}" << endl;
		mylatex << "\\usepackage{xcolor}" << endl;
		mylatex << "%\\usepackage{fontspec}" << endl;
		mylatex << "%\\usepackage{booktabs,caption}" << endl;
		mylatex << "%\\usepackage{eulervm}" << endl;
		mylatex << "\\usepackage{textpos}" << endl;
		mylatex << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;
		mylatex << "\\usetheme{Warsaw}" << endl;
		mylatex << "\\usecolortheme{default}" << endl;
		mylatex << "%\\setbeamercovered{transparent}" << endl;
		mylatex << "\\beamertemplatenavigationsymbolsempty" << endl;
		mylatex << "%\\setbeamertemplate" << endl;
		mylatex << "\\setbeamertemplate{footline}[page number]" << endl;
		mylatex << "%%\\setsansfont{Fontin Sans}" << endl;
		mylatex << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;
		mylatex << "\\definecolor{jgreen}{cmyk}{0.99,0,0.52,0}" << endl;
		mylatex << "\\definecolor{green}{rgb}{0.,.6,0.}" << endl;
		mylatex << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;
		mylatex << "\\title{Plotbook: Electron Merging schemes}" << endl;
		mylatex << "\\author{ Hasib Ahmed}" << endl;
		mylatex << "\\institute{ University of Edinburgh}" << endl;
		mylatex << " % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % " << endl;
		mylatex << "\\begin{document}" << endl;
		mylatex << " \\maketitle" << endl;

		mylatex << "\\begin{frame}" << endl;
		mylatex << "\\frametitle{Parameters}" << endl;
		mylatex << "\\begin{center}" << endl;
		mylatex << "\\begin{itemize}" << endl;
		mylatex << "\\item Particle = $\\pi^ {\\pm}$" << endl;
		mylatex << "\\item Energy = 50 GeV" << endl;
		mylatex << "\\item  0.20 $ < | \\eta | < $ 0.25"  << endl;
		mylatex << "\\item Merging schemes: 1mm and optimized" << endl;
		mylatex << "\\item 5mm  binning is used for the plots except for $\\delta\\eta$ in EMB1µ" << endl;
		mylatex << "\\end{itemize}" << endl;
		mylatex << "\\includegraphics[width = 5cm] {" + topDir + "optimized_merging.png}" << endl;
		mylatex << "\\end{center}" << endl;
		mylatex << "\\end{frame}" << endl;




		for (int i = 0; i < layer.size(); ++i)
		{
			for (int j = 1; j < 5; ++j)
			{

				std::string hitseta  = "hits_deta_layer" + std::to_string(layer.at(i)) + "_pca" + std::to_string(j) + ".png";
				std::string hitsphi = "hits_dphi_layer" + std::to_string(layer.at(i)) + "_pca" + std::to_string(j) + ".png";
				std::string energyeta = "energy_deta_layer" + std::to_string(layer.at(i)) + "_pca" + std::to_string(j) + ".png";
				std::string energyphi = "energy_dphi_layer" + std::to_string(layer.at(i)) + "_pca" + std::to_string(j) + ".png";

				std::string label = "EMB" + std::to_string(layer.at(i)) + ", bin(PCA) = " + std::to_string(j);

				mylatex << "\\begin{frame}" << endl;
				mylatex << "\\frametitle{" + label + "}" << endl;
				mylatex << "\\includegraphics[width = 6cm] {" + topDir + hitseta + "}" << endl;
				mylatex << "\\includegraphics[width = 6cm] {" + topDir + energyeta + "}" << endl;
				mylatex << "\\end{frame}" << endl;

				mylatex << "\\begin{frame}" << endl;
				mylatex << "\\frametitle{" + label + "}" << endl;
				mylatex << "\\includegraphics[width = 6cm] {" + topDir + hitsphi + "}" << endl;
				mylatex << "\\includegraphics[width = 6cm] {" + topDir + energyphi + "}" << endl;
				mylatex << "\\end{frame}" << endl;

			}
		}

		mylatex << "\\end{document}" << endl;
		mylatex.close();
	}












}
