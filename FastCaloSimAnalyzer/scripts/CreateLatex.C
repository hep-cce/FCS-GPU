/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "TROOT.h"
#include "TFile.h"
#include "TH2.h"
#include "TString.h"
#include "TLatex.h"

#include <iostream>

void CreateLatex() {


	std::vector<int> layer;
// std::vector<int> pca;

	layer.push_back(0);
	layer.push_back(1);
	layer.push_back(2);
	layer.push_back(3);
	layer.push_back(12);
	//layer.push_back(13);
	//layer.push_back(14);


	std::string file = "/Users/ahasib/Documents/Analysis/FastCaloSim/ISF_FastCaloSim/run/output/shape_para/el_1mm/plot_books/InputDistribution_el_1mm_50.000000GeV_eta_0.200000_0.250000.tex";




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
		mylatex << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl
		        ;
		mylatex << "\\title{Plotbook: 65 GeV, 0.20$< |\\eta| <$ 0.25 el}" << endl;
		mylatex << "\\author{ Hasib Ahmed}" << endl;
		mylatex << "\\institute{ University of Edinburgh}" << endl;
		mylatex << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;
		mylatex << "\\begin{document}" << endl;
		mylatex << " \\maketitle" << endl;

		mylatex << "\\begin{frame}" << endl;
		mylatex << "\\frametitle{Parameters}" << endl;
		mylatex << "\\begin{center}" << endl;
		mylatex << "\\begin{itemize}" << endl;
		mylatex << "\\item Particle = $\\gamma^{0}$" << endl;
		mylatex << "\\item Energy = 50 GeV" << endl;
		mylatex << "\\item  0.20 $< |\\eta| <$ 0.25"  << endl;

		mylatex << "\\end{itemize}" << endl;
		mylatex << "\\end{center}" << endl;
		mylatex << "\\end{frame}" << endl;




		for (int i = 0; i < layer.size(); ++i)
		{
			for (int j = 1; j < 6; ++j)
			{
				std::string inDir = "/Users/ahasib/Documents/Analysis/FastCaloSim/ISF_FastCaloSim/run/output/shape_para/el_1mm/plots_layer" + std::to_string(layer.at(i)) + "_pca" + std::to_string(j) + "/";

				std::string fig0  = "NNinput_hEnergy_zoom0.png";
				std::string fig1  = "NNinput_hEnergy_zoom1.png";
				std::string fig2  = "NNinput_hEnergy_zoom2.png";

				std::string Nfig0  = "NNinput_hEnergyNorm_zoom0.png";
				std::string Nfig1  = "NNinput_hEnergyNorm_zoom1.png";
				std::string Nfig2  = "NNinput_hEnergyNorm_zoom2.png";

				mylatex << "\\begin{frame}" << endl;
				mylatex << "\\frametitle{Calolayer and PCA bin}" << endl;
				mylatex << "\\begin{center}" << endl;
				mylatex << "\\begin{itemize}" << endl;
				mylatex << "\\item Calorimeter layer = " + std::to_string(layer.at(i)) << endl;
				mylatex << "\\item PCA bin = " + std::to_string(j) << endl;
				mylatex << "\\end{itemize}" << endl;
				mylatex << "\\end{center}" << endl;
				mylatex << "\\end{frame}" << endl;

				// mylatex << "\\begin{frame}" << endl;
				// mylatex << "\\frametitle{Gradient Energy}" << endl;
				// mylatex << "\\begin{center}" << endl;
				// mylatex << "\\includegraphics[width=8cm]{" + inDir + "EnergyDensityGradient.png" + "}" << endl;
				// mylatex << "\\end{center}" << endl;
				// mylatex << "\\end{frame}" << endl;

				mylatex << "\\begin{frame}" << endl;
				mylatex << "\\frametitle{Input distribution: Energy zoom0}" << endl;
				mylatex << "\\begin{center}" << endl;
				mylatex << "\\includegraphics[width=8cm]{" + inDir + fig0 + "}" << endl;
				mylatex << "\\end{center}" << endl;
				mylatex << "\\end{frame}" << endl;

				mylatex << "\\begin{frame}" << endl;
				mylatex << "\\frametitle{Input distribution: normalized Energy zoom0}" << endl;
				mylatex << "\\begin{center}" << endl;
				mylatex << "\\includegraphics[width=8cm]{" + inDir + Nfig0 + "}" << endl;
				mylatex << "\\end{center}" << endl;
				mylatex << "\\end{frame}" << endl;

				mylatex << "\\begin{frame}" << endl;
				mylatex << "\\frametitle{Input distribution: Energy zoom1}" << endl;
				mylatex << "\\begin{center}" << endl;
				mylatex << "\\includegraphics[width=8cm]{" + inDir + fig1 + "}" << endl;
				mylatex << "\\end{center}" << endl;
				mylatex << "\\end{frame}" << endl;

				mylatex << "\\begin{frame}" << endl;
				mylatex << "\\frametitle{Input distribution: normalized Energy zoom1}" << endl;
				mylatex << "\\begin{center}" << endl;
				mylatex << "\\includegraphics[width=8cm]{" + inDir + Nfig1 + "}" << endl;
				mylatex << "\\end{center}" << endl;
				mylatex << "\\end{frame}" << endl;

				mylatex << "\\begin{frame}" << endl;
				mylatex << "\\frametitle{Input distribution: Energy zoom2}" << endl;
				mylatex << "\\begin{center}" << endl;
				mylatex << "\\includegraphics[width=8cm]{" + inDir + fig2 + "}" << endl;
				mylatex << "\\end{center}" << endl;
				mylatex << "\\end{frame}" << endl;

				mylatex << "\\begin{frame}" << endl;
				mylatex << "\\frametitle{Input distribution: normalized Energy zoom2}" << endl;
				mylatex << "\\begin{center}" << endl;
				mylatex << "\\includegraphics[width=8cm]{" + inDir + Nfig2 + "}" << endl;
				mylatex << "\\end{center}" << endl;
				mylatex << "\\end{frame}" << endl;

			}
		}

		mylatex << "\\end{document}" << endl;
		mylatex.close();
	}


}