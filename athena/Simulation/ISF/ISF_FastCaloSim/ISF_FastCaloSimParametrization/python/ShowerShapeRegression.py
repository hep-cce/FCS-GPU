# Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration

__author__ = 'Christopher Bock - LMU'


class ShowerShapeRegressor():
    def __init__(self, root_file_name='ISF_HitAnalysispion.root', base_output_folder='RegressionOutput'):
        """ Most of the important settings are configured/set within this function.

        TODO: Make these options configurable via an external options file to ease the use of the script.
        TODO: Make remaining hardcoded options configurable here.
        """

        self.neuron_type = 'tanh'   # linear does not work when fitting the efrac without log, seems to cause huge troubles somewhere

        self.log_efrac = False

        self.num_neurons = 10
        self.root_file_name = root_file_name
        self.base_output_folder = base_output_folder

        self.truth_energy_threshold = 40000

        self.check_output_folders()

        self.etaphi_nbins = [400, 400]
        self.etaphi_xrange = [-0.4, 0.4]
        self.etaphi_yrange = [-0.4, 0.4]

        self.cumulative_etaphi_nbins = [400, 400]
        self.cumulative_etaphi_xrange = [-0.4, 0.4]
        self.cumulative_etaphi_yrange = [-0.4, 0.4]

        self.equidistant_binning = False

        # self.etaphi_nbins = [200, 200]
        # self.etaphi_xrange = [-0.2, 0.2]
        # self.etaphi_yrange = [-0.2, 0.2]
        #
        # self.cumulative_etaphi_nbins = [200, 200]
        # self.cumulative_etaphi_xrange = [-0.2, 0.2]
        # self.cumulative_etaphi_yrange = [-0.2, 0.2]

        self.eta_phi_efrac_hists = []

        self.cumulated_events = 0
        self.cumulated_events_threshold = 1000

        self.layer_names = ["PreSamplerB", "EMB1", "EMB2", "EMB3",
                            "PreSamplerE", "EME1", "EME2", "EME3",
                            "HEC0", "HEC1", "HEC2", "HEC3",
                            "TileBar0", "TileBar1", "TileBar2",
                            "TileGap1", "TileGap2", "TileGap3",
                            "TileExt0", "TileExt1", "TileExt2",
                            "FCAL0", "FCAL1", "FCAL2"]

        self.selected_layer = 13

        # Trees needed for regression
        self.cum_distribution_trees = None

        from array import array
        self.cum_eta_array = array('f', [0.])
        self.cum_phi_array = array('f', [0.])
        self.cum_r_array = array('f', [0.])
        self.cum_efrac_array = array('f', [0.])

        self.regression_method = 'MLP'
        #self.regression_method = 'FDA_GA'

        self.base_file_name = 'no_name'
        self.mlp_base_file_name = 'no_name'
        self.obtain_output_names()

        self.do_regression = True

    def initialize_cumulative_trees(self):
        import ROOT
        self.cum_distribution_trees = ROOT.TTree('cumulative_distributions',
                                                 'Cumulative eta phi energy fraction distributions')

        self.cum_distribution_trees.Branch('d_eta', self.cum_eta_array, 'd_eta/F')
        self.cum_distribution_trees.Branch('d_phi', self.cum_phi_array, 'd_phi/F')
        self.cum_distribution_trees.Branch('r', self.cum_r_array, 'r/F')
        self.cum_distribution_trees.Branch('energy_fraction', self.cum_efrac_array, 'energy_fraction/F')

        pass

    def fill_regression_tree(self, cumulative_histogram):
        """ Uses the supplied histogram to fill the regression tree with values. Depending on self.log_efrac the trees
        are either filled with the logarithm of the energy fraction or with just the energy fraction.
        """
        import math

        n_bins_x = cumulative_histogram.GetNbinsX()
        n_bins_y = cumulative_histogram.GetNbinsY()

        for x in xrange(n_bins_x + 1):
            eta_value = cumulative_histogram.GetXaxis().GetBinCenter(x)
            for y in xrange(n_bins_y + 1):
                value = cumulative_histogram.GetBinContent(x, y)
                if value <= 0:
                    continue

                phi_value = cumulative_histogram.GetYaxis().GetBinCenter(y)
                r_value = math.sqrt(phi_value*phi_value + eta_value*eta_value)

                if self.log_efrac:
                    self.cum_efrac_array[0] = math.log(value)
                else:
                    self.cum_efrac_array[0] = value

                self.cum_eta_array[0] = eta_value
                self.cum_phi_array[0] = phi_value
                self.cum_r_array[0] = r_value

                self.cum_distribution_trees.Fill()

        pass

    def run_regression(self):
        import ROOT

        output_file = ROOT.TFile("%s.root" % self.mlp_base_file_name, "RECREATE")

        factory = ROOT.TMVA.Factory(self.mlp_base_file_name, output_file, "!V:!Silent:!Color:DrawProgressBar")

        factory.AddVariable("d_eta")
        factory.AddVariable("d_phi")
        factory.AddVariable("r")

        factory.AddTarget("energy_fraction")

        factory.AddRegressionTree(self.cum_distribution_trees, 1.0)

        cut = ROOT.TCut('')

        factory.PrepareTrainingAndTestTree(cut, 'nTrain_Regression=0:nTest_Regression=0:SplitMode=Random:NormMode=NumEvents:!V')

        # Default: HiddenLayers=N+20
        factory.BookMethod(ROOT.TMVA.Types.kMLP, "MLP",
                           "!H:!V:VarTransform=Norm:NeuronType=%s:NCycles=20000:HiddenLayers=N+%i:TestRate=6:TrainingMethod=BFGS:Sampling=0.3:SamplingEpoch=0.8:ConvergenceImprove=1e-6:ConvergenceTests=15:!UseRegulator"
                           % (self.neuron_type, self.num_neurons))

        factory.TrainAllMethods()
        factory.TestAllMethods()
        factory.EvaluateAllMethods()

        output_file.Close()

        pass

    def run(self):
        """ This function iterates over all events and calls for each event the process_entry function. The event loop
        breaks either once process_entry returns false or once all events in the input file have been processed.
        """
        import ROOT

        print('--------------------------------------------')
        print('Welcome to ShowerShapeRegressor.py')
        print('Using %i neurons of type %s using log of efrac: %s' % (self.num_neurons, self.neuron_type,
                                                                      str(self.log_efrac)))
        print('--------------------------------------------')

        self.obtain_output_names()

        f = ROOT.TFile(self.root_file_name)
        chain = ROOT.gDirectory.Get('ISF_HitAnalysis/CaloHitAna')
        entries = chain.GetEntriesFast()

        for current_entry in xrange(entries):
            print(' Loading entry: %i' % current_entry)
            j = chain.LoadTree(current_entry)
            if j < 0:
                break

            # copy next entry into memory and verify
            bites_read = chain.GetEntry(current_entry)
            if bites_read <= 0:
                continue

            if not self.process_entry(chain, current_entry):
                break

        if len(self.eta_phi_efrac_hists) > 0:
            self.create_output()

        pass

    def process_entry(self, chain, event_number):
        """ This processes a single event in the chain. It first checks whether this event should be processed by
        calling the take_event_into_account function. If this function returns true processing of this event continues
        else the event will be skipped.
        Only a single layer (self.selected_layer) will be processed. A histogram will be filled with the energy
        distribution vs eta and phi. Corrections by the track to calo extrapolator will be applied here. Once all
        events have been finished the histogram will be divided by its integral.
        Once the desired number of events (self.cumulated_events_threshold) has been reached the self.create_output()
        function will be called.
        """
        import ROOT

        if not self.take_event_into_account(chain, event_number):
            return True

        num_hits = len(chain.HitX)

        eta_phi_efrac_dist = ROOT.TH2D('eta_phi_efrac_dist_%i' % event_number, 'eta vs phi vs efrac',
                                       self.cumulative_etaphi_nbins[0], self.cumulative_etaphi_xrange[0],
                                       self.cumulative_etaphi_xrange[1], self.cumulative_etaphi_nbins[1],
                                       self.cumulative_etaphi_yrange[0], self.cumulative_etaphi_yrange[1])
        for i in xrange(num_hits):
            layer_id = chain.HitSampling[i]

            if not layer_id == self.selected_layer:
                continue

            pos = ROOT.TVector3(chain.HitX[i], chain.HitY[i], chain.HitZ[i])
            eta = pos.PseudoRapidity()
            phi = pos.Phi()

            eta_correction = chain.TTC_entrance_eta[0][layer_id]
            phi_correction = chain.TTC_entrance_phi[0][layer_id]

            if eta_correction < -900:
                eta_correction = 0
                print('no valid eta_correction found')

            if phi_correction < -900:
                phi_correction = 0
                print('no valid phi_correction found')

            d_eta = eta - eta_correction
            d_phi = self.phi_mpi_pi(phi - phi_correction)

            eta_phi_efrac_dist.Fill(d_eta, d_phi, chain.HitE[i])

        self.cumulated_events += 1

        if eta_phi_efrac_dist.Integral() > 0:
                eta_phi_efrac_dist.Scale(1.0/eta_phi_efrac_dist.Integral())

        self.eta_phi_efrac_hists.append(eta_phi_efrac_dist)
        if self.cumulated_events >= self.cumulated_events_threshold:
            self.create_output()
            return False

        return True

    def take_event_into_account(self, chain, event_number):
        """ This function is used to filter events.
        """
        if chain.TruthE[0] < self.truth_energy_threshold:
            print('  Pion does not pass energy threshold of %f skipping process_entry!' %
                  self.truth_energy_threshold)
            return False

        return True

    def create_output(self):
        """ This function creates the average shower shape from all the energy fraction distributions and plots it. In
        case self.do_regression is set to True, it will initialize the input trees used by TMVA (using
        self.initialize_cumulative_trees), it will fill them (using self.fill_regression_tree) and run the regression
        by calling self.run_regression. Once TMVA finished, regression test/control plots and our shower shape control
        plots will be created.
        """
        import ROOT

        cumulative_histogram = self.create_eta_phi_histogram(self.base_file_name, 'cumulative eta vs phi vs efrac')

        if self.equidistant_binning:
            for hist in self.eta_phi_efrac_hists:
                cumulative_histogram.Add(hist)

            if cumulative_histogram.Integral() > 0:
                cumulative_histogram.Scale(1.0/cumulative_histogram.Integral())
        else:
            # first fill the cumulative histogram
            x_axis = self.eta_phi_efrac_hists[0].GetXaxis()
            n_bins_x = x_axis.GetNbins()

            y_axis = self.eta_phi_efrac_hists[0].GetYaxis()
            n_bins_y = y_axis.GetNbins()
            for x_bin in xrange(n_bins_x + 1):
                x_center = x_axis.GetBinCenter(x_bin)
                for y_bin in xrange(n_bins_y + 1):
                    y_center = x_axis.GetBinCenter(y_bin)
                    for hist in self.eta_phi_efrac_hists:
                        value = hist.GetBinContent(x_bin, y_bin)
                        cumulative_histogram.Fill(x_center, y_center, value)

            # now calculate the PDF
            x_axis = cumulative_histogram.GetXaxis()
            n_bins_x = x_axis.GetNbins()

            y_axis = cumulative_histogram.GetYaxis()
            n_bins_y = y_axis.GetNbins()
            for x_bin in xrange(n_bins_x + 1):
                x_bin_width = x_axis.GetBinWidth(x_bin)
                for y_bin in xrange(n_bins_y + 1):
                    y_bin_width = y_axis.GetBinWidth(y_bin)
                    area = x_bin_width*y_bin_width

                    cumulative_content = cumulative_histogram.GetBinContent(x_bin, y_bin) / area
                    cumulative_histogram.SetBinContent(x_bin, y_bin, cumulative_content)

            if cumulative_histogram.Integral() > 0:
                cumulative_histogram.Scale(1.0/cumulative_histogram.Integral())

        from os import path
        cumulative_etaphi_file_name = path.join(self.base_output_folder, 'cumulative_eta_phi',
                                                'average_shower_%s.pdf' % self.base_file_name)
        cumulative_etaphi_root_file_name = path.join(self.base_output_folder, 'cumulative_eta_phi',
                                                     '%s.root' % self.base_file_name)

        root_file = ROOT.TFile(cumulative_etaphi_root_file_name, 'RECREATE')
        root_file.cd()
        self.plot_root_hist2d(cumulative_histogram.Clone(), cumulative_etaphi_file_name, '#eta', '#phi',
                              root_file=root_file)
        root_file.Write()
        root_file.Close()

        if self.do_regression:
            if cumulative_histogram.Integral() > 0:
                cumulative_histogram.Scale(1.0/cumulative_histogram.Integral())

            self.initialize_cumulative_trees()
            self.fill_regression_tree(cumulative_histogram)

            self.run_regression()
            self.plot_regression_control_plots()
            self.shower_test_plot_tmva(cumulative_histogram)
            self.shower_test_plot_tmva(cumulative_histogram, False)
            self.shower_test_plot_getrandom2(cumulative_histogram)

        for hist in self.eta_phi_efrac_hists:
            hist.Delete()

        self.eta_phi_efrac_hists = []
        self.cumulated_events = 0

        return

    """
    =========================================== Output Folder Functions ===============================================
    The following are functions used to create the output folders and to obtain the names of the outputs.
    ===================================================================================================================
    """

    @staticmethod
    def check_folder(folder):
        import os
        from os import path
        if not path.exists(folder):
            print('Output folder %s does not exist, will try to create it.' % folder)
            os.makedirs(folder)
            if not path.exists(folder):
                raise Exception('ERROR: Could not create output folder!')
            else:
                print('Basic output folder successfully created :)')

    def check_output_folders(self):
        from os import path
        self.check_folder(self.base_output_folder)
        self.check_folder(path.join(self.base_output_folder, 'cumulative_eta_phi'))
        pass

    def obtain_output_names(self):
        ext = ''
        if self.log_efrac:
            ext = '_logEfrac'

        eq_bin = ''
        if self.equidistant_binning:
            eq_bin = '_eqbin'

        self.base_file_name = 'Evts%i_Lay%i_E%i_eta%.2f_PID%i%s' \
                              % (self.cumulated_events_threshold, self.selected_layer, 50, 0.20, 211, eq_bin)
        self.mlp_base_file_name = 'NN_reg%s_%s_NNeur%i_NT%s_%s' % (ext, self.regression_method, self.num_neurons,
                                                                   self.neuron_type, self.base_file_name)

    """
    =============================================== Plot Functions ====================================================
    The following functions  areused to create the plots. The "plot_regression_control_plots" function is a
    reimplementation of a similar function found in TMVA's examples. It searches for TMVA's control plots inside the
    root file and plots them.
    ===================================================================================================================
    """

    def create_eta_phi_histogram(self, name, title):
        """ This is a helper function to create the eta phi histograms either with equidistant or non-equidistant
        binning depending on the settings. Please use this function to avoid code duplication.
        """

        import ROOT
        if self.equidistant_binning:
            eta_phi_histogram = ROOT.TH2D(name, title,
                                          self.cumulative_etaphi_nbins[0], self.cumulative_etaphi_xrange[0],
                                          self.cumulative_etaphi_xrange[1], self.cumulative_etaphi_nbins[1],
                                          self.cumulative_etaphi_yrange[0], self.cumulative_etaphi_yrange[1])
        else:
            import array
            x_binning = [-0.4, -0.1, -0.05, -0.04, -0.03, -0.02, -0.016, -0.014, -0.012, -0.01, -0.008, -0.006,
                         -0.004, -0.002, 0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.02, 0.03, 0.04,
                         0.05, 0.1, 0.4]
            y_binning = x_binning

            eta_phi_histogram = ROOT.TH2D(name, title,
                                          len(x_binning) - 1, array.array('d', x_binning),
                                          len(y_binning) - 1, array.array('d', y_binning))

        return eta_phi_histogram

    def shower_test_plot_getrandom2(self, histogram):
        """ This function uses the input histogram's GetRandom2 function to draw random numbers according to the
        distribution found in the input histogram and fills 50000 entries into a new histogram and in the end prints
        it to a file.
        """
        import ROOT
        from os import path
        ROOT.gROOT.SetBatch(True)

        ROOT.gStyle.SetNumberContours(999)

        d_eta = ROOT.Double(0)
        d_phi = ROOT.Double(0)

        eta_phi_efrac_dist = self.create_eta_phi_histogram('shower_test_plot',
                                                           'eta vs phi vs efrac - using TH2F::GetRandom2')

        accepted_points = 0
        accepted_points_target = 50000
        while accepted_points < accepted_points_target:
            histogram.GetRandom2(d_eta, d_phi)

            eta_phi_efrac_dist.Fill(d_eta, d_phi, 1)
            accepted_points += 1

        if eta_phi_efrac_dist.Integral() > 0:
            eta_phi_efrac_dist.Scale(1.0/eta_phi_efrac_dist.Integral())
        self.plot_root_hist2d(eta_phi_efrac_dist,
                              path.join(self.base_output_folder, '%s_th2f.pdf' % self.mlp_base_file_name), '#eta', '#phi')
        self.difference_to_reference(histogram, eta_phi_efrac_dist,
                                     path.join(self.base_output_folder, '%s_th2f_diff.pdf' % self.mlp_base_file_name),
                                     '#eta', '#phi')

        self.difference_to_reference(histogram, eta_phi_efrac_dist,
                                     path.join(self.base_output_folder, '%s_th2f_diff_inf.pdf' % self.mlp_base_file_name),
                                     '#eta', '#phi', inverted_ratio=True)

        pass

    def shower_test_plot_tmva(self, reference, randomize_pos=True):
        """ This function uses the NN obtained by TMVA to plot the fitted shower shape and the ratio to the reference.
        In case randomize_pos is set to true (which is the default), random positions (flat distributed) in the eta-phi
        plane will drawn. The NN will be evaluated there and the value will be filled into the output histogram. This
        method is faster than drawing a third random number and accepting the point based on whether this rnd number is
        greater than the NN at this point or not.
        If randomize_pos is set to false, a histogram will be filled by evaluating the NN at each point and filling the
        value into the histogram.
        """
        import ROOT
        import math
        ROOT.gROOT.SetBatch(True)

        ROOT.gStyle.SetNumberContours(999)

        reader = ROOT.TMVA.Reader('!Color')

        from array import array
        d_eta = array('f', [0.])
        d_phi = array('f', [0.])
        r = array('f', [0])
        reader.AddVariable('d_eta', d_eta)
        reader.AddVariable('d_phi', d_phi)
        reader.AddVariable('r', r)

        from os import path
        base_weight_name = self.mlp_base_file_name
        method_name = self.regression_method
        weight_file = path.join('weights', '%s_%s.weights.xml' % (base_weight_name, method_name))
        reader.BookMVA(self.regression_method, weight_file)

        ext = ''
        if randomize_pos:
            ext = ' rnd pos'

        eta_phi_efrac_dist = self.create_eta_phi_histogram('shower_test_plot', 'eta vs phi vs efrac - using %s%s' %
                                                           (self.regression_method, ext))
        if randomize_pos:
            accepted_points = 0
            accepted_points_target = 50000
            while accepted_points < accepted_points_target:
                d_eta[0] = ROOT.gRandom.Rndm() - 0.5
                d_phi[0] = ROOT.gRandom.Rndm() - 0.5
                r[0] = math.sqrt(d_eta[0]*d_eta[0] + d_phi[0]*d_phi[0])

                value = reader.EvaluateRegression(self.regression_method)[0]

                if self.log_efrac:
                    value = math.exp(value)

                eta_phi_efrac_dist.Fill(d_eta[0], d_phi[0], value)
                accepted_points += 1
        else:
            x_axis = reference.GetXaxis()
            y_axis = reference.GetYaxis()
            n_bins_x = x_axis.GetNbins()
            n_bins_y = y_axis.GetNbins()
            for x in xrange(n_bins_x + 1):
                d_eta[0] = x_axis.GetBinCenter(x)
                for y in xrange(n_bins_y + 1):
                    d_phi[0] = y_axis.GetBinCenter(y)
                    r[0] = math.sqrt(d_eta[0]*d_eta[0] + d_phi[0]*d_phi[0])

                    value = reader.EvaluateRegression(self.regression_method)[0]

                    if self.log_efrac:
                        value = math.exp(value)

                    eta_phi_efrac_dist.Fill(d_eta[0], d_phi[0], value)

        if eta_phi_efrac_dist.Integral() > 0:
            eta_phi_efrac_dist.Scale(1.0/eta_phi_efrac_dist.Integral())

        if randomize_pos:
            ext = '_rnd'

        self.plot_root_hist2d(eta_phi_efrac_dist,
                              path.join(self.base_output_folder, '%s%s.pdf' % (self.mlp_base_file_name, ext)),
                              '#eta', '#phi')
        self.difference_to_reference(reference, eta_phi_efrac_dist,
                                     path.join(self.base_output_folder, '%s%s_diff.pdf' %
                                               (self.mlp_base_file_name, ext)), '#eta', '#phi')

        self.difference_to_reference(reference, eta_phi_efrac_dist,
                                     path.join(self.base_output_folder, '%s%s_diff_inv.pdf' %
                                               (self.mlp_base_file_name, ext)), '#eta', '#phi', inverted_ratio=True)

        pass

    def plot_regression_control_plots(self, file_name=None, show_target=False, compare_to_train=False):
        """ This is a pythonic translation of the deviations function found in the TMVA examples.
        """
        import ROOT
        ROOT.gROOT.SetBatch(True)

        #ROOT.TMVA.TMVAGlob.Initialize(True)
        ROOT.gStyle.SetNumberContours(999)

        if not file_name:
            file_name = '%s.root' % self.mlp_base_file_name
        input_file = ROOT.TFile(file_name)

        current_canvas = 0
        for key in input_file.GetListOfKeys():
            print('Key: %s' % key.GetName())

            if not 'Method_' in key.GetName():
                continue
            if not ROOT.gROOT.GetClass(key.GetClassName()).InheritsFrom('TDirectory'):
                continue

            method_name = key.GetName().replace('Method_', '')

            print('Now plotting control plots for method %s' % method_name)

            directory = key.ReadObj()

            for sub_method in directory.GetListOfKeys():
                print('sub_method: %s' % sub_method.GetName())

                if not ROOT.gROOT.GetClass(sub_method.GetClassName()).InheritsFrom('TDirectory'):
                    continue

                sub_method_name = sub_method.GetName()

                print('Now plotting control plots for sub-method %s' % sub_method_name)

                sub_directory = sub_method.ReadObj()

                for plot_key in sub_directory.GetListOfKeys():
                    print('plot_key: %s' % plot_key.GetName())

                    plot_obj = plot_key.ReadObj()
                    if isinstance(plot_obj, ROOT.TH2F):
                        plot_name = plot_key.GetName()
                        if not '_reg_' in plot_name:
                            continue

                        if not ((show_target and '_tgt' in plot_name) or
                                (not show_target and (not '_tgt' in plot_name))):
                            continue

                        if not ((compare_to_train and 'train' in plot_name) or
                                (not compare_to_train and 'test' in plot_name)):
                            continue

                        inortarg = 'input'
                        type_string = 'test'
                        if show_target:
                            inortarg = 'target'

                        if compare_to_train:
                            type_string = 'training'

                        canvas = ROOT.TCanvas('canvas_%i' % current_canvas,
                                              'Regression output deviation versus %s for method %s' % (inortarg,
                                              method_name), 800, 600)

                        canvas.SetRightMargin(0.10)

                        plot_obj.SetTitle('Output deviation for method: %s (%s sample)' % (sub_method_name,
                                                                                           type_string))
                        plot_obj.Draw('colz')

                        line = ROOT.TLine(plot_obj.GetXaxis().GetXmin(), 0, plot_obj.GetXaxis().GetXmax(), 0)
                        line.SetLineStyle(2)
                        line.Draw()

                        canvas.Update()

                        import os
                        output_name = os.path.join(self.base_output_folder, '%s_dev_%s_%s_%s_%i.pdf' %
                                                   (self.mlp_base_file_name, method_name, inortarg, type_string,
                                                    current_canvas))
                        #ROOT.TMVAGlob.imgconv(canvas, output_name)
                        canvas.Print(output_name, 'pdf')

                        current_canvas += 1

        pass

    @staticmethod
    def plot_root_hist2d(histogram, file_name='root_hist2d.pdf', x_label='X-Axis', y_label='Y-Axis',
                         z_label='Probability', logscale=False, normalize=False, root_file=None):
        import ROOT
        ROOT.gROOT.SetBatch(True)
        ROOT.gStyle.SetOptStat(1)

        from os import path
        ext = path.splitext(path.basename(file_name))[1]

        canv = ROOT.TCanvas('canvas', 'test', 0, 0, 600, 600)
        #canv.SetTopMargin(0.05)
        #canv.SetLeftMargin(0.13)
        canv.SetRightMargin(0.15)
        #canv.SetBottomMargin(0.035)

        if logscale:
            canv.SetLogz()

        if normalize:
            integral = histogram.Integral()
            if integral > 0:
                histogram.Scale(1.0 / integral)

        histogram.GetXaxis().SetTitle(x_label)
        histogram.GetYaxis().SetTitle(y_label)
        histogram.GetZaxis().SetTitle(z_label)

        # histogram.GetYaxis().SetLabelSize(0.06)
        # histogram.GetYaxis().SetLabelOffset(0.0015)
        #
        # histogram.GetYaxis().SetTitleSize(0.06)
        # histogram.GetYaxis().SetTitleOffset(0.85)

        histogram.Draw('COLZ')

        canv.Update()
        canv.Print(file_name, ext)

        if root_file:
            histogram.SetDirectory(root_file)
            histogram.Write()

        pass

    """
    ========================================== Utility Functions ======================================================
    The following section will contain utility functions to ease some repetitive tasks. Those functions should be self-
    explanatory.
    ===================================================================================================================
    """

    def difference_to_reference(self, reference, histogram, output_name, x_label, y_label, inverted_ratio=False):
        if inverted_ratio:
            clone = reference.Clone(reference.GetName() + 'diff_to_ref_clone')
            base = histogram
        else:
            clone = histogram.Clone(histogram.GetName() + 'diff_to_ref_clone')
            base = reference

        clone.Add(base, -1.0)
        clone.Divide(base)

        if inverted_ratio:
            z_label = 'reference - output / output'
        else:
            z_label = 'output - reference / reference'

        self.plot_root_hist2d(clone, output_name, x_label, y_label, z_label, logscale=True)

        pass

    def get_layer_name(self, layer_id):
        if layer_id < 0 or layer_id >= len(self.layer_names):
            return 'Unknown'
        else:
            return self.layer_names[layer_id]

    @staticmethod
    def phi_mpi_pi(phi):
        import ROOT
        pi = ROOT.TMath.Pi()
        while phi >= pi:
            phi -= pi
        while phi < -pi:
            phi += pi
        return phi


def run_reg(neuron_type, num_neurons):
    plotter = ShowerShapeRegressor()

    plotter.cumulated_events_threshold = 1000

    plotter.neuron_type = neuron_type
    plotter.num_neurons = num_neurons

    plotter.log_efrac = True

    plotter.run()

    return


if __name__ == "__main__":

    #for nn in [5, 10, 15]:
        #for nt in ['tanh', 'radial', 'sigmoid']:
    run_reg('tanh', 5)

    import time
    from distutils import archive_util
    # make_zipfile automagically adds the .zip extension
    archive_util.make_zipfile('reg_out_%s' % (time.strftime("%y-%m-%d_%H-%M")),
                              'RegressionOutput')
    archive_util.make_zipfile('weights_%s' % (time.strftime("%y-%m-%d_%H-%M")),
                              'weights')
