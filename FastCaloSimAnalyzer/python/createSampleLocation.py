from __future__ import print_function
from argparse import ArgumentParser
import os


def getArguments():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", nargs='+', help="Text files containing a list of DSIDs. Otherwise a list of all samples that exist in the EOS will be created")
    parser.add_argument("-d", "--directory", default="/eos/atlas/atlascerngroupdisk/proj-simul/InputSamplesSummer18Complete/", help="Directory where datasets where the samples exist.")
    parser.add_argument("-o", "--output", default="inputSampleList.txt", help="Name of the output text file")
    return parser.parse_args()


def getAllSamples(path):
    samples = []
    for root, dirs, files in os.walk(path, topdown=False):
           for name in files:
            label = root.split('.')[2]
            pid = label.split('_')[1]
            pid = pid.replace('pid', "")
            energy = label.split('_')[2]
            energy = energy.replace('E', "")
            etamin = label.split('_')[7]
            etamax = label.split('_')[8]
            zv = label.split('_')[10]
            file = os.path.join(root, name)
            file2 = os.path.join(root, '*root*')
            dsid = file.split('.')[1]
            label = "mc16_13TeV." + dsid+ '.'+ label
            map = (dsid, file2, label, pid, energy, etamin, etamax, zv)
            #print (map)
            samples.append(map)
    return samples

def main():
    options = getArguments()

    #if no input sample list provided list all the samples that exist in Eos
    if not options.input:
        print("Creating a list of all samples that are in Eos directory")
        samples = getAllSamples(options.directory)

        with open(options.output, 'w') as outFile:
            outFile.write('\n'.join('%s %s %s %s %s %s %s %s' % sample for sample in samples))
            print("Created :" + str(options.output))



if __name__ == '__main__':
    main()


