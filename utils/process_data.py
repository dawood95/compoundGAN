import argparse
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, required=True)
args = parser.parse_args()

outfile = args.filename.split('.')[0]+'_stat.txt'

with open(args.filename, 'r') as f:
    for line in f:
        line = line.strip()
        data = line.split(',')
        expected_val = data[0]
        experimental_data = list(filter(None, data[1:]))
        experimental_data = [float(elem) for elem in experimental_data]
        mean = statistics.mean(experimental_data)
        std_dev = statistics.pstdev(experimental_data)
        out = open(outfile, 'a')
        out.write("Value: " + expected_val + ", Mean: " + str(mean) + ", StdDev: " + str(std_dev) + "\n")
        out.close()

f.close()
