from utils.ngram_utils import *

if __name__ == "__main__":
    input_txt = sys.argv[1]
    output_hdf5 = sys.argv[2]
    build_predictions(input_txt, output_hdf5)