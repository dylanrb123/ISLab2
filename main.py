#!/usr/bin/python3

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import sys
import glob
import time
import json
import base64


def main(arg):
    if arg == "feature_extraction":
        extract_mfcc()


def extract_mfcc():
    """
    Extracts the MFCC feature from each one second interval of each file in the raw_data folder (training data)
    Saves results in a JSON file called 'extracted_features.json'. The format is as follows:
    {
        in_filename:
        [
            {file_segment_name: [numpy_data_type, base64_encoded_mfcc_data, numpy_array_shape]},
            ...
        ],
        ...
    }
    """
    start_time = time.time()
    json_dict = {}
    for filename in glob.iglob('raw_data/*.wav'):
        sample_rate, data = wav.read(filename)

        # split files into separate 1 second clips
        name_counter = 1
        filename_parts = filename.split('.')
        filename_path = filename_parts[0].split('/')
        examples = []
        for i in range(0, len(data), sample_rate):
            new_data = data[i:i+sample_rate]
            new_filename = filename_path[1] + "_" + str(name_counter) + '.' + filename_parts[1]
            mfcc_feat = mfcc(new_data, sample_rate)
            numpy_list = [str(mfcc_feat.dtype), base64.b64encode(mfcc_feat).decode("utf-8"), mfcc_feat.shape]
            examples.append({new_filename: numpy_list})
            name_counter += 1
        json_dict[filename] = examples

    with open('extracted_features.json', 'w') as outfile:
        json.dump(json_dict, outfile)
    end_time = time.time()
    print("runtime: ", end_time - start_time)

if __name__ == "__main__":
    main(sys.argv[1])

