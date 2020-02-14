import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle
from utils import Hps
from preprocess.tacotron.norm_utils import spectrogram2wav, get_spectrograms
from scipy.io.wavfile import write
import glob
import os
import argparse
from solver import Solver

def find_test_case(s, t, sl, tl):
    if s:
        if t:
            return 'OvO'
        else:
            return 'OvA'
    else:
        if t:
            return 'AvO'
        else:
            return 'AvA'

def one_vs_one(source, target, output):
    _, spec = get_spectrograms(source)
    spec_expand = np.expand_dims(spec, axis=0)
    spec_tensor = torch.from_numpy(spec_expand).type(torch.FloatTensor)
    c = Variable(torch.from_numpy(np.array([int(target)]))).cuda()
    result = solver.test_step(spec_tensor, c, gen=args.use_gen)
    result = result.squeeze(axis=0).transpose((1, 0))
    wav_data = spectrogram2wav(result)
    write(output, rate=args.sample_rate, data=wav_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hps', help='The path of hyper-parameter set', default='vctk.json')
    parser.add_argument('-model', '-m', help='The path of model checkpoint')
    parser.add_argument('-source', '-s', help='The path of source .wav file')
    parser.add_argument('-target', '-t', help='Target speaker id (integer). Same order as the speaker list when preprocessing (en_speaker_used.txt)')
    parser.add_argument('-output', '-o', help='output .wav path')
    parser.add_argument('-sample_rate', '-sr', default=16000, type=int)
    parser.add_argument('--use_gen', default=True, action='store_true')
    parser.add_argument('-source_list','-sl', help='Path of a file with a list of source audios')
    parser.add_argument('-target_list', '-tl', help='Path of a file with a list of target speakers')

    args = parser.parse_args()

    case = find_test_case(args.source, args.target, args.source_list, args.target_list)
    print(case)

    switch


    # hps = Hps()
    # hps.load(args.hps)
    # hps_tuple = hps.get_tuple()
    # solver = Solver(hps_tuple, None)
    # solver.load_model(args.model)


    # with open(args.source_list, 'r') as f:
    #     source_list = f.read().splitlines() 

    # with open(args.target_list, 'r') as f:
    #     target_list = f.read().splitlines() 



