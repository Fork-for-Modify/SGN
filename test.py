from __future__ import print_function
import argparse
import json
import os
import torch

from utils import build_loaders, build_model, test, score
from config import Config as C, MSRVTTLoaderConfig, MSVDLoaderConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", type=str, default='train')
    parser.add_argument("--corpus", type=str)
    parser.add_argument("--ckpt_fpath", type=str)
    parser.add_argument("--save_dir", type=str)
    return parser.parse_args()


def run(corpus, ckpt_fpath, save_dir, run_mode='train'):
    C.run_mode = run_mode
    C.corpus = corpus
    if corpus == 'MSVD':
        C.loader = MSVDLoaderConfig
    elif corpus == 'MSR-VTT':
        C.loader = MSRVTTLoaderConfig
    else:
        raise NotImplementedError('Unknown corpus: {}'.format(corpus))

    train_iter, val_iter, test_iter, vocab = build_loaders(C)

    model = build_model(C, vocab)
    model.load_state_dict(torch.load(ckpt_fpath))
    model.cuda()
    model.eval()

    if run_mode=='test':
        hypos, refs, vid2idx = test(model, test_iter, vocab)
    elif run_mode=='val':
        scores, refs, hypos, vid2idx = score(model, test_iter, vocab)
        print('\n',scores)
    
    # save result
    # print(hypos)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = save_dir+'hypos.json'
    with open(save_path,'w') as f:
        json.dump(hypos,f)
        

if __name__ == '__main__':
    args = parse_args()
    run(args.corpus, args.ckpt_fpath, args.save_dir, args.run_mode)

