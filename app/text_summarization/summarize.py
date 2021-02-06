
from deepsegment import DeepSegment
from app.text_summarization.models import data_loader
from app.text_summarization.others.logging import init_logger, logger

import argparse
import os
import torch 
import torch
from pytorch_transformers import BertTokenizer

from app.text_summarization.models.model_builder import AbsSummarizer
from app.text_summarization.models.predictor import build_predictor


model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def get_segmenter():
    segmenter = DeepSegment('en')
    return segmenter

def get_summary(args, source_string, device, predictor):
    test_iter = data_loader.load_text_from_string(args, source_string, device)    
    summarized_string, source_string = predictor.translate(test_iter, -1)
    
    return summarized_string, source_string 


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    config = {}
    config['task'] = 'abs'
    config['encoder'] = 'bert'
    config['mode'] = 'test_text'
    config['bert_data_path'] = '/home/daniel/NLP/PreSumm/bert_data_new/cnndm'
    config['model_path'] = '/home/daniel/NLP/PreSumm/models/'  
    config['result_path'] = '/home/daniel/NLP/PreSumm/results/cnndm'
    config['temp_dir'] = '/home/daniel/NLP/PreSumm/temp'
    config['text_src'] = ''
    config['text_tgt'] = ''
    config['batch_size'] = 140
    config['test_batch_size'] = 200
    config['max_ndocs_in_batch'] = 6
    config['max_pos'] = 512
    config['use_interval'] = True
    config['large'] = False
    config['load_from_extractive'] = ''
    config['sep_optim'] = False
    config['lr_bert'] = 2e-3
    config['lr_dec'] = 2e-3
    config['use_bert_emb'] = False
    config['share_emb'] = False
    config['finetune_bert'] = True
    config['dec_dropout'] = 0.2
    config['dec_layers'] = 6
    config['dec_hidden_size'] = 768
    config['dec_heads'] = 8
    config['dec_ff_size'] = 2048
    config['enc_hidden_size'] = 512
    config['enc_ff_size'] = 512
    config['enc_dropout'] = 0.2
    config['enc_layers'] = 6



    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)



    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)


    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='/home/daniel/NLP/PreSumm//logs/cnndm.log')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='/home/daniel/NLP/PreSumm/models/model_step_148000.pt')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)

    return args


def initialize(args):
    logger.info('Loading checkpoint from %s' % args.test_from)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"

    checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])

    model = AbsSummarizer(args, device, checkpoint)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    return predictor
  