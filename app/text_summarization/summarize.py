
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
    config['label_smoothing'] = 0.1
    config['generator_shard_size'] = 32
    config['alpha'] = 0.6
    config['beam_size'] = 5
    config['min_length'] = 15
    config['max_length'] = 150
    config['max_tgt_len'] = 140
    config['param_init'] = 0
    config['param_init_glorot'] = True
    config['optim'] = 'adam'
    config['lr'] = 1
    config['beta1'] = 0.9
    config['beta2'] = 0.999
    config['warmup_steps'] = 8000
    config['warmup_steps_bert'] = 8000
    config['warmup_steps_dec'] = 8000
    config['max_grad_norm'] = 0
    config['save_checkpoint_steps'] = 5
    config['accum_count'] = 1
    config['report_every'] = 1
    config['train_steps'] = 1000
    config['recall_eval'] = False
    config['visible_gpus'] = '-1'
    config['gpu_ranks'] = [0]
    config['log_file'] = '/home/daniel/NLP/PreSumm//logs/cnndm.log'
    config['seed'] = 666
    config['test_all'] = False
    config['test_from'] = '/home/daniel/NLP/PreSumm/models/model_step_148000.pt'
    config['test_start_from'] = -1
    config['train_from'] = ''
    config['report_rouge'] = True
    config['block_trigram'] = True
    config['world_size'] = 1

    return config


def initialize(args):
    logger.info('Loading checkpoint from %s' % args['test_from'])
    device = "cpu" if args['visible_gpus'] == '-1' else "cuda"

    checkpoint = torch.load(args['test_from'], map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            args[k] = opt[k]
            
    model = AbsSummarizer(args, device, checkpoint)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args['temp_dir'])
    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    return predictor
  