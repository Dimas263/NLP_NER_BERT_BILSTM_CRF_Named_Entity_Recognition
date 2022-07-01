import argparse


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):
        # args for path
        parser.add_argument('--output_dir', default='./drive/MyDrive/bert_bilstm_crf_named_entity_recognition/BERT-BILSTM-CRF-NER/output/checkpoints',
                            help='the output dir for model checkpoints')

        parser.add_argument('--bert_dir', default='./drive/MyDrive/bert_bilstm_crf_named_entity_recognition/BERT-BILSTM-CRF-NER/model/biobert',
                            help='bert dir for uer')
        parser.add_argument('--data_dir', default='./drive/MyDrive/bert_bilstm_crf_named_entity_recognition/BERT-BILSTM-CRF-NER/input/',
                            help='data dir for uer')
        parser.add_argument('--log_dir', default='./drive/MyDrive/bert_bilstm_crf_named_entity_recognition/BERT-BILSTM-CRF-NER/logs/',
                            help='log dir for uer')

        # other args
        parser.add_argument('--bio_tags', default=53, type=int,
                            help='number of tags')
        parser.add_argument('--att_tags', default=53, type=int,
                            help='number of tags')
        parser.add_argument('--seed', type=int, default=123, help='random seed')

        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')

        parser.add_argument('--max_seq_len', default=256, type=int)

        parser.add_argument('--eval_batch_size', default=12, type=int)

        parser.add_argument('--swa_start', default=3, type=int,
                            help='the epoch when swa start')

        # train args
        parser.add_argument('--train_epochs', default=15, type=int,
                            help='Max training epoch')

        parser.add_argument('--dropout_prob', default=0.1, type=float,
                            help='drop out probability')

        # 2e-5
        parser.add_argument('--lr', default=3e-5, type=float,
                            help='bert learning rate')
        # 2e-3
        parser.add_argument('--other_lr', default=3e-4, type=float,
                            help='bilstm and multilayer perceptron learning rate')
        parser.add_argument('--crf_lr', default=3e-2, type=float,
                            help='crf learning rate')
        # 0.5
        parser.add_argument('--max_grad_norm', default=1, type=float,
                            help='max grad clip')

        parser.add_argument('--warmup_proportion', default=0.1, type=float)

        parser.add_argument('--weight_decay', default=0.01, type=float)

        parser.add_argument('--adam_epsilon', default=1e-8, type=float)

        parser.add_argument('--train_batch_size', default=32, type=int)
        parser.add_argument('--use_lstm', type=str, default='True',
                            help='Whether to use BiLstm')
        parser.add_argument('--lstm_hidden', default=128, type=int,
                            help='lstm hidden layer size')
        parser.add_argument('--num_layers', default=1, type=int,
                            help='lstm layer size')
        parser.add_argument('--dropout', default=0.3, type=float,
                            help='Settings for dropout in lstm')
        parser.add_argument('--use_crf', type=str, default='True',
                            help='Whether to use Crf')

        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()

