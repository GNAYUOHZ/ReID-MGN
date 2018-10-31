import argparse

parser = argparse.ArgumentParser(description='reid')

parser.add_argument('--data_path',
                    default="/path to/Market-1501-v15.09.15",
                    help='path of Market-1501-v15.09.15')

parser.add_argument('--mode',
                    default='train', choices=['train', 'evaluate'],
                    help='train or evaluate ')

parser.add_argument('--backbone',
                    default='resnet50',
                    choices=['resnet50', 'resnet101'],
                    help='load weights ')

parser.add_argument('--freeze',
                    default=False,
                    help='freeze backbone or not ')

parser.add_argument('--cpt',
                    default='weights/model_400.pt',
                    help='load weights ')

parser.add_argument('--epoch',
                    default=400,
                    help='number of epoch to train')

parser.add_argument('--lr',
                    default=2e-4,
                    help='learning_rate')

parser.add_argument('--lr_scheduler',
                    default=[320, 380],
                    help='MultiStepLR')

parser.add_argument("--batchid",
                    default=4,
                    help='the batch for id')

parser.add_argument("--batchimage",
                    default=4,
                    help='the batch of per id')

parser.add_argument("--batchtest",
                    default=8,
                    help='the batch size for test')

opt = parser.parse_args()