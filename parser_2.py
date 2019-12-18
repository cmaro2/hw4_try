from __future__ import absolute_import
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV TA\'s tutorial in image classification using pytorch')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='hw3_data',
                    help="root path to data directory")
    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")

    # training parameters
    parser.add_argument('--gpu', default=0, type=int,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--batch_size', default=16, type=int,
                    help="train batch size")
    parser.add_argument('--ngpu', default=1, type=int,
                        help="number of gpus")

    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--fullvid_dir', type=str, default='hw4_data/FullLengthVideos/videos/valid')
    parser.add_argument('--random_seed', type=int, default=999)
    parser.add_argument('--dir_vid', type=str, default='hw4_data/TrimmedVideos/video/valid')
    parser.add_argument('--dir_lab', type=str, default='hw4_data/TrimmedVideos/label/gt_valid.csv')

    args = parser.parse_args()

    return args
