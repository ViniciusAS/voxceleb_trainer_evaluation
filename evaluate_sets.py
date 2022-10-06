import os
from glob import glob
import time
import warnings
from pprint import pprint

import torch

from DatasetLoader import train_dataset_loader, train_dataset_sampler, worker_init_fn
from SpeakerNet import SpeakerNet, WrappedModel, ModelTrainer
from tuneThreshold import ComputeErrorRates, ComputeMinDcf, tuneThresholdfromScore
import trainSpeakerNet

warnings.simplefilter("ignore")

base_dataset_dir = '/media/viniciusas/ExtremeSSD/Datasets'
datasets = [
    #'simulated-noise-w-reverb',
    #'random-noises',
    'VoxCeleb1',
    # ## Random 1 ##
    # 'random-simulated',
    # 'random-lms-a1',
    # 'random-nlms-a1',
    # 'random-klms-a1',
    # 'random-nklms-a1',
    # ## Random 2 ##
    # 'random2-simulated',
    # 'random2-noises',
    # 'random2-nlms-a1',
    # 'random2-nklms-a1',
    # 'random2-lms-a1',
    # 'random2-klms-a1',
]
models = [
    (
        'ResNetSE34L',
        dict(
            model='ResNetSE34L',
            trainfunc='angleproto',
            initial_model='baseline_lite_ap.model',
            eval_frames=400,
        )
    ),
    (
        'ResNetSE34V2',
        dict(
            model='ResNetSE34V2',
            encoder_type='ASP',
            trainfunc='softmaxproto',
            initial_model='baseline_v2_smproto.model',
            eval_frames=400,
            n_mels=64,
        ),
    ),
    (
        'RawNet3',
        dict(
            model='RawNet3',
            initial_model='RawNet3/model.pt',
            encoder_type='ECA',
            save_path='exps/RawNet3_AAM',
            nOut=256,
            sinc_stride=10,
            stride=10,
            max_frames=200,
            eval_frames=400,
            trainfunc='aamsoftmax',
            batch_size=32,
            lr_decay=0.75,
            weight_decay=5e-05,
            max_seg_per_spk=500,
            augment=True,
        ),
    ),
]

for model_name, model_params in models[2:]:
    print("Testing model", model_name)

    args = trainSpeakerNet.parser.parse_args()
    for key, val in model_params.items():
        args.__dict__[key] = val
    args.gpu = 0
    args.log_input = True
    args.nDataLoaderThread = 10

    for dataset in datasets:
        args.test_path = f'{base_dataset_dir}/{dataset}/test/wav'
        args.test_list = f'{base_dataset_dir}/VoxCeleb1/voxceleb1_test_v2.txt'
        print("Testing dataset", dataset)

        # load weights
        speaker_net = SpeakerNet(**vars(args))
        speaker_net = WrappedModel(speaker_net).cuda(0)
        trainer = ModelTrainer(speaker_net, **vars(args))
        trainer.loadParameters(args.initial_model)
        sc, lab, _ = trainer.evaluateFromList(**vars(args))

        result = tuneThresholdfromScore(sc, lab, [1, 0.1])

        fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
        mindcf, threshold = ComputeMinDcf(
            fnrs, fprs, thresholds,
            args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa,
        )

        print("\n")
        print("############################")
        print(f"Model {model_name} with dataset {dataset}")
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        print("VEER {:2.4f}".format(result[1]))
        print("MinDCF {:2.5f}".format(mindcf))
        print("############################")
        print()

