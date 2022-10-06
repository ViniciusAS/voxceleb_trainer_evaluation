## Models Evaluation
### Declare base vars
```bash
dataset_dir=/media/viniciusas/ExtremeSSD/Datasets
paths="
    --test_path $dataset_dir/VoxCeleb1/test/wav
    --test_list $dataset_dir/VoxCeleb1/veri_test2.txt
    --train_list $dataset_dir/VoxCeleb1/list_test_all.txt
    --musan_path $dataset_dir/musan
"
base_1="
    --eval
    --log_input True
    --eval_frames 400  
"
# --test_list $dataset_dir/VoxCeleb1/voxceleb1_test_v2.txt
# --save_path exps/test
```

###  Chung et. al. 2020: In defence of metric learning for speaker recognition
```bash
python trainSpeakerNet.py $paths $base_1 \
    --model ResNetSE34L \
    --trainfunc angleproto \
    --initial_model baseline_lite_ap.model
```

###  Kwon et. al. 2021: The ins and outs of speaker recognition: lessons from VoxSRC 2020
```bash
python trainSpeakerNet.py $paths $base_1 \
    --model ResNetSE34V2 \
    --encoder_type ASP \
    --n_mels 64 \
    --trainfunc softmaxproto \
    --initial_model baseline_v2_smproto.model
```

###  Jung et. al. 2022: Pushing the limits of raw waveform speaker recognition
```bash
python trainSpeakerNet.py $paths --eval \
  --config ./configs/RawNet3_AAM.yaml \
  --initial_model RawNet3/model.pt
```
