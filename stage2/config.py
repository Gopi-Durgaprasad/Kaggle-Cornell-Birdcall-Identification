%%writefile config.py

class args:

    DEBUG = False

    mixup = True

    new_mixup = False

    alpha = 1

    exp_name = "stage1_Cnn14_DecisionLevelAtt_1sec_aug"
    output_dir = "drive/My Drive/Cornell Birdcall Identification/weights"
    train_csv = "drive/My Drive/Cornell Birdcall Identification/input/stage1_sudo.csv"
    valid_csv = "drive/My Drive/Cornell Birdcall Identification/input/valid_stage1_df3.csv"
    pretrained_path = "Cnn14_DecisionLevelAtt_mAP=0.425.pth?download=1" #False #"Cnn14_16k_mAP=0.438.pth?download=1"

    network = "Cnn14_DecisionLevelAtt" #"Cnn14_16k" #"BirdClassifier"
    encoder = "resnest50d"

    losses = "PANNsLoss"

    mel_param = {
        #"hop_lenght" : 345 * 2,
        "fmin" : 20,
        "fmax" : 16000 // 2,
        "n_mels" : 128,
        "n_fft" : 128 * 20
    }

    model_config = {
        #"encoder" : "resnest50d",
        "sample_rate": 16000,
        "window_size": 512,
        "hop_size": 160,
        "mel_bins": 64,
        "fmin": 50,
        "fmax": 8000,
        "classes_num": 264 #209 #54 #264
    }

    PERIOD = 1
    
    device = "cuda"
    seed = 42
    epochs = 50
    batch_size = 32 * 4
    num_workers = 4
    start_epoch = 0

    warmup_epo = 2
    cosine_epo = 8
    init_lr = 3e-5

    load_from = False
