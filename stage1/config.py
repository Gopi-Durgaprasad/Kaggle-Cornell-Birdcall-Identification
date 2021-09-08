
class args:

    DEBUG = False

    mixup = False

    new_mixup = True

    alpha = 1

    exp_name = "stage1_Cnn14_16k_valid"
    output_dir = "drive/My Drive/Cornell Birdcall Identification/weights"
    train_csv = "drive/My Drive/Cornell Birdcall Identification/input/train_stage1_df.csv"
    valid_csv = "drive/My Drive/Cornell Birdcall Identification/input/valid_stage1_df.csv"
    pretrained_path = False #"Cnn14_16k_mAP=0.438.pth?download=1"

    network = "Cnn14_16k" #"Cnn14_16k" #"BirdClassifier"
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
    seed = 1001
    epochs = 150
    batch_size = 32 * 4
    num_workers = 4
    start_epoch = 0

    warmup_epo = 2
    cosine_epo = 8
    init_lr = 3e-5

    load_from = True
