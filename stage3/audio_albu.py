import audiomentations as A
import pandas as pd

#noise_path = pd.read_csv("drive/My Drive/Cornell Birdcall Identification/input/noise_df.csv")

augmenter = A.Compose([
    A.AddGaussianNoise(p=0.2),
    A.AddGaussianSNR(p=0.2),
    A.AddBackgroundNoise("/content/train_stage1_1sec_sudo_noise/tmp/birdsongWav/train_stage1_1sec_sudo_noise/", p=0.5),
    #A.AddImpulseResponse(p=1),
    #A.AddShortNoises("../input/train_audio/", p=1)
    #A.FrequencyMask(min_frequency_band=0.0,  max_frequency_band=0.2, p=0.2),
    #A.TimeMask(min_band_part=0.0, max_band_part=0.2, p=0.2),
    #A.PitchShift(min_semitones=-0.5, max_semitones=0.5, p=0.2),
    #A.Shift(p=0.2),
    A.Normalize(p=0.2),
    #A.ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=1, p=0.2),
    #A.PolarityInversion(p=0.2),
    A.Gain(p=0.2)
])