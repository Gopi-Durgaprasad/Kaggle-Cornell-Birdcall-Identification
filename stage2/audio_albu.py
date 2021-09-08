import audiomentations as A

augmenter = A.Compose([
    A.AddGaussianNoise(p=0.3),
    A.AddGaussianSNR(p=0.3),
    #A.AddBackgroundNoise("../input/train_audio/", p=1)
    #A.AddImpulseResponse(p=1),
    #A.AddShortNoises("../input/train_audio/", p=1)
    A.FrequencyMask(min_frequency_band=0.0,  max_frequency_band=0.2, p=0.2),
    A.TimeMask(min_band_part=0.0, max_band_part=0.2, p=0.2),
    A.PitchShift(min_semitones=-0.5, max_semitones=0.5, p=0.3),
    A.Shift(p=0.3),
    A.Normalize(p=0.3),
    A.ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=1, p=0.3),
    A.PolarityInversion(p=0.3),
    A.Gain(p=0.3)
])