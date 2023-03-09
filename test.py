from calendar import timegm
from lib2to3.pgen2 import token
import numpy as np
import soundfile as sf
import yaml

import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import AutoConfig
from tokenizer import tokenize_sentence
# initialize fastspeech2 model.
#fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")
#tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en", name="tacotron2")
# TensorFlowTTS/examples/tacotron2/exp/train.tacotron2.v1
tacotron2_config = AutoConfig.from_pretrained("./test_model/tacotron2.v1.yaml")   
tacotron2 = TFAutoModel.from_pretrained("./test_model/model-120000.h5",tacotron2_config )

# initialize mb_melgan model
mb_melgan_config = AutoConfig.from_pretrained("./tts-mb_melgan-ljspeech-en/config.yml")   
mb_melgan = TFAutoModel.from_pretrained("./tts-mb_melgan-ljspeech-en/model.h5",mb_melgan_config)


# inference
#processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")
processor = AutoProcessor.from_pretrained("./examples/tacotron2/exp/train.tacotron2.v1/processor.json")

#input_ids = processor.text_to_sequence("Recently, there is an increasing interest in learning analytics in Technology Enhanced Learning.Generally, learning analytics deals with the development of methods that harness educational data sets to support the learning process. Learning analytics is a multi-disciplinary field involving machine learning, artificial intelligence, information retrieval, statistics, and visualization.")
#input_ids = processor.text_to_sequence("Recent research at Harvard has shown meditating for as little as 8 weeks, can actually increase the grey matter in the parts of the brain responsible for emotional regulation, and learning.")
#input_ids = processor.text_to_sequence("I like it. Generally, learning analytics deals with the development of methods that harness educational data sets to support the learning process.")
# fastspeech inference
text = "Learning analytics is a multi-disciplinary field involving machine learning, artificial intelligence, information retrieval, statistics, and visualization. Recent research at Harvard has shown meditating for as little as 8 weeks, can actually increase the grey matter in the parts of the brain responsible for emotional regulation, and learning."

# mel_before, mel_after, duration_outputs, _, _ = tacotron2.inference(
#     input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
#     speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
#     speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
#     f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
#     energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
# )

def sythenize(text):
    input_ids = processor.text_to_sequence(text)

    decoder_output, mel_outputs, stop_token_prediction, alignment_history = tacotron2.inference(
        input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
    )
    audio = mb_melgan.inference(mel_outputs)[0, :, 0]
    return audio

import numpy as np
import time

def load_pause(filename):
    y,sr = sf.read(filename)
    return y

def sythnize_full_text(text):
    full_audio = np.array([])
    sents = tokenize_sentence(text)
    #pause = load_pause("short pause.wav")
    for sent in sents:
        audio = sythenize(sent)
        print(audio.numpy().shape)
        full_audio = np.append(full_audio,audio.numpy())
        # # add pause
        # full_audio = np.append(full_audio,pause)
    return full_audio
# # melgan inference
# audio_before = mb_melgan.inference(mel_before)[0, :, 0]
# audio_after = mb_melgan.inference(mel_after)[0, :, 0]

# # save to file
# sf.write('./audio_before 2.wav', audio_before, 22050, "PCM_16")
# sf.write('./audio_after 2.wav', audio_after, 22050, "PCM_16")

full_a = sythnize_full_text(text)
sf.write('/home/ludwig/share/audio-120000-'+time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())+'.wav', full_a, 22050, "PCM_16")
