import numpy as np
import time


#udp_server.listen(0)

import numpy as np
import soundfile as sf
import yaml

import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import AutoConfig

from tokenizer import tokenize_sentence
import os
class Sythensis:
    def __init__(self):
        # initialize fastspeech2 model.
        print(os.path.dirname("."))
        curr_dir = os.path.dirname(__file__)
        print(os.path.dirname(__file__))
        print(os.path.isfile("./models/tts-tacotron2-ljspeech-en/config.yml"))
        tacotron2_config = AutoConfig.from_pretrained(curr_dir+"/models/tts-tacotron2-ljspeech-en/config.yml")   
        self.tacotron2 = TFAutoModel.from_pretrained(curr_dir+"/models/tts-tacotron2-ljspeech-en/model.h5",tacotron2_config )

        # initialize mb_melgan model
        mb_melgan_config = AutoConfig.from_pretrained(curr_dir+"/models/tts-mb_melgan-ljspeech-en/config.yml")   
        self.mb_melgan = TFAutoModel.from_pretrained(curr_dir+"/models/tts-mb_melgan-ljspeech-en/model.h5",mb_melgan_config)


        # inference
        self.processor = AutoProcessor.from_pretrained(curr_dir+"/models/tts-tacotron2-ljspeech-en/processor.json")
    def __sythenize__(self,text):
        input_ids = self.processor.text_to_sequence(text)

        decoder_output, mel_outputs, stop_token_prediction, alignment_history = self.tacotron2.inference(
            input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        )
        audio = self.mb_melgan.inference(mel_outputs)[0, :, 0]
        return audio


    def __load_pause__(self,filename):
        y,sr = sf.read(filename)
        return y

    def sythnize(self,text):
        full_audio = np.array([])
        sents = tokenize_sentence(text)
        curr_dir = os.path.dirname(__file__)
        pause = self.__load_pause__(curr_dir+"/short pause.wav")
        for sent in sents:
            audio = self.__sythenize__(sent)
            print(audio.numpy().shape)
            full_audio = np.append(full_audio,audio.numpy())
            # add pause
            full_audio = np.append(full_audio,pause)
        return full_audio

if __name__ == "__main__":
    tts_synthesis = Sythensis()
