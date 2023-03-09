import time
import socket
import uuid
import tensorflow_tts
import sys
sys.path.append("/home/ludwig/projects/FInal Project/Tacotron Server/sythensis")
from sythensis import Sythensis
import soundfile as sf
# text = "Learning analytics is a multi-disciplinary field involving machine learning, artificial intelligence, information retrieval, statistics, and visualization."
# input_ids = processor.text_to_sequence(text)

# decoder_output, mel_outputs, stop_token_prediction, alignment_history = tacotron2.inference(
#     input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
#     input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
#     speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
# )
# audio = mb_melgan.inference(mel_outputs)[0, :, 0]
# sf.write('./audio 10-24.wav', audio, 22050, "PCM_16")
# def generate_speech_and_write_to_wav_file(text):
#     input_ids = processor.text_to_sequence(text)

#     decoder_output, mel_outputs, stop_token_prediction, alignment_history = tacotron2.inference(
#         input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
#         input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
#         speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
#     )
#     audio = mb_melgan.inference(mel_outputs)[0, :, 0]
#     filename = str(uuid.uuid1())+".wav"
#     sf.write('../audio_output/'+filename, audio, 22050, "PCM_16")
    
#     return filename

def generate_speech_and_write_to_wav_file(text):
    audio = tts_synthesis.sythnize(text)
    filename = str(uuid.uuid1())+".wav"
    sf.write('../audio_output/'+filename, audio, 22050, "PCM_16")
    
    return filename

host = ''
port = 10888

udp_server = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
udp_server.bind((host,port))
tts_synthesis = Sythensis()

print("start TTS server")
while True:
    data,address = udp_server.recvfrom(4096)
    print("get data from",address)
    filename = generate_speech_and_write_to_wav_file(data.decode())
    output_speech_url = "/api/download/tts_output/"+filename
    udp_server.sendto(output_speech_url.encode(),address)
