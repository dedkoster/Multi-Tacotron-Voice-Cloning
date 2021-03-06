from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from utils.argutils import print_args
from g2p.train import g2p
from pathlib import Path
import numpy as np
import librosa
import argparse
import torch
import sys
import pickle

if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path, 
                        default="encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_dir", type=Path, 
                        default="synthesizer/saved_models/logs-pretrained/",
                        help="Directory containing the synthesizer model")
    parser.add_argument("-v", "--voc_model_fpath", type=Path, 
                        default="vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--low_mem", action="store_true", help=\
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("-t", "--text", 
                        default="Hello my friends. Я многоязычный синтез построенный на tacotron. Шла саша по шоссе и сосала сушку",
                        help="Text") 
    parser.add_argument("-p", "--path_embeds", type=Path, 
                        default="voice_embeds.pickle",
                        help="voice embeddings")                           
    args = parser.parse_args()
    print_args(args, parser)
    if not args.no_sound:
        import sounddevice as sd
        
    
    ## Print some environment information (for debugging purposes)
    print("Running a test of your configuration...\n")
    if not torch.cuda.is_available():
        print("Your PyTorch installation is not configured to use CUDA. If you have a GPU ready "
              "for deep learning, ensure that the drivers are properly installed, and that your "
              "CUDA version matches your PyTorch installation. CPU-only inference is currently "
              "not supported.", file=sys.stderr)
        quit(-1)
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
          "%.1fGb total memory.\n" % 
          (torch.cuda.device_count(),
           device_id,
           gpu_properties.name,
           gpu_properties.major,
           gpu_properties.minor,
           gpu_properties.total_memory / 1e9))
    
    
    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=args.low_mem)
    vocoder.load_model(args.voc_model_fpath)

    ## Interactive speech generation
    print("This is a GUI-less example of interface to SV2TTS. The purpose of this script is to "
          "show how you can interface this project easily with your own. See the source code for "
          "an explanation of what is happening.\n")
    
    print("Interactive generation loop")
    num_generated = 0

    # Get the reference audio filepath
    #message = "Reference voice: enter an audio filepath of a voice to be cloned(Введите путь до клонируемого файла, например ex.wav) (mp3, " \
    #          "wav, m4a, flac, ...):\n"
    #in_fpath = Path(input(message).replace("\"", "").replace("\'", ""))
    in_fpath = args.path_embeds

    with open(in_fpath, 'rb') as f:
        embeds = pickle.load(f)

    
    ## Generating the spectrogram
    # text = input("Write a sentence (+-20 words) to be synthesized:(Введите предложение для синтеза)\n")
    
    # The synthesizer works in batch, so you need to put your data in a list or numpy array
    texts = [args.text]
    texts = g2p(texts)
    print(texts)
    # If you know what the attention layer alignments are, you can retrieve them here by
    # passing return_alignments=True
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    print("Created the mel spectrogram")
    
    
    ## Generating the waveform
    print("Synthesizing the waveform:")
    # Synthesizing the waveform is fairly straightforward. Remember that the longer the
    # spectrogram, the more time-efficient the vocoder.
    generated_wav = vocoder.infer_waveform(spec)
    
    
    ## Post-generation
    # There's a bug with sounddevice that makes the audio cut one second earlier, so we
    # pad it.
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    
    # Play the audio (non-blocking)
    if not args.no_sound:
        sd.stop()
        sd.play(generated_wav, synthesizer.sample_rate)
        
    # Save it on the disk
    fpath = "demo_output_%02d.wav" % num_generated
    print(generated_wav.dtype)
    librosa.output.write_wav(fpath, generated_wav.astype(np.float32), 
                             synthesizer.sample_rate)
    num_generated += 1
    print("\nSaved output as %s\n\n" % fpath)