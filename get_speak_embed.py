from encoder.params_model import model_embedding_size as speaker_embedding_size
from encoder import inference as encoder
from utils.argutils import print_args
from pathlib import Path
import numpy as np
import argparse
import librosa
import pickle
import torch
import sys

if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path, 
                        default="encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("--low_mem", action="store_true", help=\
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    parser.add_argument("-p", "--path_wav", type=Path, 
                        default="ex.wav",
                        help="wav file")
    parser.add_argument("-s", "--path_embeds", type=Path,
                        default="voice_embeds.pickle",
                        help="voice embeddings")
    args = parser.parse_args()
    print_args(args, parser)

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
    print("Preparing the encoder...")
    encoder.load_model(args.enc_model_fpath)

    ## Run a test
    print("Testing your configuration with small inputs.")
    # Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
    # sampling rate, which may differ.
    # If you're unfamiliar with digital audio, know that it is encoded as an array of floats
    # (or sometimes integers, but mostly floats in this projects) ranging from -1 to 1.
    # The sampling rate is the number of values (samples) recorded per second, it is set to
    # 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond
    # to an audio of 1 second.
    print("\tTesting the encoder...")
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))

    # Create a dummy embedding. You would normally use the embedding that encoder.embed_utterance
    # returns, but here we're going to make one ourselves just for the sake of showing that it's
    # possible.
    embed = np.random.rand(speaker_embedding_size)
    # Embeddings are L2-normalized (this isn't important here, but if you want to make your own
    # embeddings it will be).
    embed /= np.linalg.norm(embed)
    # The synthesizer can handle multiple inputs with batching. Let's create another embedding to
    # illustrate that
    embeds = [embed, np.zeros(speaker_embedding_size)]
    print("All test passed! Now we save your voice.\n\n")
        
    # Get the reference audio filepath
    #message = "Reference voice: enter an audio filepath of a voice to be cloned(Введите путь до клонируемого файла, например ex.wav) (mp3, " \
    #          "wav, m4a, flac, ...):\n"
    #in_fpath = Path(input(message).replace("\"", "").replace("\'", ""))
    in_fpath = args.path_wav
    
    ## Computing the embedding
    # First, we load the wav using the function that the speaker encoder provides. This is 
    # important: there is preprocessing that must be applied.
    
    # The following two methods are equivalent:
    # - Directly load from the filepath:
    preprocessed_wav = encoder.preprocess_wav(in_fpath)
    # - If the wav is already loaded:
    original_wav, sampling_rate = librosa.load(in_fpath)
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    print("Loaded file succesfully")
    
    # Then we derive the embedding. There are many functions and parameters that the 
    # speaker encoder interfaces. These are mostly for in-depth research. You will typically
    # only use this function (with its default parameters):
    embed = encoder.embed_utterance(preprocessed_wav)
    embeds = [embed]
    print("Created the embedding")
    
    # Save it on the disk
    embeds_path = args.path_embeds
    with open(embeds_path, 'wb') as f:
        pickle.dump(embeds, f)
    print("Voice embedding was save")