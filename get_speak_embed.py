from encoder import inference as encoder
from utils.argutils import print_args
from pathlib import Path
import pickle
import argparse
import librosa

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
    args = parser.parse_args()
    print_args(args, parser)
        
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
    with open('voice_embeds.pickle', 'wb') as f:
        pickle.dump(embeds, f)
    print("Voice embedding was save")