import torch, torchaudio
import torchaudio.transforms as T
import yaml
import pandas as pd
from TTS.api import TTS

def process(text):
    chunk_size = 15
    for i in range(0, len(text), chunk_size):
        yield " ".join(text[i:i+chunk_size])

if __name__ == "__main__":
    # setup
    synthesised_sr = 24000
    target_sr = 16000

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on {device}")

    # load df
    with open("config.yml") as f:
        config = yaml.safe_load(f)
    df = pd.read_csv(config["data_args"]["csv_fi"], 
                 usecols=["sample", "student", "task_id", "cefr_mean", "split", "transcript_normalized", "recording_path"])
    df = df.rename(columns={"transcript_normalized": "text"})
    df.insert(len(df.columns), "text_chunks", df.text.str.split().apply(lambda l: list(process(l))))

    # load tts model 
    tts_en= TTS(model_name='tts_models/en/ljspeech/tacotron2-DDC', progress_bar=False).to(device)

    # define resampler
    if synthesised_sr != target_sr:
        resampler = T.Resample(synthesised_sr, target_sr)
    
    # start 
    for i, row in df.iterrows():
        print(f"--------sample {i} -------")
        whole_utterance = []
        for text_chunk in row.text_chunks:
            text_chunk = text_chunk.replace("ä", "a").replace("ö", "o")
            utterance_chunk = tts_en.tts(text=text_chunk)
            whole_utterance += utterance_chunk
        
        whole_utterance = torch.Tensor(whole_utterance)
        whole_utterance = resampler(whole_utterance)
        
        path = f"./synthesised/synsample_{i}_opiskelija_{row.student}_teht_{row.task_id}.wav"
        torchaudio.save(path, whole_utterance.unsqueeze(dim=0), target_sr)
        print(f"--------sample {i} synthesis done -------")

