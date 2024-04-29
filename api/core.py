from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import shutil
import time
import os
from .config import CHECKPOINT_DIR, VOICE_DIR
from .utils import file_sha256_hash, spacer, split_on_pause, string_to_sha256
import torch
import torchaudio
import json
from VoiceCraft.models.voicecraft import VoiceCraft
from VoiceCraft.data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_text,
)


device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Voice:
    id: str
    name: str
    directory: str
    transcript: str

    @staticmethod
    def from_sample(name: str, audio_path: str, transcript: str) -> "Voice":
        dst = os.path.join(VOICE_DIR, file_sha256_hash(audio_path))
        os.makedirs(dst, exist_ok=True)

        shutil.copy(audio_path, os.path.join(dst, "source.wav"))

        with open(os.path.join(dst, "source.txt"), "w") as f:
            f.write(transcript)

        voice = Voice(
            id=file_sha256_hash(audio_path),
            name=name,
            directory=dst,
            transcript=transcript,
        )

        with open(os.path.join(dst, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(voice), f, indent=4)

        return voice

    def describe(self):
        return {
            "name": self.name,
            "voice_id": self.id,
            "preview_url": self.get_sample_path(),
        }

    def get_sample_path(self):
        return os.path.join(self.directory, "source.wav")

    @staticmethod
    def voice_config_paths():
        return Path(VOICE_DIR).glob("*/config.json")

    @staticmethod
    def get_voices() -> dict[str, "Voice"]:
        voices = {}
        for config in Voice.voice_config_paths():
            with open(config) as f:
                data = json.load(f)
                voices[data["id"]] = Voice(
                    id=data["id"],
                    name=data["name"],
                    directory=data["directory"],
                    transcript=data["transcript"],
                )
        return voices


@dataclass
class DecodeConfig:
    codec_audio_sr: int = 16000
    codec_sr: int = 50
    top_k: int = 0
    top_p: float = 0.8
    temperature: float = 1
    silence_tokens: list[int] = field(default_factory=lambda: [1388, 1898, 131])
    kvcache: int = 1
    # NOTE if the model generate long silence, reduce the stop_repetition to 3, 2 or even 1
    stop_repetition: int = 3
    # NOTE: if the if there are long silence or unnaturally strecthed words, increase sample_batch_size to 5 or higher. What this will do to the model is that the model will run sample_batch_size examples of the same audio, and pick the one that's the shortest. So if the speech rate of the generated is too fast change it to a smaller number.
    sample_batch_size: int = 4


def download_encodec(encodec_name="encodec_4cb2048_giga.th"):
    encodec_path = os.path.join(CHECKPOINT_DIR, encodec_name)
    if not os.path.exists(encodec_path):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/{encodec_name} -O /tmp/{encodec_name}")
        os.system(f"mv /tmp/{encodec_name} {encodec_path}")
    return encodec_path


@dataclass
class InferenceStorage:
    _strategy: "Strategy"
    _voice: Voice
    _history: list[tuple[str, torch.Tensor]] = field(default_factory=list)

    def __post_init__(self):
        self._history.append(
            (
                self._voice.transcript,
                torchaudio.load(self._voice.get_sample_path())[0].cuda(),
            )
        )

    def update(self, transcript: str, gen_frames: torch.Tensor):
        self._history.append((transcript, gen_frames))

    def next_transcript(self, transcript):
        return self._strategy.next_transcript(self._history, transcript)

    def next_tokenized_sample(self):
        return self._strategy.next_tokenized_sample(self._history)


class Strategy(ABC):
    @abstractmethod
    def next_transcript(self, history, transcript):
        pass

    @abstractmethod
    def next_tokenized_sample(self, history):
        pass


class GroundTruthStrategy(Strategy):
    def next_transcript(self, history, transcript):
        return history[0][0] + " " + transcript

    def next_tokenized_sample(self, history):
        return history[0][1]


class SlidingWindowStrategy(Strategy):
    def next_transcript(self, history, transcript):
        t = history[0][0]

        if len(history) > 1:
            t += " " + history[-1][0]

        return t + " " + transcript

    def next_tokenized_sample(self, history):
        if len(history) == 1:
            return history[0][1]

        return torch.cat(
            spacer(
                [
                    history[0][1],
                    history[-1][1],
                ],
                torch.zeros((1, int(16e3 * 0.2))).cuda(),
            ),
            dim=1,
        )


class LastGenerationStrategy(Strategy):
    def next_transcript(self, history, transcript):
        return history[-1][0] + " " + transcript

    def next_tokenized_sample(self, history):
        return history[-1][1]


class Middleware:
    def load(self, model_path, encodec_path):
        self.model = VoiceCraft.from_pretrained(model_path)
        self.phn2num = self.model.args.phn2num
        self.config = vars(self.model.args)
        self.model.to(device)

        self.text_tokenizer = TextTokenizer(backend="espeak")
        self.audio_tokenizer = AudioTokenizer(signature=encodec_path, device=device)

    def generate(
        self,
        voice: Voice,
        transcript: str,
        decode_config: DecodeConfig,
        strategy: Strategy,
        chunker=split_on_pause,
    ) -> str:
        dst = os.path.join(voice.directory, string_to_sha256(transcript))
        os.makedirs(dst, exist_ok=True)

        with open(os.path.join(dst, "config.json"), "w") as f:
            json.dump({"transcript": transcript}, f)

        storage = InferenceStorage(
            strategy,
            voice,
        )

        waveforms = []
        for i, part in enumerate(chunker(transcript)):
            print(f"Generating: {part}")
            _, gen_frames = self._infer(
                storage.next_transcript(part),
                storage.next_tokenized_sample(),
                decode_config,
            )
            gen_sample = self.audio_tokenizer.decode([(gen_frames, None)])
            gen_audio = gen_sample[0].cpu()
            waveforms.append(gen_audio)
            storage.update(part, gen_sample[0])

        joined_waveform = torch.cat(spacer(waveforms, torch.zeros((1, int(16e3 * 0.2)))), dim=1)
        torchaudio.save(os.path.join(dst, "gen_audio.wav"), joined_waveform, decode_config.codec_audio_sr)

        return os.path.join(dst, "gen_audio.wav")

    @torch.no_grad()
    def _infer(
        self, target_text, wav, decode_config: DecodeConfig
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_tokens = [
            self.phn2num[phn]
            for phn in tokenize_text(self.text_tokenizer, text=target_text.strip())
            if phn in self.phn2num
        ]
        text_tokens = torch.LongTensor(text_tokens).unsqueeze(0)
        text_tokens_lens = torch.LongTensor([text_tokens.shape[-1]])

        wav = wav.unsqueeze(0)
        with torch.no_grad():
            encoded_frames = self.audio_tokenizer.encode(wav)

        original_audio = encoded_frames[0][0].transpose(2, 1)  # [1,T,K]
        assert (
            original_audio.ndim == 3
            and original_audio.shape[0] == 1
            and original_audio.shape[2] == self.config["n_codebooks"]
        ), original_audio.shape
        logging.info(
            f"original audio length: {original_audio.shape[1]} codec frames, which is {original_audio.shape[1]/decode_config.codec_sr:.2f} sec."
        )
        stime = time.time()
        logging.info(f"running inference with batch size {decode_config.sample_batch_size}")
        concat_frames, gen_frames = self.model.inference_tts_batch(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            original_audio[..., : self.config["n_codebooks"]].to(device),  # [1,T,8]
            top_k=decode_config.top_k,
            top_p=decode_config.top_p,
            temperature=decode_config.temperature,
            stop_repetition=decode_config.stop_repetition,
            kvcache=decode_config.kvcache,
            batch_size=decode_config.sample_batch_size,
            silence_tokens=(
                eval(decode_config.silence_tokens)
                if type(decode_config.silence_tokens) == str
                else decode_config.silence_tokens
            ),
        )
        logging.info(f"inference on one sample take: {time.time() - stime:.4f} sec.")
        logging.info(
            f"generated encoded_frames.shape: {gen_frames.shape}, which is {gen_frames.shape[-1]/decode_config.codec_sr} sec."
        )

        return concat_frames, gen_frames
