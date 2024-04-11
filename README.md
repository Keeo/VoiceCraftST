# VoiceCraftST

VoiceCraftST is a Python API that integrates the advanced text-to-speech capabilities of VoiceCraft with SillyTavern. This API enables the use of VoiceCraft's features within SillyTavern without any need for modifications on the SillyTavern platform.

## Installation

VoiceCraftST is supported on Linux systems with NVIDIA GPUs due to specific hardware dependencies. The API installation is facilitated through Docker to manage its complex dependencies.

### Prerequisites

1. **Docker**: Install Docker by following the guide on [Docker's official website](https://docs.docker.com/get-docker/).
2. **NVIDIA Docker**: Required for NVIDIA GPU support. Installation instructions can be found on [NVIDIA Docker's GitHub repository](https://github.com/NVIDIA/nvidia-docker).

### Installation Steps

1. **Clone the Repository**: Clone VoiceCraftST and initialize its submodule:
   ```bash
   git clone https://github.com/Keeo/VoiceCraftST.git
   cd VoiceCraftST
   git submodule init
   git submodule update
   ```

2. **Deploy Using Docker-Compose**: Navigate to the directory containing `docker-compose.yaml` and execute:
   ```
   docker-compose up
   ```
   This command builds the necessary Docker images and starts the service, making VCST available at port 5000 on your local machine.

## Configuration for SillyTavern

Configure SillyTavern to use VoiceCraftST by following these steps:

### Setup TTS Provider

1. **Choose TTS Provider**: In SillyTavern settings, select 'XTTSv2' as the TTS provider.
2. **TTS Features**:
   - **Enable TTS**: Ensure the 'Enabled' checkbox is checked.
   - **Auto Generation**: Activate automatic speech generation from text.
   - **Narration and Text Preferences**: Configure preferences for narrating user messages, handling special text formats (like quotes or code), and processing text with special characters.

### Advanced TTS Settings

1. **Provider Endpoint Configuration**: Set up the XTTS endpoint with specific parameters to optimize performance:
   ```
   http://localhost:5000/{username}/{stop_repetition}/{sample_batch_size}
   ```
   Customize settings such as `username`, `stop_repetition`, and `sample_batch_size` based on your requirements.
   - `username`: Choose any username you want, it is used as a key for rest of the configuration.
   - `stop_repetition`: if the model generate long silence, reduce the stop_repetition to 3, 2 or even 1 (default: 3)
   - `sample_batch_size`: if the if there are long silence or unnaturally strecthed words, increase sample_batch_size to 5 or higher. What this will do to the model is that the model will run sample_batch_size examples of the same audio, and pick the one that's the shortest. So if the speech rate of the generated is too fast change it to a smaller number. (default: 4)

2. **Parameter Tuning**: Adjust TTS parameters like `Temperature`, `Top P` and `Top K` to fine-tune the speech generation characteristics.
   - Speed: Does nothing.
   - Temperature: Sets the variance in speech generation. (default: 1.0)
   - Length Penalty: Does nothing.
   - Repetition Penalty: Does nothing.
   - Top K: (default: 0)
   - Top P: (default: 0.8)
   - Stream Chunk Size: Does nothing.

### Optional: Enable Text Splitting

For better handling of large text blocks, enable the 'Text Splitting' feature to ensure smooth and continuous narration. More at [VoiceCraft#39](https://github.com/jasonppy/VoiceCraft/issues/39). (default: True) 

## Adding New Voices

To add new voices to the system:

1. **Prepare Voice Sample**: Record a mono WAV file at 16,000 Hz sample rate.
2. **Create Transcript**: Accurately transcribe the voice sample.
3. **Upload Command**: Use the following command to upload the new voice sample and its transcript:
   ```bash
   make upload SPEAKER_NAME=sample TRANSCRIPT="Your transcript here." FILE_PATH=sample.wav
   ```
   Replace placeholders with actual data. After uploading, reload SillyTavern to access the new voice.

## System requirements
In current state it requires 24gb GPU [VRAM requirements for training, finetuning, and inference?](https://github.com/jasonppy/VoiceCraft/issues/76) but it is likely to go down. Speed is slower compared to XTTSv2 and with TopP below 0.85 likes to generate long silences which decrease the speed further. `sample_batch_size` parameter is supposed to help with that by brute-force and generating multiple alternatives and picking shortest.