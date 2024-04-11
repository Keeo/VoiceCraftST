# VoiceCraftST

VoiceCraftST is a Python-based API designed to seamlessly integrate SillyTavern with VoiceCraft. It brings XTTs like style api to be used without any modification on ST side.


## Installation

The API utilizes the VoiceCraft model, due to its complex dependencies VCST supports installation via Docker only. Please note that the model requires an NVIDIA GPU, which is currently supported only on Linux systems. Windows users are advised to consult the official VoiceCraft documentation for installation instructions.

Follow these steps to install the API on Linux:

### Step 1: Install Docker

Before you can run the API, you must install Docker. Visit [Docker's official website](https://docs.docker.com/get-docker/) and follow the instructions to install Docker on your Linux distribution.

### Step 2: Install NVIDIA Docker

To support NVIDIA GPUs, you must install NVIDIA Docker. Go to [NVIDIA Docker's GitHub repository](https://github.com/NVIDIA/nvidia-docker) and follow their installation guide for your specific Linux distribution.

### Step 3: Clone the Repository

Once Docker and NVIDIA Docker are installed, clone the API repository to your local machine using the following command and navigate to the cloned folder:

```bash
git clone https://github.com/Keeo/VoiceCraftST.git
cd VoiceCraftST
```

### Step 4: Start the API Using Docker-Compose

In the cloned directory, you will find a `docker-compose.yaml` file provided with the API. Use this file to start the API by running:

```
docker-compose up
```

This command will build the necessary Docker images and start the service defined in the `docker-compose.yaml` file.

### Note:

By following these installation steps, VCST is available on port 5000 on your local system. 


## Configuration in SillyTavern

This section will guide you through the process of setting up SillyTavern to connect with VoiceCraftST. Follow the steps below to configure the integration correctly.

### Step 1: Select TTS Provider

- Open the SillyTavern application.
- Navigate to the TTS section in the settings.
- From the 'Select TTS Provider' dropdown, choose 'XTTSv2' as your Text-to-Speech (TTS) provider.

### Step 2: Configure TTS Settings

- Make sure the 'Enabled' checkbox is checked to activate TTS functionalities.
- Check the 'Auto Generation' box if you want SillyTavern to automatically generate speech from text.
- Optionally, you can adjust the following settings to your preference:
  - Narrate User Messages: If checked, SillyTavern will narrate messages made by users.
  - Only Narrate "quotes": SillyTavern will only narrate text enclosed in quotation marks.
  - Ignore Text in Asterisks: Text within asterisks will not be narrated.
  - Skip Codeblocks: Ensures sections of code are not read aloud.
  - Pass Asterisks to TTS Engine: If checked, asterisks are sent to the TTS engine as part of the text.

### Step 3: Set Up XTTS Settings

- Under 'XTTS Settings,' find the 'Provider Endpoint' field.
- Enter the following URL to connect to the VoiceCraftST TTS Server: `http://localhost:5000/{username}/{stop_repetition}/{sample_batch_size}`. The url contains three parameters as they are not part of default XTTSv2 configuration.
  - username: Choose any username you want, it is used as a key for rest of the configuration.
  - stop_repetition: if the model generate long silence, reduce the stop_repetition to 3, 2 or even 1 (default: 3)
  - sample_batch_size: if the if there are long silence or unnaturally strecthed words, increase sample_batch_size to 5 or higher. What this will do to the model is that the model will run sample_batch_size examples of the same audio, and pick the one that's the shortest. So if the speech rate of the generated is too fast change it to a smaller number. (default: 4)

- Adjust the TTS parameters according to your needs:
  - Speed: Does nothing.
  - Temperature: Sets the variance in speech generation. (default: 1.0)
  - Length Penalty: Does nothing.
  - Repetition Penalty: Does nothing.
  - Top K: (default: 0)
  - Top P: (default: 0.8)
  - Stream Chunk Size: Does nothing.

### Step 4: Enable Text Splitting (Optional)

- If you're dealing with large blocks of text, check 'Enable Text Splitting' to ensure smooth and continuous narration. More at [VoiceCraft#39](https://github.com/jasonppy/VoiceCraft/issues/39). (default: True) 


## Adding New Voices

If you wish to expand the range of voices available through the API, you can add new voice samples using the following process:

### Step 1: Prepare the Voice Sample

Record a new voice sample and save it as a mono WAV file with a sample rate of 16,000 Hz.

### Step 2: Create a Transcript

Write down an exact transcript of what is spoken in the voice sample. This transcript will be used for training and referencing within the system.

### Step 3: Run the Upload Command

Use the provided command to upload your new voice sample along with its transcript to the system:

```bash
make upload SPEAKER_NAME=sample TRANSCRIPT="Use the provided command to upload your new voice sample along with its transcript to the system." FILE_PATH=sample.wav
```

Replace `sample` with your chosen `SPEAKER_NAME`, `"Use the provided command... system.` with your actual `TRANSCRIPT` of the voice sample, and `sample.wav` with the correct `FILE_PATH` where your voice sample is saved.

After running this command, the system will process the new voice and add it to the available voice options. Press `reload` in ST to see it being added.
