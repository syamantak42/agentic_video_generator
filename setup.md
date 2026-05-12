# Setup Guide

This guide explains how to set up the project for use through the Streamlit app.

## 1. Install Python Dependencies

Create and activate your Python environment, then install the requirements:

```bash
pip install -r scripts/requirements.txt
```

If you run commands from inside the `scripts/` folder instead, use:

```bash
pip install -r requirements.txt
```

## 2. Create the `.env` File

Create one `.env` file at the repo root:

```text
agentic_video_generator/
|-- .env
|-- scripts/
|-- YourProject/
```

Do not put `.env` inside `scripts/` or inside a project folder.

The `.env` file should contain only the keys for services you use:

```env
DEEPSEEK_API_KEY=your_deepseek_key_here
FAL_KEY=your_fal_key_here
INWORLD_API_KEY=your_inworld_key_here
```

Notes:

- `DEEPSEEK_API_KEY` is needed for outline, narration, validation, and image prompt generation.
- `FAL_KEY` is needed for image generation through the app.
- `INWORLD_API_KEY` is needed only if you choose the Inworld TTS option.
- Kokoro TTS does not require an API key, it runs locally.

Never commit real API keys to git.

## 3. Start the App

From the repo root:

```bash
streamlit run scripts/streamlit_app.py
```

Or from inside `scripts/`:

```bash
python streamlit_app.py
```

## 4. Project Folder Organization

Each video project lives in its own folder:

```text
agentic_video_generator/
|-- scripts/
|-- .env
`-- YourProject/
    |-- source_material/
    |   |-- config.json
    |   |-- source_notes.txt
    |   `-- source_document.pdf
    `-- outputs/
        |-- output_jsons/
        |-- images/
        |-- rejected_images/
        |-- audios/
        |-- clips/
        `-- videos/
```

The app can create this structure for you when you create a new project.


## 5. Typical App Workflow

1. Open or create a project.
2. Complete `Config Wizard` to fill in user-defined details regarding content and choice of generative models (Language, image & text-to-speech models). 
3. Go to `Script Generation` to generate sections and full script.
4. Go to `Image Generation`to generate images, review and save
5. Go to `Voice Generation` to generate the audio (text-to-speech) 
7. Go to `Video Generation` to generate clips and compose the final video.

## 8. Output Locations

Main outputs are saved here:

```text
{Project}/outputs/output_jsons/outline_texts.json
{Project}/outputs/output_jsons/narration.json
{Project}/outputs/output_jsons/image_prompts.json
{Project}/outputs/output_jsons/tts_index.json
{Project}/outputs/images/
{Project}/outputs/rejected_images/
{Project}/outputs/audios/
{Project}/outputs/clips/
{Project}/outputs/videos/
```

The final video is saved in:

```text
{Project}/outputs/videos/
```

## 9. Common Notes

- Do not manually move generated files while the app is running.
- If clip generation is interrupted, rerun `Generate Clips` with `Missing clips`.
- If an image is bad, reject it and regenerate from the same prompt.
- If a previous approved image needs replacing, jump back to that prompt in `Image Generation`, generate a new image, and keep it.
