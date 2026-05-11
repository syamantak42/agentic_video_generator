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
- Kokoro TTS does not require an API key.

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

The app opens a guided workspace with these pages:

- `Config Wizard`
- `Narration`
- `Image Review`
- `Video Generation`
- `Outputs`

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

## 5. Source Materials

Use the app's `Config Wizard > Sources` step.

You can provide:

- uploaded `.pdf` files
- uploaded `.txt` files
- PDF links
- TXT links
- HTML links
- Wikipedia links

How they are handled:

- Uploaded PDFs/TXT files are saved into `{Project}/source_material/`.
- PDF/TXT links are downloaded into `{Project}/source_material/`.
- HTML and Wikipedia links remain in `config.json` as `reference_links`.

## 6. Config Wizard

The app writes:

```text
{Project}/source_material/config.json
```

You will fill in:

- video title
- number of sections
- section guidelines
- source links/files
- narration style
- tentative words per section
- TTS model and voice
- visual style
- character canon
- image model

Always use the step's `Save ... and continue` button before moving forward.

## 7. Typical App Workflow

1. Open or create a project.
2. Complete `Config Wizard`.
3. Go to `Narration`.
4. Run the stages in order:
   - Generate Sections
   - Validate Outline
   - Generate Narration
   - Validate Narration
   - Generate Image Prompts
   - Generate Audio
5. Go to `Image Review`.
6. Generate each image, then keep or reject it.
7. Go to `Video Generation`.
8. Generate missing clips, all clips, a range, or selected clips.
9. Compose the final video.

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

- Keep project names simple: letters, numbers, underscores, or hyphens.
- Do not manually move generated files while the app is running.
- If clip generation is interrupted, rerun `Generate Clips` with `Missing clips`.
- If an image is bad, reject it and regenerate from the same prompt.
- If a previous approved image needs replacing, jump back to that prompt in `Image Review`, generate a new image, and keep it.
