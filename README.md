# Agentic Video Generator

Generate a narrated video from source material through a guided Streamlit app or the underlying scripts.

## Run the app

From the repo root:

```bash
streamlit run scripts/streamlit_app.py
```

For installation, `.env` setup, and source-material guidance, see [`setup.md`](setup.md).

## App workflow

The app is organized into:

1. `Config Wizard`
2. `Script Generation`
3. `Image Generation`
4. `Voice Generation`
5. `Video Generation`
6. `Outputs`

### Config Wizard

The wizard writes `{Project}/source_material/config.json`.

- `Content`: title, section count, guidelines, narration style, tentative words per section, tentative frames per section, aesthetic style, plus optional advanced historical context and character canon.
- `Sources`: uploaded PDF/TXT files and source links.
- `Models`: DeepSeek language model, image generator model, TTS model, and voice.

The app supports `Generate Video in Fully Automated Mode`, which runs the pipeline end to end and accepts the first generated image for each prompt without manual image review.

### Script Generation

- `Generate Section Outlines`
  - generates outlines
  - silently revises them
- `Generate Full Script`
  - generates `narration.json`
  - silently revises the script sections

### Image Generation

- Generates image prompts.
- Supports prompt-by-prompt image review with keep/reject controls.
- Also supports generating all images one by one without review.

### Voice Generation

Generates audio selectively for:

- missing frames
- all frames
- a range
- specific frames

### Video Generation

- `Generate Clips`
- `Compose Final Video`

Clips can be generated selectively for missing clips, all clips, a range, or specific clip keys.

## Project layout

```text
agentic_video_generator/
|-- .env
|-- README.md
|-- setup.md
|-- config.template.json
|-- scripts/
|   |-- streamlit_app.py
|   |-- image_models.txt
|   |-- voices.txt
|   `-- Prompts/
`-- YourProject/
    |-- source_material/
    |   `-- config.json
    `-- outputs/
        |-- output_jsons/
        |-- images/
        |-- rejected_images/
        |-- audios/
        |-- clips/
        `-- videos/
```

## Config-driven controls

Important config values include:

- `_project_config.validator_config.model`
  - `deepseek-v4-flash`
  - `deepseek-v4-pro`
- `_project_config.image_config.model`
  - selected from `scripts/image_models.txt`
- `_project_config.tts_config.model`
  - `kokoro`
  - `inworld-tts-1.5-max`
  - `inworld-tts-2`
- `_project_config.tts_config.voice_id`
  - selected from `scripts/voices.txt`
- `_project_config.narration_config.words_per_section`
- `_project_config.narration_config.frames_per_section`

The narration prompt uses `frames_per_section` as an approximate `n` to `n+1` target for narration frames per section.

## Model option files

`scripts/image_models.txt` is a JSON dictionary:

```json
{
  "seedream-v4": "fal-ai/bytedance/seedream/v4/text-to-image"
}
```

`scripts/voices.txt` is a JSON dictionary where:

- `false` means Kokoro voice
- `true` means Inworld voice

```json
{
  "af": false,
  "Ashley": true
}
```

The app filters the voice dropdown according to the selected TTS model.

## Main outputs

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

## Script entry points

The app is the preferred interface, but scripts can still be run directly from the repo root:

```bash
python scripts/generate_sections.py YourProject
python scripts/validate_outline.py YourProject
python scripts/generate_script.py YourProject
python scripts/validate_narration.py YourProject
python scripts/generate_image_prompts.py YourProject
python scripts/generate_kokoro_voice.py YourProject --audio-selection all
python scripts/generate_inworld_voice.py YourProject --audio-selection all
python scripts/generate_clips.py YourProject --clip-selection all
python scripts/make_video.py YourProject
```

The `validate_*` script filenames are retained for compatibility, but their current role is revision.
