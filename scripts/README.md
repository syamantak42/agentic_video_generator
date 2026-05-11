# Agentic Video Generator

Generate a narrated video from source material using a guided app or individual scripts.

## Recommended: Streamlit App

Run from the repo root:

```bash
streamlit run scripts/streamlit_app.py
```

From inside `scripts/`, this also works:

```bash
python streamlit_app.py
```

The app has five pages:

- `Config Wizard`: creates and edits `{Project}/source_material/config.json`
- `Narration`: generates outline JSON, validates it, generates narration, validates it, creates image prompts, and generates audio
- `Image Review`: generates images one prompt at a time, then keeps or rejects them
- `Video Generation`: creates clips and composes the final video
- `Outputs`: shows generated artifact folders

PDF/TXT links are downloaded into `source_material`. HTML and Wikipedia links stay as reference links for retrieval.

## Project Layout

```text
YourProject/
|-- source_material/
|   |-- config.json
|   |-- notes.txt
|   `-- source.pdf
`-- outputs/
    |-- output_jsons/
    |-- images/
    |-- rejected_images/
    |-- audios/
    |-- clips/
    `-- videos/
```

Use [`config.template.json`](config.template.json) as a starting point.

Important config fields:

- `video_title`
- `n_section`
- `section_outlines`
- `reference_links`
- `narration_style`
- `historical_context`
- `characters`
- `aesthetic_style`
- `_project_config.narration_config.words_per_section`
- `_project_config.image_config.model`
- `_project_config.tts_config.model`

`tts_config.model` can be `kokoro` or `inworld`.

## Environment

Add API keys to the repo-level `.env` file as needed:

```env
DEEPSEEK_API_KEY=...
FAL_KEY=...
INWORLD_API_KEY=...
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Individual Scripts

Run from `scripts/`:

```bash
python generate_sections.py YourProject
python validate_outline.py YourProject
python generate_script.py YourProject
python validate_narration.py YourProject
python generate_image_prompts.py YourProject
python generate_kokoro_voice.py YourProject
python generate_inworld_voice.py YourProject
python generate_clips.py YourProject
python make_video.py YourProject
```

Selective clip generation:

```bash
python generate_clips.py YourProject --clip-selection missing
python generate_clips.py YourProject --clip-selection all
python generate_clips.py YourProject --clip-selection 1_1
python generate_clips.py YourProject --clip-selection 1_1,2_3,5_1
python generate_clips.py YourProject --clip-selection 1_1-3_2
```

## Prompt Templates

Large LLM prompts live in `scripts/Prompts/`.
Scripts load them with `prompt_loader.py`.

## Main Outputs

```text
outputs/output_jsons/outline_texts.json
outputs/output_jsons/narration.json
outputs/output_jsons/image_prompts.json
outputs/output_jsons/tts_index.json
outputs/images/image_<section>_<segment>.png
outputs/rejected_images/
outputs/audios/audio_<section>_<segment>.<ext>
outputs/clips/image_<section>_<segment>.mp4
outputs/videos/YourProject.mp4
```
