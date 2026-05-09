# Agentic Video Generator

Generate a narrated video from source documents in a step-by-step pipeline.

## What it does

The pipeline:

1. Builds section outlines from your source material
2. Writes narration
3. Creates image prompts
4. Generates images
5. Generates voice audio
6. Turns images and audio into clips
7. Combines the clips into a final video

## Project Setup

Create a project folder like this:

```text
YourProject/
|-- source_material/
|   |-- config.json
|   |-- notes.txt
|   `-- source.pdf
`-- outputs/
```

Add your API keys to the repo-level `.env` file as needed, for example:

```env
DEEPSEEK_API_KEY=...
FAL_KEY=...
INWORLD_API_KEY=...
```

## config.json

Use [`config.template.json`] as your starting point.

Minimum example:

```json
{
  "video_title": "Your Video Title",
  "n_section": 6,
  "section_outlines": [
    "Introduction",
    "Main topic",
    "Conclusion"
  ],
  "reference_links": [],
  "narration_style": [
    "Clear and engaging"
  ],
  "historical_context": "",
  "characters": {},
  "aesthetic_style": "cinematic",
  "_project_config": {
    "project_name": "YourProject",
    "image_config": {
      "model": "seedream-v4"
    },
    "tts_config": {
      "model": "kokoro",
      "voice_id": "af"
    }
  }
}
```

`tts_config.model` can be `kokoro` or `inworld`.

## Run

### Streamlit App

For the guided app experience:

```bash
streamlit run scripts/streamlit_app.py
```

The app helps create `source_material/config.json`, download or upload source files, run pipeline stages, review generated images, and compose the final video.

### Command Line

From the `scripts/` folder:

```bash
python run_pipeline.py YourProject
```

The pipeline pauses before each stage so you can continue, rerun, skip, or quit.

## Run Individual Stages

```bash
python generate_sections.py YourProject
python generate_script.py YourProject
python generate_image_prompts.py YourProject
python generate_images.py YourProject
python generate_kokoro_voice.py YourProject
python generate_inworld_voice.py YourProject
python generate_clips.py YourProject
python make_video.py YourProject
```

## Outputs

Generated files are written inside your project folder:

```text
YourProject/
`-- outputs/
    |-- output_jsons/
    |-- images/
    |-- rejected_images/
    |-- audios/
    |-- clips/
    `-- videos/YourProject.mp4
```

Main JSON outputs:

- `outline_texts.json`
- `narration.json`
- `image_prompts.json`
- `tts_index.json`

## Requirements

- Python 3
- Install dependencies with `pip install -r requirements.txt`
- API access for the services you use
