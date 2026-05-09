# Reorganization Summary

## Current Output Layout

Generated artifacts are stored directly inside each project `outputs` category folder. The project name is not repeated after the category folder.

```text
ProjectName/
`-- outputs/
    |-- output_jsons/
    |-- images/
    |-- rejected_images/
    |-- audios/
    |-- clips/
    `-- videos/
```

## Path Rules

- JSON files save to `outputs/output_jsons/`
- Approved images save to `outputs/images/`
- Rejected images save to `outputs/rejected_images/`
- Audio files save to `outputs/audios/`
- Generated clips save to `outputs/clips/`
- Final videos save to `outputs/videos/`

The folder name is `rejected_images` with an underscore.
