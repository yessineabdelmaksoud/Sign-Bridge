# Data attribution

SignBridge uses two open datasets. Both are licensed under Creative Commons Attribution 4.0 (CC BY 4.0).

## Tunisian Sign Language Dataset

- Authors: Wassim Chakroun and Rihab Jerbi (Universite de Carthage, Ecole Superieure des Communications de Tunis), collected with the Tunisian Association of Sign Language Interpreters (ATILS).
- Source: Mendeley Data, DOI 10.17632/fbjjgzgv7f.1, https://data.mendeley.com/datasets/fbjjgzgv7f
- Used for: the image frames in `Demo/TextToSign/data/`, the 27 Tunisian everyday words of the sign model, and their stick figures in `web/data/sign_poses.json`.
- Changes: a subset of the images is redistributed in `Demo/TextToSign/data/`; body and hand points are extracted from the images, and only their movement is used in the web app.

## SWL-LSE: SignaMed Word-Level LSE

- Authors: Manuel Vazquez-Enriquez, Jose Luis Alba-Castro, Laura Docio-Fernandez, Eduardo Rodriguez-Banga and colleagues (Universidade de Vigo), with the Spanish deaf community (CNSE, FAXPG).
- Source: Zenodo record 13691887, https://zenodo.org/records/13691887
- Paper: "SWL-LSE: A Dataset of Spanish Sign Language Health Signs with an ISLR Baseline Method", Technologies 2024, 12(10), 205, DOI 10.3390/technologies12100205.
- Used for: training and evaluating the 22 medical signs of the sign model; the 22 reference videos in `web/clips/`, used only as input to the tracker in the demonstration and never displayed; the stick figures of those signs in `web/data/sign_poses.json`.
- Changes: 22 of the 300 sign classes are used; the MediaPipe landmarks are trimmed, resampled and converted to our own feature layout; only the movement of the reference videos is shown.

The medical signs recognised by SignBridge are Spanish Sign Language (LSE) signs, not Tunisian Sign Language.

## Tunisian speech model

- Source: Hugging Face, Sali7a8603/Tunisian_STT, https://huggingface.co/Sali7a8603/Tunisian_STT (Apache License 2.0).
- Used for: the doctor's dictation in Tunisian Arabic, run by Vosk on the local server.
- Not redistributed: `python3 web/speech_service.py download` fetches it into `data/tunisian_stt/`, which is not committed.

The Tunisian voice that reads the patient's sentences is ElevenLabs' service (voice `InB4o9iYj3MQ4HtGs2KV`), used with your own key under ElevenLabs' terms.
