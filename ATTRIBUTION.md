# Data attribution

SignBridge uses two open datasets. Both are licensed under Creative Commons Attribution 4.0 (CC BY 4.0).

## Tunisian Sign Language Dataset

- Authors: Wassim Chakroun and Rihab Jerbi (Universite de Carthage, Ecole Superieure des Communications de Tunis), collected with the Tunisian Association of Sign Language Interpreters (ATILS).
- Source: Mendeley Data, DOI 10.17632/fbjjgzgv7f.1, https://data.mendeley.com/datasets/fbjjgzgv7f
- Used for: the image frames in `Demo/TextToSign/data/` and the Tunisian sign models.
- Changes: a subset of the images is redistributed here; landmarks are extracted from them.

## SWL-LSE: SignaMed Word-Level LSE

- Authors: Manuel Vazquez-Enriquez, Jose Luis Alba-Castro, Laura Docio-Fernandez, Eduardo Rodriguez-Banga and colleagues (Universidade de Vigo), with the Spanish deaf community (CNSE, FAXPG).
- Source: Zenodo record 13691887, https://zenodo.org/records/13691887
- Paper: "SWL-LSE: A Dataset of Spanish Sign Language Health Signs with an ISLR Baseline Method", Technologies 2024, 12(10), 205, DOI 10.3390/technologies12100205.
- Used for: training and evaluating the health sign model, and the reference sign clips in `Demo/consultation/assets/ref_signs/`.
- Changes: 22 of the 300 sign classes are used; the MediaPipe landmarks are trimmed, resampled and converted to our own feature layout.

The medical signs recognised by SignBridge are Spanish Sign Language (LSE) signs, not Tunisian Sign Language.
