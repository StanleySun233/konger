# KONGER

This repository accompanies “KONGER: Uncovering Layer-Wise Phonetic Confuser Dynamics for Selective Adaptation in Maritime VHF Speech.”

The repository contains the manuscript source and code for constructing VHF-8K audio and its ASR manifests. It does not contain model training code, PBS job scripts, model weights, raw audio, or the VHF-500 recordings and annotations.

## Datasets

### VHF-8K

VHF-8K contains 8,000 synthetic maritime-radio utterances. Text-to-speech audio is rendered with four VHF channel profiles: `vhf_clear`, `vhf_busy`, `vhf_weak`, and `vhf_tail_beep`. The training, discovery, confirmation, and held-out test splits contain 3,952, 1,024, 1,024, and 2,000 utterances.

Each ASR manifest record links audio to a reference transcript and includes dialogue and utterance identifiers, the channel profile, and `positive_entities`, a list of vessel names appearing in the reference transcript. For synthetic utterances, entity labels are constructed by matching the transcript against the vessel roster supplied with the generated dialogue. The manifest also records split and candidate metadata used by the ASR experiments.

The construction scripts expect a dialogue JSONL input and an authorized VHF noise-audio source. Neither input data nor generated audio is distributed here.

### VHF-500

VHF-500 contains 500 real-radio sentences and is used only for evaluation. ASR transcripts were manually checked and annotated against the audio. Vessel-name information was then manually inspected and labeled. The set contains 432 sentences with vessel mentions and 68 without. The recordings and annotations are not distributed in this repository.

## Training configuration

The reported adaptation uses Qwen3-ASR-1.7B. It trains LoRA adapters on the decoder query and value projections with a total rank budget of 28. KGPRA selects one-based layers 19–21, 23, and 25–28 with ranks 2, 2, 7, 2, 2, 3, 5, and 5, respectively. The resulting adapter has 200,704 trainable parameters and uses LoRA α=8.

Optimization uses AdamW for 250 updates, batch size 1, learning rate 2×10⁻⁴, and 2% warmup. Adaptation uses the 3,952-utterance training split; discovery and confirmation each contain 1,024 utterances, and the 2,000-utterance test split is held out.

Model training and evaluation implementations are not released in this repository.

## VHF-8K construction code

The scripts are under `dataset/scripts/`; shared audio and profile modules are under `dataset/src/`.

| Script | Purpose |
|---|---|
| `build_vhf_noise_bank.py` | Extract VHF channel-noise segments from a user-supplied audio source |
| `run_step2_tts_vhf.py` | Generate profile-specific audio and an initial audio manifest from dialogue JSONL |
| `prepare_step3_asr_manifest.py` | Create transcript and vessel-entity labels and ASR manifest records |
| `build_vhf8k_manifest.py` | Select the profile-balanced VHF-8K manifest and disjoint discovery and confirmation partitions |
| `build_vhf8k_test_manifest.py` | Select a profile-balanced held-out test manifest with dialogue, utterance, and entity-family exclusions |
| `build_vhf8k_paired_evaluation.py` | Construct matched no-context, supported-context, and Konger-context evaluation views |

`build_vhf_noise_bank.py` requires `--source-dir` and does not assume a local dataset path. Dialogue JSONL records supplied to `run_step2_tts_vhf.py` contain `dialogue_id`, `event`, `intent`, `provider`, `model`, `ship_names`, and `utterance` fields. Generated manifests refer to audio paths on the user's machine.

## Manuscript

The LaTeX source, bibliography, figures, and conference style file are under `paper/`.

VHF-8K construction code is available at https://github.com/StanleySun233/konger.
