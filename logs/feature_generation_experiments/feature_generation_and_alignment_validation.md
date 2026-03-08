# Feature Generation and Alignment Validation (2026-03-08)

## 1) How the features were generated (technical overview)
The feature generation process converted each chest CT study and its paired report into two modality-specific embeddings (visual and text) in a shared multimodal representation space.

Core input standardization:
- Start from paired 3D CT volumes and radiology text per study.
- Standardize each volume before encoding:
  - resample to a common physical spacing (`0.75 x 0.75 x 1.5` mm), using NIfTI header spacing first and metadata fallback when headers are invalid
  - clip CT intensities to `[-1000, 1000]` HU
  - scale intensities by `/1000` to keep values numerically stable
  - center-crop/pad to fixed dimensions (`480 x 480 x 240`) so every sample matches the same tensor shape
- Standardize text input by tokenizing to fixed length (`max_length=512`) with attention masks.

Encoder architecture used in this generation:
- Visual encoder: a 3D vision transformer pipeline based on patch tokens and **factorized transformer blocks**:
  - patch embedding with `image_size=480`, `patch_size=20`, and `temporal_patch_size=10`
  - hidden width `dim=512`, `heads=8`, `dim_head=32`
  - separate **spatial transformer** (`spatial_depth=4`) and **temporal transformer** (`temporal_depth=4`)
  - spatial relative positional bias and causal positional encoding behavior enabled in the transformer configuration (`peg_causal=true`)
- Text encoder: `BertModel` initialized from `microsoft/BiomedVLP-CXR-BERT-specialized` (a radiology-domain BERT family model), producing contextual token embeddings from report text.

Multimodal projection/output behavior:
- Text and visual outputs are projected into a shared latent space (`dim_latent=512`) for cross-modal alignment.
- For each study, one text embedding and one visual embedding are persisted as reusable `.npz` features.

For downstream experiments, this avoids repeatedly reprocessing raw CT volumes and preserves a consistent multimodal representation across runs.

## 2) How we validated that extracted features were correct
After feature generation, we ran an alignment validation step to verify that paired visual and text embeddings were consistent.

The validation checks whether each image embedding is most similar to the correct text embedding (and vice versa), instead of unrelated samples. In practice, this was evaluated from two complementary perspectives:
- Instance-level alignment: checks matching pairs from the same study/exam.
- Semantic-level alignment: checks whether samples with the same label pattern are close, even if they are not the exact same instance.

This gave us a quantitative sanity check that the extracted features are coherent and usable.

## 3) Alignment analysis result snapshot (from run log)
Run log reviewed from the alignment output:
- Samples analyzed: `22185`

Instance-level summary:
- Mean paired similarity: `0.624284`
- Mean unpaired similarity: `-0.001245`
- Visual -> Text: `MRR=0.0156`, `MAP=0.0184`, `Recall@1=0.0079`, `Recall@5=0.0203`, `Recall@10=0.0314`, `Recall@50=0.0965`
- Text -> Visual: `MRR=0.0172`, `MAP=0.0151`, `Recall@1=0.0075`, `Recall@5=0.0215`, `Recall@10=0.0378`, `Recall@50=0.1081`

Semantic-level summary (exact label-match criterion):
- Mean paired similarity: `0.183173`
- Mean unpaired similarity: `-0.013819`
- Visual -> Text: `MRR=0.2145`, `MAP=0.1990`, `Recall@1=0.1418`, `Recall@5=0.2952`, `Recall@10=0.3970`, `Recall@50=0.7173`
- Text -> Visual: `MRR=0.2847`, `MAP=0.2102`, `Recall@1=0.1638`, `Recall@5=0.4132`, `Recall@10=0.5603`, `Recall@50=0.8710`

Interpretation in plain terms:
- Exact one-to-one retrieval is difficult at this dataset scale, which is expected.
- Even so, true pairs are clearly more similar than unrelated pairs.
- Semantic retrieval is substantially stronger, indicating the embeddings encode meaningful pathology-level structure.

## 4) Conclusion
The extracted feature set passed a meaningful alignment sanity check. The validation supports that the generated embeddings are suitable for downstream classification and comparative experiments.