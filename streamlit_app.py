from pathlib import Path
import streamlit as st
import torch


from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torch.nn.functional as F
import librosa

from scratch_model import model_crnn
from secret_model import model_oth


ROOT_DIR = Path(__file__).resolve().parents[1]

AST_BACKBONE = "MIT/ast-finetuned-audioset-10-10-0.4593"
MODEL_PATHS = {
    "AST Model": ROOT_DIR / "models" / "best_ast.pt",
    "CRNN": ROOT_DIR / "models" / "model_scratch.pth",
    "Secret Sauce": ROOT_DIR / "models" / "best_oth.pt",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_label_maps(ckpt: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = ckpt.get("label2id") or {}
    id2label = ckpt.get("id2label") or {}

    label2id = {str(k): int(v) for k, v in label2id.items()}
    id2label = {int(k): str(v) for k, v in id2label.items()}

    if not label2id and id2label:
        label2id = {v: k for k, v in id2label.items()}
    if not id2label and label2id:
        id2label = {v: k for k, v in label2id.items()}

    return label2id, id2label

def build_scratch_model(num_classes=10):
    # Initialize CRNN model with the same architecture as used during training
    ckpt = torch.load(MODEL_PATHS["Secret Sauce"], map_location=DEVICE)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    else:
        state_dict = ckpt

    if state_dict is None or not isinstance(state_dict, dict):
        raise ValueError("SecretNN checkpoint does not contain a valid state_dict.")

    model_crnn.load_state_dict(state_dict, strict=True)
    return model_crnn.to(DEVICE).eval()

def build_oth_model(num_classes=10):
    # Initialize CRNN model with the same architecture as used during training
    ckpt = torch.load(MODEL_PATHS["CRNN"], map_location=DEVICE)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    else:
        state_dict = ckpt

    if state_dict is None or not isinstance(state_dict, dict):
        raise ValueError("CRNN checkpoint does not contain a valid state_dict.")

    model_oth.load_state_dict(state_dict, strict=True)
    return model_oth.to(DEVICE).eval()


def wav_to_logmel(
    wav: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
    target_frames: int = 600,
) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, top_db=80)

    frames = log_mel.shape[-1]
    if frames < target_frames:
        log_mel = np.pad(log_mel, ((0, 0), (0, target_frames - frames)), mode="constant")
    elif frames > target_frames:
        log_mel = log_mel[:, :target_frames]

    x = torch.tensor(log_mel, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
    return x

def build_ast_from_checkpoint(ckpt: Dict[str, Any]):
    from transformers import AutoConfig, ASTForAudioClassification, AutoFeatureExtractor

    label2id, id2label = normalize_label_maps(ckpt)
    num_labels = len(label2id) if len(label2id) else 10

    config = AutoConfig.from_pretrained(
        AST_BACKBONE,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    model = ASTForAudioClassification.from_pretrained(
        AST_BACKBONE,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained(AST_BACKBONE)
    return model, feature_extractor, label2id, id2label


@st.cache_resource(show_spinner=False)
def load_cached_model(model_name: str):
    path = MODEL_PATHS[model_name]
    ckpt = torch.load(path, map_location=DEVICE)

    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
        meta = {"inference_type": "generic", "label2id": {}, "id2label": {}}
    elif isinstance(ckpt, dict):
        if isinstance(ckpt.get("model"), torch.nn.Module):
            model = ckpt["model"]
            label2id, id2label = normalize_label_maps(ckpt)
            meta = {
                "inference_type": "generic",
                "label2id": label2id,
                "id2label": id2label,
            }
        elif "model_state_dict" in ckpt:
            if model_name == "AST Model":
                model, feature_extractor, label2id, id2label = build_ast_from_checkpoint(ckpt)
                meta = {
                    "inference_type": "ast",
                    "feature_extractor": feature_extractor,
                    "label2id": label2id,
                    "id2label": id2label,
                    "sample_rate": int((ckpt.get("cfg") or {}).get("sample_rate", 16_000)),
                    "clip_seconds": float((ckpt.get("cfg") or {}).get("clip_seconds", 6.0)),
                }
            elif model_name == "CRNN":
                label2id, id2label = normalize_label_maps(ckpt)
                cfg = ckpt.get("cfg") or {}
                model = build_scratch_model(num_classes=10)
                meta = {
                    "inference_type": "crnn",
                    "label2id": label2id,
                    "id2label": id2label,
                    "sample_rate": int(cfg.get("sample_rate", 16_000)),
                    "clip_seconds": float(cfg.get("clip_seconds", 6.0)),
                    "n_mels": int(cfg.get("n_mels", 128)),
                    "n_fft": int(cfg.get("n_fft", 400)),
                    "hop_length": int(cfg.get("hop_length", 160)),
                    "win_length": int(cfg.get("win_length", 400)),
                    "target_frames": int(cfg.get("target_frames", 600)),
                }

            else:
                raise ValueError(
                    f"{model_name} checkpoint has only model_state_dict. "
                    "Add architecture code to load this model."
                )
        else:
            raise ValueError(f"Unsupported checkpoint dictionary format. Keys: {list(ckpt.keys())}")
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt).__name__}")

    model = model.to(DEVICE)
    model.eval()
    return model, meta


def load_audio(uploaded_file, target_sr: int, clip_seconds: float) -> np.ndarray:
    y, sr = sf.read(uploaded_file, always_2d=False)
    if isinstance(y, np.ndarray) and y.ndim == 2:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    target_len = int(target_sr * clip_seconds)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    elif len(y) > target_len:
        y = y[:target_len]

    y = np.clip(y, -1.0, 1.0)
    return y


def predict_genre(model_name: str, model: torch.nn.Module, meta: Dict[str, Any], uploaded_file):
    inference_type = meta.get("inference_type", "generic")
    id2label = meta.get("id2label") or {}

    sample_rate = int(meta.get("sample_rate", 16_000))
    clip_seconds = float(meta.get("clip_seconds", 6.0))
    wav = load_audio(uploaded_file, target_sr=sample_rate, clip_seconds=clip_seconds)

    with torch.no_grad():
        if inference_type == "ast":
            feature_extractor = meta["feature_extractor"]
            inputs = feature_extractor(wav, sampling_rate=sample_rate, return_tensors="pt")
            x = inputs["input_values"].to(DEVICE)
            logits = model(input_values=x).logits
        elif inference_type == "crnn":
            x = wav_to_logmel(
                wav,
                sample_rate=sample_rate,
                n_mels=int(meta.get("n_mels", 128)),
                n_fft=int(meta.get("n_fft", 400)),
                hop_length=int(meta.get("hop_length", 160)),
                win_length=int(meta.get("win_length", 400)),
                target_frames=int(meta.get("target_frames", 600)),
            )
            logits = model(x)
        else:
            x = wav_to_logmel(
                wav,
                sample_rate=sample_rate,
                n_mels=int(meta.get("n_mels", 128)),
                n_fft=int(meta.get("n_fft", 400)),
                hop_length=int(meta.get("hop_length", 160)),
                win_length=int(meta.get("win_length", 400)),
                target_frames=int(meta.get("target_frames", 600)),
            )
            logits = model(x)

    probs = F.softmax(logits, dim=-1).squeeze(0).detach().cpu()
    top_probs, top_ids = torch.topk(probs, k=min(3, probs.shape[0]))

    top_predictions = []
    for p, i in zip(top_probs.tolist(), top_ids.tolist()):
        label = id2label.get(i, f"class_{i}")
        top_predictions.append((label, float(p)))

    pred_label = top_predictions[0][0]
    pred_conf = top_predictions[0][1]
    return pred_label, pred_conf, top_predictions


def load_model(model_name: str):
    path = MODEL_PATHS[model_name]
    if not path.exists():
        st.error(f"Checkpoint not found: {path}")
        return None

    try:
        model, meta = load_cached_model(model_name)
    except Exception as e:
        st.error(str(e))
        return None

    st.session_state["model"] = model
    st.session_state["model_meta"] = meta
    st.session_state["loaded_model_name"] = model_name
    return model


def main():
    st.set_page_config(page_title="Music Genre Predictor", page_icon="🎵", layout="centered")
    st.title("🎵 Music Genre Predictor")
    st.caption("Upload a .wav music clip and predict its genre using one of your trained models.")

    model_type = st.selectbox("Select model", ["AST Model", "CRNN", "Secret Sauce"], index=0)
    uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

    predict_clicked = st.button("Predict genre", type="primary", use_container_width=True)

    if predict_clicked:
        if uploaded_file is None:
            st.warning("Please upload a .wav file first.")
            return

        with st.spinner("Loading model..."):
            model = load_model(model_type)

        if model is None:
            return

        uploaded_file.seek(0)

        with st.spinner("Running inference..."):
            pred_label, pred_conf, top_predictions = predict_genre(
                model_type,
                st.session_state["model"],
                st.session_state["model_meta"],
                uploaded_file,
            )

        st.success(f"Predicted genre: {pred_label}")
        st.metric("Confidence", f"{pred_conf * 100:.2f}%")
        st.subheader("Top predictions")
        st.table(
            [
                {"Genre": label, "Probability (%)": f"{prob * 100:.2f}"}
                for label, prob in top_predictions
            ]
        )


if __name__ == "__main__":
    main()
