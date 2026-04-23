"""
Onboarding Digital — Validação de Documento
Aula 3 — OpenCV + TensorFlow/PyTorch | Caixa Econômica Federal

Rodar: streamlit run app_onboarding.py
"""

import streamlit as st
import cv2, os, time
import numpy as np
from PIL import Image, ImageDraw

st.set_page_config(page_title="Onboarding Caixa — CV", page_icon="🏦", layout="wide")


# ── Modelos (cache) ──
@st.cache_resource
def load_mobilenet_tf():
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    return MobileNetV2(weights="imagenet")


@st.cache_resource
def load_resnet_pt():
    from torchvision import models, transforms
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.eval()
    labels = models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return m, labels, tfm


def gerar_documento_sintetico():
    """Gera um RG fictício para demo."""
    img = Image.new("RGB", (800, 500), "#F5F5DC")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 780, 480], outline="#003366", width=3)
    draw.rectangle([20, 20, 780, 80], fill="#003366")
    draw.text((200, 35), "REPÚBLICA FEDERATIVA DO BRASIL", fill="white")
    draw.text((280, 55), "CARTEIRA DE IDENTIDADE", fill="#B0D4F1")
    for label, valor, y in [
        ("Nome:", "FULANO DE TAL SILVA", 120),
        ("CPF:", "123.456.789-00", 180),
        ("Data Nasc.:", "01/01/1990", 240),
        ("Naturalidade:", "BRASÍLIA - DF", 300),
        ("Filiação:", "BELTRANO SILVA / CICLANA TAL", 360),
    ]:
        draw.text((50, y), label, fill="#666666")
        draw.text((200, y), valor, fill="#000000")
    draw.rectangle([580, 120, 740, 320], outline="#999999", width=2)
    draw.text((620, 210), "FOTO", fill="#999999")
    draw.text((580, 340), "Assinatura", fill="#999999")
    return np.array(img)


# ── Header ──
st.markdown("""
<div style="background:#005CA9;padding:20px;border-radius:8px;margin-bottom:20px">
    <h1 style="color:white;margin:0">🏦 Onboarding Digital — Validação de Documento</h1>
    <p style="color:#B0D4F1;margin:0">Aula 3 — OpenCV + TensorFlow/PyTorch | Caixa Econômica Federal</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
tab_pipeline, tab_opencv, tab_vertices, tab_comparativo, tab_ocr, tab_ref = st.tabs(
    ["📋 Pipeline Onboarding", "🔧 Lab OpenCV", "📐 Vértices", "⚖️ TF vs PyTorch", "🔤 OCR (Extra)", "📚 Referência"]
)

# ══════════════════════════════════════════════════════════
# ABA 1 — PIPELINE COMPLETO DE ONBOARDING
# ══════════════════════════════════════════════════════════
with tab_pipeline:
    st.markdown("""
    **Fluxo real do onboarding no app da Caixa:**
    1. Cliente tira foto do RG com celular (iluminação ruim, torta)
    2. OpenCV pré-processa (equaliza, binariza, corrige)
    3. Modelo classifica o tipo de documento
    4. Sistema decide: aceita ou pede nova foto
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 Foto do documento")

        # Imagens de exemplo (RGs na pasta imagens/)
        IMAGENS_DIR = os.path.join(os.path.dirname(__file__), "imagens")
        exemplos_rg = sorted([f for f in os.listdir(IMAGENS_DIR) if f.endswith((".jpg", ".png"))]) if os.path.isdir(IMAGENS_DIR) else []

        fonte = st.radio("Fonte da imagem:", ["📁 Exemplos de RG", "📤 Upload", "🎨 Documento fictício"], horizontal=True, key="pipe_fonte")

        uploaded = None
        img_exemplo = None
        if fonte == "📁 Exemplos de RG" and exemplos_rg:
            escolhido = st.selectbox("Selecione o RG:", exemplos_rg, key="pipe_exemplo")
            img_exemplo = os.path.join(IMAGENS_DIR, escolhido)
            st.image(img_exemplo, caption=escolhido, width=300)
        elif fonte == "📤 Upload":
            uploaded = st.file_uploader("Foto do RG, CNH ou passaporte", type=["png", "jpg", "jpeg"], key="pipe_img")

        threshold = st.slider("Threshold de aceitação", 0.3, 0.95, 0.5, 0.05, key="pipe_thresh")
        btn = st.button("🔍 Processar documento", type="primary", use_container_width=True, key="pipe_btn")

    with col2:
        if btn:
            # Carregar imagem
            if fonte == "📁 Exemplos de RG" and img_exemplo:
                img_bgr = cv2.imread(img_exemplo)
            elif fonte == "📤 Upload" and uploaded:
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            else:
                img_bgr = cv2.cvtColor(gerar_documento_sintetico(), cv2.COLOR_RGB2BGR)

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # ── Etapa 1: Pré-processamento ──
            st.subheader("1️⃣ Pré-processamento (OpenCV)")
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalizado = clahe.apply(gray)
            blur = cv2.GaussianBlur(equalizado, (3, 3), 0)
            _, binarizado = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            tabs_pre = st.tabs(["Original", "CLAHE", "Blur", "Binarizado"])
            with tabs_pre[0]:
                st.image(img_rgb, use_container_width=True)
            with tabs_pre[1]:
                st.image(equalizado, use_container_width=True, clamp=True)
            with tabs_pre[2]:
                st.image(blur, use_container_width=True, clamp=True)
            with tabs_pre[3]:
                st.image(binarizado, use_container_width=True, clamp=True)

            # ── Etapa 2: Classificação ──
            st.subheader("2️⃣ Classificação (MobileNetV2)")
            import torch
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

            with st.spinner("Classificando..."):
                modelo_tf = load_mobilenet_tf()
                # Classificar imagem original
                img224 = cv2.resize(img_rgb, (224, 224))
                x = preprocess_input(np.expand_dims(img224.astype("float32"), axis=0))
                t0 = time.time()
                preds_orig = modelo_tf.predict(x, verbose=0)
                t_orig = time.time() - t0

                # Classificar imagem pré-processada
                prep_rgb = cv2.cvtColor(cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
                img224p = cv2.resize(prep_rgb, (224, 224))
                xp = preprocess_input(np.expand_dims(img224p.astype("float32"), axis=0))
                t0 = time.time()
                preds_prep = modelo_tf.predict(xp, verbose=0)
                t_prep = time.time() - t0

            top_orig = decode_predictions(preds_orig, top=3)[0]
            top_prep = decode_predictions(preds_prep, top=3)[0]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Sem pré-processamento:**")
                for _, nome, conf in top_orig:
                    st.progress(float(conf), text=f"{nome}: {conf:.1%}")
                st.caption(f"⏱️ {t_orig*1000:.0f} ms")
            with c2:
                st.markdown("**Com pré-processamento:**")
                for _, nome, conf in top_prep:
                    st.progress(float(conf), text=f"{nome}: {conf:.1%}")
                st.caption(f"⏱️ {t_prep*1000:.0f} ms")

            # ── Etapa 3: Decisão ──
            st.subheader("3️⃣ Decisão")
            melhor_conf = float(top_prep[0][2])
            if melhor_conf >= threshold:
                st.success(f"✅ **DOCUMENTO ACEITO** — {top_prep[0][1]} ({melhor_conf:.1%}) acima do threshold ({threshold:.0%})")
            else:
                st.error(f"❌ **PEDIR NOVA FOTO** — confiança {melhor_conf:.1%} abaixo do threshold ({threshold:.0%})")

            st.info("💡 Em produção: após aceitar, o próximo passo seria OCR (EasyOCR/Tesseract) para extrair nome, CPF e data.")

# ══════════════════════════════════════════════════════════
# ABA 2 — LAB OPENCV INTERATIVO
# ══════════════════════════════════════════════════════════
with tab_opencv:
    st.subheader("🔧 Lab OpenCV — Ajuste os parâmetros ao vivo")
    st.markdown("Suba uma imagem ou use o documento demo e veja como cada operação afeta o resultado.")

    col_cfg, col_result = st.columns([1, 2])

    with col_cfg:
        IMAGENS_DIR = os.path.join(os.path.dirname(__file__), "imagens")
        exemplos_rg = sorted([f for f in os.listdir(IMAGENS_DIR) if f.endswith((".jpg", ".png"))]) if os.path.isdir(IMAGENS_DIR) else []

        fonte_cv = st.radio("Fonte:", ["📁 Exemplos de RG", "📤 Upload", "🎨 Demo"], horizontal=True, key="cv_fonte")
        if fonte_cv == "📁 Exemplos de RG" and exemplos_rg:
            escolhido_cv = st.selectbox("RG:", exemplos_rg, key="cv_exemplo")
            img_cv = cv2.imread(os.path.join(IMAGENS_DIR, escolhido_cv))
        elif fonte_cv == "📤 Upload":
            uploaded_cv = st.file_uploader("Imagem", type=["png", "jpg", "jpeg"], key="cv_img")
            if uploaded_cv:
                file_bytes = np.asarray(bytearray(uploaded_cv.read()), dtype=np.uint8)
                img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            else:
                img_cv = cv2.cvtColor(gerar_documento_sintetico(), cv2.COLOR_RGB2BGR)
        else:
            img_cv = cv2.cvtColor(gerar_documento_sintetico(), cv2.COLOR_RGB2BGR)

        st.markdown("### Parâmetros")
        blur_k = st.slider("Blur kernel", 1, 31, 5, 2, key="cv_blur")
        canny_low = st.slider("Canny threshold baixo", 0, 255, 50, key="cv_canny_low")
        canny_high = st.slider("Canny threshold alto", 0, 255, 150, key="cv_canny_high")
        clahe_clip = st.slider("CLAHE clip limit", 1.0, 10.0, 2.0, 0.5, key="cv_clahe")
        morph_iter = st.slider("Morfologia iterações", 0, 5, 2, key="cv_morph")
        operacao = st.selectbox("Operação em destaque", [
            "Todas (grid)", "Grayscale", "Blur", "CLAHE", "Canny", "Otsu", "Morfologia (close)"
        ], key="cv_op")

    with col_result:
        gray_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blur_cv = cv2.GaussianBlur(gray_cv, (blur_k, blur_k), 0)
        clahe_obj = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        clahe_cv = clahe_obj.apply(gray_cv)
        canny_cv = cv2.Canny(blur_cv, canny_low, canny_high)
        _, otsu_cv = cv2.threshold(gray_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        morph_cv = cv2.morphologyEx(otsu_cv, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)

        if operacao == "Todas (grid)":
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 4, figsize=(14, 6))
            rgb_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            dados = [
                (rgb_cv, "Original"), (gray_cv, "Cinza"), (blur_cv, "Blur"),
                (clahe_cv, "CLAHE"), (canny_cv, "Canny"), (otsu_cv, "Otsu"),
                (morph_cv, "Morfologia"),
            ]
            for ax, (im, t) in zip(axes.flat, dados):
                ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
                ax.set_title(t, fontweight="bold")
                ax.axis("off")
            axes.flat[-1].axis("off")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            mapa = {
                "Grayscale": gray_cv, "Blur": blur_cv, "CLAHE": clahe_cv,
                "Canny": canny_cv, "Otsu": otsu_cv, "Morfologia (close)": morph_cv,
            }
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original**")
                st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_container_width=True)
            with c2:
                st.markdown(f"**{operacao}**")
                st.image(mapa[operacao], use_container_width=True, clamp=True)

        # Histograma
        import matplotlib.pyplot as plt
        fig_h, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        ax1.hist(gray_cv.ravel(), 256, [0, 256], color="#005CA9", alpha=0.7)
        ax1.set_title("Histograma original")
        ax2.hist(clahe_cv.ravel(), 256, [0, 256], color="#238636", alpha=0.7)
        ax2.set_title("Histograma após CLAHE")
        for ax in (ax1, ax2):
            ax.set_xlim([0, 256])
        plt.tight_layout()
        st.pyplot(fig_h)

# ══════════════════════════════════════════════════════════
# ABA 3 — DETECÇÃO DE VÉRTICES DO DOCUMENTO
# ══════════════════════════════════════════════════════════
with tab_vertices:
    st.subheader("📐 Encontrando os vértices do documento na foto")
    st.markdown("""
    **Técnica:** para cada canto da imagem, encontrar o ponto detectado pelo Canny
    que está mais próximo. Esses 4 pontos são os vértices do documento.

    > Baseado em: [Encontrando os vértices de um documento em uma foto](https://www.tabnews.com.br/jrdutra/encontrando-os-vertices-de-um-documento-em-uma-foto-com-python)

    **Fórmula:** `distância = √((x₂-x₁)² + (y₂-y₁)²)` — distância euclidiana de cada ponto ao canto.
    """)

    IMAGENS_DIR = os.path.join(os.path.dirname(__file__), "imagens")
    exemplos_rg = sorted([f for f in os.listdir(IMAGENS_DIR) if f.endswith((".jpg", ".png"))]) if os.path.isdir(IMAGENS_DIR) else []

    col_v1, col_v2 = st.columns([1, 2])

    with col_v1:
        fonte_v = st.radio("Fonte:", ["📁 Exemplos de RG", "📤 Upload"], horizontal=True, key="v_fonte")
        if fonte_v == "📁 Exemplos de RG" and exemplos_rg:
            escolhido_v = st.selectbox("RG:", exemplos_rg, key="v_exemplo")
            img_v = cv2.imread(os.path.join(IMAGENS_DIR, escolhido_v))
        else:
            uploaded_v = st.file_uploader("Imagem", type=["png", "jpg", "jpeg"], key="v_img")
            if uploaded_v:
                file_bytes = np.asarray(bytearray(uploaded_v.read()), dtype=np.uint8)
                img_v = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            else:
                img_v = cv2.cvtColor(gerar_documento_sintetico(), cv2.COLOR_RGB2BGR)

        st.markdown("### Parâmetros")
        v_blur = st.slider("Blur kernel", 1, 31, 5, 2, key="v_blur")
        v_canny_low = st.slider("Canny baixo", 0, 255, 50, key="v_canny_low")
        v_canny_high = st.slider("Canny alto", 0, 255, 150, key="v_canny_high")
        btn_v = st.button("📐 Detectar vértices", type="primary", use_container_width=True, key="v_btn")

    with col_v2:
        if btn_v:
            import matplotlib.pyplot as plt

            h, w = img_v.shape[:2]
            rgb_v = cv2.cvtColor(img_v, cv2.COLOR_BGR2RGB)

            # Passo 1: Grayscale + Blur
            gray_v = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)
            blur_v = cv2.GaussianBlur(gray_v, (v_blur, v_blur), 0)

            # Passo 2: Canny (detecção de bordas)
            edges = cv2.Canny(blur_v, v_canny_low, v_canny_high)

            # Passo 3: Encontrar pontos das bordas
            pontos = np.column_stack(np.where(edges > 0))  # (y, x)
            pontos_xy = pontos[:, ::-1]  # converter para (x, y)

            # Passo 4: Para cada canto da imagem, achar o ponto mais próximo
            cantos = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
            vertices = []
            for canto in cantos:
                dists = np.sqrt(np.sum((pontos_xy - canto) ** 2, axis=1))
                idx_min = np.argmin(dists)
                vertices.append(pontos_xy[idx_min])
            vertices = np.array(vertices, dtype="float32")

            # Passo 5: Transformação de perspectiva
            # Ordenar: top-left, top-right, bottom-right, bottom-left
            s = vertices.sum(axis=1)
            d = np.diff(vertices, axis=1).ravel()
            ordered = np.array([
                vertices[s.argmin()],   # top-left
                vertices[d.argmin()],   # top-right
                vertices[s.argmax()],   # bottom-right
                vertices[d.argmax()],   # bottom-left
            ], dtype="float32")

            # Calcular dimensões do retângulo destino
            w_top = np.linalg.norm(ordered[1] - ordered[0])
            w_bot = np.linalg.norm(ordered[2] - ordered[3])
            h_left = np.linalg.norm(ordered[3] - ordered[0])
            h_right = np.linalg.norm(ordered[2] - ordered[1])
            new_w, new_h = int(max(w_top, w_bot)), int(max(h_left, h_right))

            dst = np.array([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]], dtype="float32")
            M = cv2.getPerspectiveTransform(ordered, dst)
            warped = cv2.warpPerspective(img_v, M, (new_w, new_h))

            # ── Visualização dos 5 passos ──
            st.markdown("### Passo a passo")

            # Passos 1-3
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].imshow(rgb_v); axes[0].set_title("1. Original")
            axes[1].imshow(gray_v, cmap="gray"); axes[1].set_title("2. Cinza + Blur")
            axes[2].imshow(edges, cmap="gray"); axes[2].set_title(f"3. Canny ({len(pontos_xy)} pontos)")
            for ax in axes: ax.axis("off")
            plt.tight_layout(); st.pyplot(fig)

            # Passo 4: vértices marcados
            fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

            img_marcado = rgb_v.copy()
            cores = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            nomes = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
            for i, (pt, cor, nome) in enumerate(zip(ordered, cores, nomes)):
                x_pt, y_pt = int(pt[0]), int(pt[1])
                cv2.circle(img_marcado, (x_pt, y_pt), 12, cor, -1)
                cv2.putText(img_marcado, nome, (x_pt + 15, y_pt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
            # Desenhar polígono
            pts_poly = ordered.astype(int).reshape((-1, 1, 2))
            cv2.polylines(img_marcado, [pts_poly], True, (0, 255, 0), 2)

            axes2[0].imshow(img_marcado)
            axes2[0].set_title("4. Vértices detectados")
            axes2[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            axes2[1].set_title("5. Documento corrigido (perspectiva)")
            for ax in axes2: ax.axis("off")
            plt.tight_layout(); st.pyplot(fig2)

            # Coordenadas
            st.markdown("### Coordenadas dos vértices")
            for nome, pt in zip(nomes, ordered):
                st.markdown(f"- **{nome}:** ({int(pt[0])}, {int(pt[1])})")

            st.info(f"💡 Imagem original: {w}×{h} → Documento corrigido: {new_w}×{new_h}")
            st.markdown("""
            **Como funciona:**
            1. Canny detecta bordas → milhares de pontos
            2. Para cada canto da imagem, calcula `√((x₂-x₁)² + (y₂-y₁)²)` até todos os pontos
            3. O ponto mais próximo de cada canto = vértice do documento
            4. `getPerspectiveTransform` + `warpPerspective` = documento "escaneado"
            """)

# ══════════════════════════════════════════════════════════
# ABA 4 — COMPARATIVO TF vs PYTORCH
# ══════════════════════════════════════════════════════════
with tab_comparativo:
    st.subheader("⚖️ TensorFlow vs PyTorch — Mesma imagem, dois frameworks")

    IMAGENS_DIR = os.path.join(os.path.dirname(__file__), "imagens")
    exemplos_rg = sorted([f for f in os.listdir(IMAGENS_DIR) if f.endswith((".jpg", ".png"))]) if os.path.isdir(IMAGENS_DIR) else []

    fonte_cmp = st.radio("Fonte:", ["📁 Exemplos de RG", "📤 Upload", "🎨 Demo"], horizontal=True, key="cmp_fonte")
    if fonte_cmp == "📁 Exemplos de RG" and exemplos_rg:
        escolhido_cmp = st.selectbox("RG:", exemplos_rg, key="cmp_exemplo")
        img_cmp = cv2.imread(os.path.join(IMAGENS_DIR, escolhido_cmp))
    elif fonte_cmp == "📤 Upload":
        uploaded_cmp = st.file_uploader("Imagem para comparar", type=["png", "jpg", "jpeg"], key="cmp_img")
        if uploaded_cmp:
            file_bytes = np.asarray(bytearray(uploaded_cmp.read()), dtype=np.uint8)
            img_cmp = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            img_cmp = cv2.cvtColor(gerar_documento_sintetico(), cv2.COLOR_RGB2BGR)
    else:
        img_cmp = cv2.cvtColor(gerar_documento_sintetico(), cv2.COLOR_RGB2BGR)

    img_cmp_rgb = cv2.cvtColor(img_cmp, cv2.COLOR_BGR2RGB)
    preproc = st.checkbox("Aplicar pré-processamento OpenCV antes", value=True, key="cmp_preproc")

    if st.button("🚀 Classificar nos dois frameworks", type="primary", use_container_width=True, key="cmp_btn"):
        import torch
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

        # Pré-processamento opcional
        if preproc:
            gray_cmp = cv2.cvtColor(img_cmp, cv2.COLOR_BGR2GRAY)
            clahe_cmp = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_cmp)
            blur_cmp = cv2.GaussianBlur(clahe_cmp, (3, 3), 0)
            img_para_modelo = cv2.cvtColor(cv2.cvtColor(blur_cmp, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        else:
            img_para_modelo = img_cmp_rgb

        col_tf, col_pt = st.columns(2)

        # ── TensorFlow ──
        with col_tf:
            st.markdown("### 🟠 TensorFlow (MobileNetV2)")
            with st.spinner("Carregando MobileNetV2..."):
                modelo_tf = load_mobilenet_tf()
                img224 = cv2.resize(img_para_modelo, (224, 224))
                x = preprocess_input(np.expand_dims(img224.astype("float32"), axis=0))
                t0 = time.time()
                preds = modelo_tf.predict(x, verbose=0)
                t_tf = time.time() - t0
                top5_tf = decode_predictions(preds, top=5)[0]

            st.metric("Tempo", f"{t_tf*1000:.0f} ms")
            st.metric("Parâmetros", "3.4M")
            for _, nome, conf in top5_tf:
                st.progress(float(conf), text=f"{nome}: {conf:.1%}")

        # ── PyTorch ──
        with col_pt:
            st.markdown("### 🔵 PyTorch (ResNet18)")
            with st.spinner("Carregando ResNet18..."):
                modelo_pt, labels_pt, tfm_pt = load_resnet_pt()
                pil = Image.fromarray(img_para_modelo)
                x = tfm_pt(pil).unsqueeze(0)
                t0 = time.time()
                with torch.no_grad():
                    out = modelo_pt(x)
                    probs = torch.softmax(out, dim=1)[0]
                t_pt = time.time() - t0
                top5_pt = torch.topk(probs, 5)

            st.metric("Tempo", f"{t_pt*1000:.0f} ms")
            st.metric("Parâmetros", "11.7M")
            for p, i in zip(top5_pt.values, top5_pt.indices):
                st.progress(float(p), text=f"{labels_pt[int(i)]}: {float(p):.1%}")

        # ── Tabela comparativa ──
        st.divider()
        st.markdown("### Comparativo")
        st.markdown(f"""
| Aspecto | TensorFlow (MobileNetV2) | PyTorch (ResNet18) |
|---|---|---|
| **Top-1** | {top5_tf[0][1]} ({top5_tf[0][2]:.1%}) | {labels_pt[int(top5_pt.indices[0])]} ({float(top5_pt.values[0]):.1%}) |
| **Tempo** | {t_tf*1000:.0f} ms | {t_pt*1000:.0f} ms |
| **Parâmetros** | 3.4M | 11.7M |
| **Deploy mobile** | TF Lite (maduro) | ExecuTorch (novo) |
| **Pré-processamento** | `preprocess_input()` | `transforms.Compose()` |
        """)

        st.info("💡 **Regra prática Caixa:** Google Cloud → TF. Hugging Face → PyTorch. Mobile → TF Lite.")

# ══════════════════════════════════════════════════════════
# ABA 5 — OCR (EXTRA)
# ══════════════════════════════════════════════════════════
with tab_ocr:
    st.subheader("🔤 OCR — Extrair texto do documento (Extra)")
    st.markdown("""
    **Pipeline completo de onboarding:** OpenCV pré-processa → modelo classifica → **OCR extrai texto**.

    Usamos **keras-ocr** (baseado em TensorFlow): detecta onde está o texto (CRAFT) e reconhece os caracteres (CRNN).
    Tudo roda local, sem API externa.
    """)

    IMAGENS_DIR = os.path.join(os.path.dirname(__file__), "imagens")
    exemplos_rg = sorted([f for f in os.listdir(IMAGENS_DIR) if f.endswith((".jpg", ".png"))]) if os.path.isdir(IMAGENS_DIR) else []

    col_ocr1, col_ocr2 = st.columns([1, 2])

    with col_ocr1:
        fonte_ocr = st.radio("Fonte:", ["📁 Exemplos de RG", "📤 Upload"], horizontal=True, key="ocr_fonte")
        if fonte_ocr == "📁 Exemplos de RG" and exemplos_rg:
            escolhido_ocr = st.selectbox("RG:", exemplos_rg, key="ocr_exemplo")
            img_ocr_path = os.path.join(IMAGENS_DIR, escolhido_ocr)
        else:
            uploaded_ocr = st.file_uploader("Imagem", type=["png", "jpg", "jpeg"], key="ocr_img")
            img_ocr_path = None
            if uploaded_ocr:
                import tempfile
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                tmp.write(uploaded_ocr.read())
                tmp.close()
                img_ocr_path = tmp.name

        preproc_ocr = st.checkbox("Aplicar pré-processamento OpenCV antes do OCR", value=True, key="ocr_preproc")
        corrigir_persp = st.checkbox("Corrigir perspectiva (vértices)", value=False, key="ocr_persp")
        btn_ocr = st.button("🔍 Extrair texto", type="primary", use_container_width=True, key="ocr_btn")

    with col_ocr2:
        if btn_ocr and img_ocr_path:
            try:
                import keras_ocr
            except ImportError:
                st.warning("Instalando keras-ocr (primeira vez, ~30s)...")
                import subprocess
                subprocess.check_call(["pip", "install", "-q", "keras-ocr"])
                import keras_ocr

            img_bgr = cv2.imread(img_ocr_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Pré-processamento opcional
            if preproc_ocr or corrigir_persp:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3, 3), 0)

                if corrigir_persp:
                    h, w = img_bgr.shape[:2]
                    edges = cv2.Canny(blur, 50, 150)
                    pontos = np.column_stack(np.where(edges > 0))[:, ::-1]
                    if len(pontos) > 0:
                        cantos = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
                        verts = np.array([pontos[np.argmin(np.sqrt(np.sum((pontos - c) ** 2, axis=1)))] for c in cantos], dtype="float32")
                        sv = verts.sum(axis=1); dv = np.diff(verts, axis=1).ravel()
                        ordered = np.array([verts[sv.argmin()], verts[dv.argmin()], verts[sv.argmax()], verts[dv.argmax()]], dtype="float32")
                        nw = int(max(np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3])))
                        nh = int(max(np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1])))
                        dst = np.array([[0, 0], [nw, 0], [nw, nh], [0, nh]], dtype="float32")
                        img_bgr = cv2.warpPerspective(img_bgr, cv2.getPerspectiveTransform(ordered, dst), (nw, nh))
                        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                        blur = cv2.GaussianBlur(gray, (3, 3), 0)

                if preproc_ocr:
                    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe_obj.apply(blur)
                    img_para_ocr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
                else:
                    img_para_ocr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                img_para_ocr = img_rgb

            # OCR
            with st.spinner("Executando OCR (CRAFT + CRNN)..."):
                pipeline = keras_ocr.pipeline.Pipeline()
                t0 = time.time()
                predictions = pipeline.recognize([img_para_ocr])
                elapsed = time.time() - t0

            st.success(f"OCR concluído em {elapsed:.1f}s — {len(predictions[0])} palavras detectadas")

            # Visualização com bounding boxes
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            axes[0].imshow(img_para_ocr)
            axes[0].set_title("Imagem processada", fontweight="bold")
            axes[0].axis("off")

            # Desenhar boxes
            img_boxes = img_para_ocr.copy()
            for text, box in predictions[0]:
                box = np.array(box).astype(int)
                cv2.polylines(img_boxes, [box], True, (0, 255, 0), 2)
                cv2.putText(img_boxes, text, (box[0][0], box[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            axes[1].imshow(img_boxes)
            axes[1].set_title(f"Texto detectado ({len(predictions[0])} palavras)", fontweight="bold")
            axes[1].axis("off")
            plt.tight_layout()
            st.pyplot(fig)

            # Texto extraído
            st.markdown("### Texto extraído")
            palavras = [text for text, box in predictions[0]]
            # Ordenar por posição Y (linha) depois X (coluna)
            items = [(text, box) for text, box in predictions[0]]
            items.sort(key=lambda x: (int(np.mean(x[1][:, 1]) / 30) * 30, np.mean(x[1][:, 0])))
            linhas = []
            linha_atual = []
            y_atual = -1
            for text, box in items:
                y = int(np.mean(box[:, 1]) / 30) * 30
                if y != y_atual and linha_atual:
                    linhas.append(" ".join(linha_atual))
                    linha_atual = []
                y_atual = y
                linha_atual.append(text)
            if linha_atual:
                linhas.append(" ".join(linha_atual))

            st.code("\n".join(linhas), language=None)

            st.markdown(f"**Total:** {len(palavras)} palavras em {len(linhas)} linhas")
            st.info("💡 keras-ocr usa CRAFT (detecção) + CRNN (reconhecimento), ambos baseados em TensorFlow/Keras. Para português, EasyOCR ou Tesseract podem dar resultados melhores.")

        elif btn_ocr:
            st.warning("Selecione uma imagem.")

# ══════════════════════════════════════════════════════════
# ABA 6 — REFERÊNCIA
# ══════════════════════════════════════════════════════════
with tab_ref:
    st.subheader("📚 Pipeline canônico de Visão Computacional")
    st.markdown("""
    > **70% do resultado vem do pré-processamento, não do modelo.**

    ```
    Aquisição → Pré-processamento → Features → Modelo → Pós-processamento
    (câmera)    (OpenCV)            (CNN)      (classificação)  (regras de negócio)
    ```
    """)

    st.markdown("### 🖼️ Operações OpenCV — Quando usar cada uma")
    st.markdown("""
| Operação | Para quê | Quando usar |
|----------|---------|-------------|
| **Grayscale** | Remove cor (irrelevante para texto) | Sempre, primeiro passo |
| **Gaussian Blur** | Remove ruído da câmera | Antes de Canny/Sobel |
| **CLAHE** | Equaliza contraste (iluminação desigual) | Fotos de celular |
| **Sobel** | Gradiente direcional (bordas H/V) | Quando precisa direção |
| **Canny** | Bordas finas e limpas | Encontrar contorno do documento |
| **Otsu** | Binarização automática | Separar texto de fundo |
| **Adaptive Threshold** | Binarização com iluminação desigual | Fotos reais (sombra) |
| **Morfologia (close)** | Fecha buracos, conecta letras | Antes do OCR |
| **Morfologia (open)** | Remove pontos soltos | Limpar após binarização |
    """)

    st.divider()
    st.markdown("### 🤖 Modelos usados")
    st.markdown("""
| Modelo | Framework | Parâmetros | Tamanho | Uso |
|---|---|---|---|---|
| **MobileNetV2** | TensorFlow/Keras | 3.4M | ~14 MB | Classificação leve (mobile) |
| **ResNet18** | PyTorch/torchvision | 11.7M | ~44 MB | Classificação mais precisa |
    """)

    st.markdown("""
    ### 🔗 Links úteis
    - [OpenCV docs](https://docs.opencv.org/4.x/)
    - [TF MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)
    - [PyTorch torchvision models](https://pytorch.org/vision/stable/models.html)
    - [TF Lite para mobile](https://www.tensorflow.org/lite)
    """)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### 📎 Links da Aula")
    st.markdown("[📓 Notebook no Colab](https://colab.research.google.com/drive/10qAFzuWOb2BFvmzZhNI9fEQZY2ryxXD-)")
    st.markdown("[👨‍🏫 Prof. Itamar — LinkedIn](https://www.linkedin.com/in/itamusic/)")
    st.divider()
    st.markdown("### Caso Caixa — Onboarding")
    st.markdown("""
    **Fluxo no app:**
    1. 📸 Foto do RG
    2. 🔧 OpenCV pré-processa
    3. 🤖 Modelo classifica
    4. ✅ Aceita ou ❌ pede nova foto

    **Regra dos 70%:**
    O pré-processamento importa mais que o modelo.
    """)
    st.divider()
    st.markdown("### Sobre as abas")
    st.markdown("""
    - **Pipeline**: fluxo completo de onboarding
    - **Lab OpenCV**: ajuste parâmetros ao vivo
    - **TF vs PyTorch**: comparação lado a lado
    - **Referência**: operações, modelos, links
    """)
