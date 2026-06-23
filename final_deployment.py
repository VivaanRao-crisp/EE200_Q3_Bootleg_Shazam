import os
import base64
import pickle
import tempfile

import librosa
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import fingerprint_engine as fe

st.set_page_config(page_title="Shazam from wish.com", page_icon="⚡️", layout="wide")


#nyan cat everywhere
def set_nyan_background():
    # embed the gif as base64 so it ships with the app and works offline / when deployed
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "nyan_cat.gif"), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        /* tile the animated nyan cat across the whole app + sidebar */
        [data-testid="stApp"], [data-testid="stSidebar"] {{
            background-image: url("data:image/gif;base64,{b64}");
            background-repeat: repeat;
            background-size: 160px;
        }}
        /* let the nyan cat show through the header bar too */
        [data-testid="stHeader"] {{ background: rgba(0, 0, 0, 0); }}
        /* keep the actual content readable: a solid dark card matching the dark theme */
        [data-testid="stMainBlockContainer"] {{
            background: rgba(14, 17, 23, 0.96);
            border-radius: 16px;
            padding: 2.5rem 3rem;
            margin-top: 1rem;
            box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.12);
        }}
        /* give the file-upload dropzone its own readable dark panel too */
        [data-testid="stFileUploaderDropzone"] {{
            background: rgba(14, 17, 23, 0.96);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_nyan_background()


# load the database once and keep it in memory (its ~60mb, dont reload every click)
@st.cache_resource
def load_db():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "database.pkl"), "rb") as f:
        return pickle.load(f)


master_db = load_db()


def no_ext(name):
    # prediction should be the song name without the .mp3 etc
    return os.path.splitext(name)[0]


def save_upload(uploaded_file):
    # librosa wants a real path so just dump the uploaded bytes into a temp file
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


# headers
st.title("🎸 😼 Sonic Signatures, our Bootleg Shazam ;)")
st.caption("EE200 Project Q3 by Vivaan Rao, 251192 and Mohith Lakshmana Sepur 250679")
st.write("a small Shazam clone, pipeline: spectrogram -> constellation peaks -> paired hashes -> offset voting.")

tab_single, tab_batch = st.tabs(["Single Clip Mode", "Batch Mode"])


# single clip mode
with tab_single:
    st.header("Single Clip Mode")
    st.write("upload one clip and see every step the algorithm goes through.")

    uploaded = st.file_uploader("upload a query clip", type=["mp3"], key="single")

    if uploaded is not None:
        path = save_upload(uploaded)
        st.audio(uploaded)

        # any length clip is accepted; we only ever fingerprint the first 60s of it.
        # this bounds the spectrogram size (rendering a huge one OOMs the 1GB server) while
        # letting users drop in full songs instead of pre-trimming to a short query.
        if librosa.get_duration(path=path) > 60:
            st.info("clip is longer than 60 s — using just the first 60 s for matching")

        # run the whole pipeline once and reuse the pieces for the plots
        try:
            # duration=60 makes librosa decode only the first 60s, so everything downstream stays bounded
            audio, sr = fe.load_audio(path, duration=60)
            frequencies, times, stft, stft_db = fe.compute_spectrogram(audio, sr)
            peak_freq_idx, peak_time_idx = fe.find_peaks(stft_db)
            query_hashes = fe.get_hashes(audio, sr)
            result = fe.match(query_hashes, master_db)
        except Exception:
            st.error("couldn't read this clip, try another mp3")
            st.stop()
        finally:
            os.remove(path)

        # the final answer, big and clear
        if result is None:
            st.markdown("## 💔 oh no ! there's no match in the database")
        else:
            st.markdown(f"## 🗿 nice song , ive heard it before, it's {no_ext(result['song'])}")
            st.write(f"starts ~{result['offset']} s into the track · {result['score']} aligned hashes")

        st.subheader("intermediate steps")
        c1, c2 = st.columns(2)

        # a) spectrogram
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.pcolormesh(times, frequencies, stft_db, shading="auto", cmap="inferno") #shading changed from gouraud to auto because gouraud creates an out of memory error on even slightly longer querier (It kept crashing on single mode when i was trying by a full upload of "back in the ussr" lol)
            ax.set_title("spectrogram")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("frequency (Hz)")
            ax.set_ylim([0, 8500])
            st.pyplot(fig)
            plt.close(fig)  # free the figure so it doesnt pile up in memory across reruns

        # b) constellation of peaks
        with c2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.pcolormesh(times, frequencies, stft_db, shading="auto", cmap="gray")
            ax.scatter(times[peak_time_idx], frequencies[peak_freq_idx], s=8, c="cyan")
            ax.set_title("constellation of peaks")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("frequency (Hz)")
            ax.set_ylim([0, 8500])
            st.pyplot(fig)
            plt.close(fig)  # free the figure so it doesnt pile up in memory across reruns

        # c) offset histogram (the thing that actually decides the winner)
        if result is not None:
            hist = result["offset_histogram"]
            fig, ax = plt.subplots(figsize=(14, 3))
            ax.bar(list(hist.keys()), list(hist.values()), width=0.1)
            ax.set_title(f"offset histogram for '{no_ext(result['song'])}' (one tall spike = real match)")
            ax.set_xlabel("offset (s)")
            ax.set_ylabel("aligned hashes")
            st.pyplot(fig)
            plt.close(fig)  # free the figure so it doesnt pile up in memory across reruns


# batch mode
with tab_batch:
    st.header("Batch Mode")
    st.write("upload many clips at once, identify them all, download results.csv")

    uploaded_files = st.file_uploader("upload query clips", type=["mp3"],accept_multiple_files=True, key="batch")

    # cap batch size: processing an unbounded number of clips on a 1GB server OOMs it
    if uploaded_files and len(uploaded_files) > 20:
        st.error(f"batch mode is capped at 20 files at once — you uploaded {len(uploaded_files)}. please upload 20 or fewer.")
        st.stop()

    if uploaded_files and st.button("identify all"):
        rows = []
        progress = st.progress(0.0)
        for i, uf in enumerate(uploaded_files):
            path = save_upload(uf)
            try:
                # cap each clip at the first 60s, same as single mode, so a long file cant OOM the server
                result = fe.identify(path, master_db, duration=60)
                # prediction = matched song filename without extension, or "No Match"
                prediction = no_ext(result["song"]) if result else "No Match"
            except Exception:
                prediction = "Error"
            finally:
                os.remove(path)
            rows.append({"filename": uf.name, "prediction": prediction})
            progress.progress((i + 1) / len(uploaded_files))

        # exactly two columns in this order: filename, prediction
        df = pd.DataFrame(rows, columns=["filename", "prediction"])
        st.dataframe(df, use_container_width=True)

        # download as results.csv, no pandas index
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("download results.csv", data=csv_bytes,file_name="results.csv", mime="text/csv")
