import streamlit as st
import json
import uuid
import math
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Nexora Notes", page_icon="🌙", layout="wide")

DATA_FILE = Path("nexora_data.json")

categories = ["Work", "Personal", "Ideas", "Shopping", "Learning", "Other"]

category_colors = {
    "Work": "#B59ACF",
    "Personal": "#D98BA5",
    "Ideas": "#A78BC4",
    "Shopping": "#C77D9A",
    "Learning": "#E8DDED",
    "Other": "#9B829F"
}


def load_data():
    if not DATA_FILE.exists():
        return []

    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return []


def save_data(notes):
    with open(DATA_FILE, "w") as f:
        json.dump(notes, f, indent=4)


def delete_note(note_id):
    st.session_state.notes = [
        n for n in st.session_state.notes if n["id"] != note_id
    ]
    save_data(st.session_state.notes)


def show_message(message):
    st.markdown(
        f"""
        <div style="
            background-color: #2A1825;
            color: #F4EAF5;
            padding: 18px;
            border: 1px solid #75435D;
            border-radius: 10px;
            margin: 10px 0;
            font-size: 15px;
        ">
            {message}
        </div>
        """,
        unsafe_allow_html=True
    )


def calculate_similarity(notes):
    if len(notes) < 2:
        return []

    text_data = [f"{n['title']} {n['content']}" for n in notes]

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(text_data)
        sim_matrix = cosine_similarity(tfidf_matrix)
    except Exception:
        return []

    connections = []

    for i in range(len(notes)):
        for j in range(i + 1, len(notes)):
            score = sim_matrix[i][j]

            if score > 0.15:
                connections.append((i, j, round(float(score), 2)))

    return connections


def build_graph(notes, connections):
    fig = go.Figure()

    if not notes:
        return fig

    num_nodes = len(notes)

    x_positions = []
    y_positions = []
    z_positions = []

    for i in range(num_nodes):
        angle = (2 * math.pi * i) / num_nodes

        x_positions.append(math.cos(angle) * 5)
        y_positions.append(math.sin(angle) * 5)
        z_positions.append(math.sin(angle * 2) * 2)

    for idx_a, idx_b, score in connections:
        fig.add_trace(go.Scatter3d(
            x=[x_positions[idx_a], x_positions[idx_b]],
            y=[y_positions[idx_a], y_positions[idx_b]],
            z=[z_positions[idx_a], z_positions[idx_b]],
            mode="lines",
            line=dict(color="#75435D", width=4),
            hoverinfo="skip",
            showlegend=False
        ))

    for i, note in enumerate(notes):
        fig.add_trace(go.Scatter3d(
            x=[x_positions[i]],
            y=[y_positions[i]],
            z=[z_positions[i]],
            mode="markers+text",
            text=[note["title"]],
            textposition="top center",
            marker=dict(
                size=8,
                color=category_colors.get(note["category"], "#B59ACF"),
                line=dict(color="#E8DDED", width=1)
            ),
            hovertext=f"Category: {note['category']}<br>{note['content'][:50]}...",
            hoverinfo="text",
            showlegend=False
        ))

    fig.update_layout(
        height=600,
        paper_bgcolor="#160B12",
        plot_bgcolor="#160B12",
        font=dict(color="#E8DDED"),
        scene=dict(
            bgcolor="#160B12",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        margin=dict(l=10, r=10, t=10, b=10)
    )

    return fig


if "notes" not in st.session_state:
    st.session_state.notes = load_data()


st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #160B12, #2A101F, #160B12);
    color: #F4EAF5;
}

h1, h2, h3 {
    color: #E8DDED !important;
}

p, label {
    color: #E8DDED;
}

[data-testid="stSidebar"] {
    background-color: #210F1B;
}

.stButton button {
    background-color: #75435D;
    color: white;
    border: 1px solid #B59ACF;
    border-radius: 10px;
}

.stButton button:hover {
    background-color: #A78BC4;
    color: #160B12;
}

input, textarea {
    background-color: #2A1825 !important;
    color: white !important;
}

[data-baseweb="select"] {
    background-color: #2A1825 !important;
}

[data-testid="stMetric"] {
    background-color: #2A1825;
    border: 1px solid #75435D;
    padding: 15px;
    border-radius: 10px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #E8DDED !important;
}
</style>
""", unsafe_allow_html=True)


st.sidebar.title("Nexora 🌙")

view_mode = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Add New Note", "Browse Notes", "About App"]
)

st.sidebar.divider()

st.sidebar.markdown(
    f"""
    <div style="
        background-color: #2A1825;
        color: #F4EAF5;
        padding: 15px;
        border: 1px solid #75435D;
        border-radius: 10px;
        font-size: 15px;
    ">
        Total Notes Saved: {len(st.session_state.notes)}
    </div>
    """,
    unsafe_allow_html=True
)


if view_mode == "Dashboard":

    st.title("My Workspace")

    if not st.session_state.notes:
        show_message(
            "Your notebook is empty. Head over to 'Add New Note' to start."
        )

    else:
        all_connections = calculate_similarity(st.session_state.notes)

        st.subheader("Graph View")

        st.plotly_chart(
            build_graph(st.session_state.notes, all_connections),
            use_container_width=True
        )

        st.subheader("Quick Stats")

        col1, col2, col3 = st.columns(3)

        col1.metric("Notes", len(st.session_state.notes))

        col2.metric(
            "Categories Used",
            len(set(n["category"] for n in st.session_state.notes))
        )

        col3.metric("Discovered Links", len(all_connections))


elif view_mode == "Add New Note":

    st.title("Create Note")

    title_input = st.text_input("Note Title")
    content_input = st.text_area("Content", height=150)

    selected_cat = st.selectbox("Category", categories)

    note_date = st.date_input("Associated Date", value=None)

    if st.button("Save Entry", type="primary"):

        if not title_input.strip() or not content_input.strip():
            st.error("Both Title and Content fields are required.")

        else:
            new_note = {
                "id": str(uuid.uuid4()),
                "title": title_input.strip(),
                "content": content_input.strip(),
                "category": selected_cat,
                "date": str(note_date) if note_date else "",
                "timestamp": str(datetime.now())
            }

            st.session_state.notes.append(new_note)

            save_data(st.session_state.notes)

            st.success("Note added successfully!")
            st.rerun()


elif view_mode == "Browse Notes":

    st.title("Your Notebook")

    if not st.session_state.notes:
        show_message("Nothing to show yet.")

    else:
        search_query = st.text_input("Filter notes by keyword...")

        for note in reversed(st.session_state.notes):

            search_text = f"{note['title']} {note['content']} {note['category']}".lower()

            if search_query.lower() in search_text:

                with st.expander(f"{note['title']} ({note['category']})"):

                    st.write(note["content"])

                    if note["date"]:
                        st.caption(f"Target Date: {note['date']}")

                    st.button(
                        "Delete Note",
                        key=f"del_{note['id']}",
                        on_click=delete_note,
                        args=(note["id"],)
                    )

        st.divider()

        raw_json = json.dumps(st.session_state.notes, indent=4)

        st.download_button(
            "Export Notes (JSON)",
            data=raw_json,
            file_name="backup_notes.json",
            mime="application/json"
        )


elif view_mode == "About App":

    st.title("About Nexora")

    st.markdown("""
    Nexora is a space to keep notes, ideas and plans together.

    You can save different things and find notes that may be related to each other.

    It also has a 3D graph to show how your notes connect.

    Built using **Python, Streamlit, Plotly and Scikit-learn**.
    """)
