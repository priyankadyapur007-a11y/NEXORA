
import streamlit as st
import json
import uuid
import math
from pathlib import Path
from datetime import date
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="NEXORA",
    page_icon="🌙",
    layout="wide"
)

DATA_FILE = Path("nexora_data.json")

categories = [
    "Work",
    "Personal",
    "Ideas",
    "Shopping",
    "Learning",
    "Other"
]

category_colors = {
    "Work": "#B59ACF",
    "Personal": "#D98BA5",
    "Ideas": "#A78BC4",
    "Shopping": "#C77D9A",
    "Learning": "#E8DDED",
    "Other": "#9B829F"
}

st.markdown("""
<style>
.stApp {
    background-color: #160B12;
    color: #E8DDED;
}

[data-testid="stSidebar"] {
    background-color: #210F1B;
}

h1, h2, h3, p, label {
    color: #E8DDED;
}

.stButton > button {
    background-color: #75435D;
    color: white;
    border: none;
    border-radius: 8px;
}

.stButton > button:hover {
    background-color: #A78BC4;
    color: white;
}

.stTextInput input,
.stTextArea textarea,
.stSelectbox div,
.stDateInput input {
    background-color: #2A1825;
    color: #E8DDED;
}

[data-testid="stMetric"] {
    background-color: #2A1825;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


def load_data():
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except:
            return []
    return []


def save_data(notes):
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(notes, file, indent=4)


def show_message(message):
    st.markdown(
        f"""
        <div style="
            background-color: #2A1825;
            color: #F4EAF5;
            border: 1px solid #75435D;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        ">
            {message}
        </div>
        """,
        unsafe_allow_html=True
    )


def calculate_similarity(notes):
    connections = []

    if len(notes) < 2:
        return connections

    note_texts = []

    for note in notes:
        text = note["title"] + " " + note["content"]
        note_texts.append(text)

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        note_vectors = vectorizer.fit_transform(note_texts)

        similarity = cosine_similarity(note_vectors)

        for i in range(len(notes)):
            for j in range(i + 1, len(notes)):
                score = similarity[i][j]

                if score > 0.15:
                    connections.append((i, j, round(float(score), 2)))

    except ValueError:
        return connections

    return connections


def build_graph(notes, connections):
    fig = go.Figure()

    if not notes:
        return fig

    total_notes = len(notes)

    x_positions = []
    y_positions = []
    z_positions = []

    for i in range(total_notes):
        angle = i * 2.399963
        height = 1 - (2 * (i + 0.5) / total_notes)
        radius = math.sqrt(1 - height * height)

        x_positions.append(math.cos(angle) * radius * 6)
        y_positions.append(math.sin(angle) * radius * 6)
        z_positions.append(height * 6)

    for first, second, score in connections:
        fig.add_trace(go.Scatter3d(
            x=[
                x_positions[first],
                x_positions[second]
            ],
            y=[
                y_positions[first],
                y_positions[second]
            ],
            z=[
                z_positions[first],
                z_positions[second]
            ],
            mode="lines",
            line=dict(
                color="#75435D",
                width=5
            ),
            hoverinfo="skip",
            showlegend=False
        ))

    note_colors = [
        category_colors.get(note["category"], "#B59ACF")
        for note in notes
    ]

    note_titles = [
        note["title"]
        for note in notes
    ]

    note_details = [
        "Category: " + note["category"] +
        "<br>" + note["content"][:120]
        for note in notes
    ]

    fig.add_trace(go.Scatter3d(
        x=x_positions,
        y=y_positions,
        z=z_positions,
        mode="markers+text",
        text=note_titles,
        textposition="top center",
        hovertext=note_details,
        hoverinfo="text",
        marker=dict(
            size=11,
            color=note_colors,
            line=dict(
                color="#E8DDED",
                width=2
            ),
            opacity=0.95
        ),
        showlegend=False
    ))

    fig.update_layout(
        height=700,
        paper_bgcolor="#160B12",
        font=dict(
            color="#E8DDED",
            size=12
        ),
        margin=dict(
            l=0,
            r=0,
            t=20,
            b=0
        ),
        scene=dict(
            bgcolor="#160B12",
            xaxis=dict(
                visible=False,
                showbackground=False
            ),
            yaxis=dict(
                visible=False,
                showbackground=False
            ),
            zaxis=dict(
                visible=False,
                showbackground=False
            ),
            aspectmode="cube",
            dragmode="orbit",
            camera=dict(
                eye=dict(
                    x=1.8,
                    y=1.8,
                    z=1.5
                )
            )
        ),
        showlegend=False
    )

    return fig


if "notes" not in st.session_state:
    st.session_state.notes = load_data()


notes = st.session_state.notes


st.sidebar.title("🌙 NEXORA")
st.sidebar.caption("Your thoughts, connected.")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Add New Note",
        "Browse Notes",
        "About App"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"""
    <div style="
        background-color: #2A1825;
        padding: 15px;
        border-radius: 10px;
        color: #E8DDED;
    ">
        <h3 style="margin: 0;">{len(notes)}</h3>
        <p style="margin: 0;">Saved Notes</p>
    </div>
    """,
    unsafe_allow_html=True
)


if page == "Dashboard":

    st.title("My Workspace")
    st.write("A space where all your thoughts come together.")

    if len(notes) == 0:
        show_message("You haven't added any notes yet. Start by adding your first note.")
    else:
        connections = calculate_similarity(notes)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Notes", len(notes))

        used_categories = len(set(
            note["category"] for note in notes
        ))

        col2.metric("Categories Used", used_categories)
        col3.metric("Discovered Links", len(connections))

        st.markdown("---")

        st.subheader("Your 3D Thought Network")

        st.caption(
            "Drag the graph with your mouse to rotate it. "
            "Scroll to zoom in and out."
        )

        graph = build_graph(notes, connections)

        st.plotly_chart(
            graph,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displayModeBar": True
            }
        )

        st.markdown("---")
        st.subheader("Category Colours")

        for category, color in category_colors.items():
            st.markdown(
                f"""
                <span style="
                    display: inline-block;
                    background-color: {color};
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 6px;
                "></span>
                <span style="color: #E8DDED; margin-right: 15px;">
                    {category}
                </span>
                """,
                unsafe_allow_html=True
            )


elif page == "Add New Note":

    st.title("Add a New Note")
    st.write("Save an idea, thought, plan or anything you want.")

    with st.form("new_note_form"):

        title = st.text_input("Note Title")

        content = st.text_area(
            "What's on your mind?",
            height=180
        )

        category = st.selectbox(
            "Choose Category",
            categories
        )

        note_date = st.date_input(
            "Associated Date",
            value=date.today()
        )

        save_button = st.form_submit_button("Save Note")

        if save_button:

            if title.strip() and content.strip():

                new_note = {
                    "id": str(uuid.uuid4()),
                    "title": title.strip(),
                    "content": content.strip(),
                    "category": category,
                    "date": str(note_date)
                }

                st.session_state.notes.append(new_note)

                save_data(st.session_state.notes)

                st.success("Your note has been saved!")
                st.rerun()

            else:
                st.warning("Please enter both a title and some content.")


elif page == "Browse Notes":

    st.title("Browse Your Notes")

    if len(notes) == 0:
        show_message("No notes found. Add a note to get started.")

    else:

        search_text = st.text_input(
            "Search your notes",
            placeholder="Search by title or content..."
        )

        filtered_notes = []

        for note in notes:
            if (
                search_text.lower() in note["title"].lower()
                or search_text.lower() in note["content"].lower()
            ):
                filtered_notes.append(note)

        st.write(f"{len(filtered_notes)} notes found")

        for note in reversed(filtered_notes):

            with st.expander(
                note["title"] + " · " + note["category"]
            ):

                st.write(note["content"])

                st.caption("Date: " + note.get("date", "Not added"))

                if st.button(
                    "Delete Note",
                    key="delete_" + note["id"]
                ):

                    st.session_state.notes = [
                        item for item in st.session_state.notes
                        if item["id"] != note["id"]
                    ]

                    save_data(st.session_state.notes)

                    st.rerun()

        st.markdown("---")

        json_data = json.dumps(
            st.session_state.notes,
            indent=4
        )

        st.download_button(
            "Export All Notes",
            data=json_data,
            file_name="nexora_notes.json",
            mime="application/json"
        )


elif page == "About App":

    st.title("About NEXORA")

    st.write("""
    NEXORA is a simple note management application that helps
    people keep their thoughts, plans and ideas in one place.

    It also looks for connections between notes using text
    similarity techniques.

    These connections are displayed through an interactive
    3D graph, making it easier to explore related ideas.
    """)

    st.markdown("---")

    st.subheader("Technology Used")

    st.write("""
    - Python
    - Streamlit
    - Plotly
    - Scikit-learn
    - TF-IDF
    - Cosine Similarity
    """)

    st.subheader("How it works")

    st.write("""
    1. Add notes with a title and description.
    2. The app converts note text into numerical values.
    3. It compares notes to find similar content.
    4. Related notes are connected in the 3D graph.
    """)
