import os
from typing import List

import streamlit as st

from recommender import Recommender
from search_utils import build_search_index, fuzzy_find_track_ids
from llm_utils import call_llm, SYSTEM_PROMPT, chat_with_tools


@st.cache_resource(show_spinner=False)
def get_recommender() -> Recommender:
    return Recommender("spotify_songs.csv")


@st.cache_data(show_spinner=False)
def get_search_index():
    rec = get_recommender()
    return build_search_index(rec.df)


def main():
    st.set_page_config(page_title="LLM Music Recommender", page_icon="🎵", layout="centered")
    st.title("🎵 LLM Music Recommender")
    st.caption("Tell the assistant songs you like; it will map them to the dataset and recommend similar tracks.")

    rec = get_recommender()
    index = get_search_index()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hi! Tell me a few songs you love (include artist names if you can)."},
        ]

    # Chat history
    for m in st.session_state.messages:
        if m["role"] == "assistant":
            st.markdown(f"**Assistant**: {m['content']}")
        elif m["role"] == "user":
            st.markdown(f"**You**: {m['content']}")

    user_input = st.text_input("Your message", placeholder="e.g., 'Blinding Lights by The Weeknd; Rolling in the Deep by Adele'", key="chat_input")

    col1, col2 = st.columns([1, 1])
    with col1:
        send = st.button("Send")
    with col2:
        if st.button("Clear Chat"):
            st.session_state.messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "assistant", "content": "Hi! Tell me a few songs you love (include artist names if you can)."},
            ]
            st.experimental_rerun()

    if send and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Tool handlers: search + recommend
        def handle_search(args):
            query = str(args.get("query", ""))
            limit = int(args.get("limit", 5))
            cutoff = int(args.get("score_cutoff", 70))
            results = fuzzy_find_track_ids(query, index, limit=limit, score_cutoff=cutoff)
            return {"matches": results}

        def handle_recommend(args):
            track_ids = list(args.get("track_ids", []))
            top_n = int(args.get("top_n", 10))
            recs = rec.recommend_by_track_ids(track_ids, top_n=top_n)
            return {"recommendations": recs}

        handlers = {"search_tracks": handle_search, "recommend_songs": handle_recommend}

        with st.spinner("Thinking with tools..."):
            assistant_reply, tool_outputs = chat_with_tools(st.session_state.messages, handlers)
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

        # If the tool flow produced recommendations, render them
        if tool_outputs.get("recommend_songs", {}).get("recommendations"):
            recs = tool_outputs["recommend_songs"]["recommendations"]
            st.subheader("Recommendations")
            for i, r in enumerate(recs, start=1):
                st.write(f"{i}. {r['track_name']} — {r['track_artist']}  (sim {r['similarity']:.3f})")


if __name__ == "__main__":
    main()


