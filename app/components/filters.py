import streamlit as st

def sidebar_filters(dim_match, dim_team, dim_player):
    st.sidebar.header("Filters")

    dm = dim_match.copy()

    # Competition / season (if present)
    if "competition_name" in dm.columns:
        comp = st.sidebar.selectbox("Competition", sorted(dm["competition_name"].dropna().unique()))
        dm = dm[dm["competition_name"] == comp]

    if "season_name" in dm.columns:
        season = st.sidebar.selectbox("Season", sorted(dm["season_name"].dropna().unique()))
        dm = dm[dm["season_name"] == season]

    # Match selection
    if "match_label" in dm.columns:
        match_choice = st.sidebar.selectbox("Match", dm["match_label"].tolist())
        match_id = int(dm.loc[dm["match_label"] == match_choice, "match_id"].iloc[0])
        dm_match = dm[dm["match_id"] == match_id]
    else:
        match_id = int(st.sidebar.selectbox("Match ID", sorted(dm["match_id"].unique())))
        dm_match = dm[dm["match_id"] == match_id]

    # Team selection
    if {"home_team_name","away_team_name"}.issubset(dm_match.columns) and len(dm_match):
        team = st.sidebar.selectbox(
            "Team",
            [dm_match["home_team_name"].iloc[0], dm_match["away_team_name"].iloc[0]]
        )
    elif "team_name" in dim_team.columns:
        team = st.sidebar.selectbox("Team", sorted(dim_team["team_name"].dropna().unique()))
    else:
        team = st.sidebar.text_input("Team name")

    # Player selection (optional)
    if "player_name" in dim_player.columns:
        player = st.sidebar.selectbox(
            "Player (optional)",
            ["(All)"] + sorted(dim_player["player_name"].dropna().unique())
        )
        player = None if player == "(All)" else player
    else:
        player = None

    return match_id, team, player
