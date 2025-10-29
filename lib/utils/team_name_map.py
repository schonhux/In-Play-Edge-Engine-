NBA_TEAM_ALIASES = {
    "LA Clippers": "Los Angeles Clippers",
    "New Orleans Pelicans": "New Orleans Pelicans",
    "NY Knicks": "New York Knicks",
    "Brooklyn Nets": "Brooklyn Nets",
    "GS Warriors": "Golden State Warriors",
    "OKC Thunder": "Oklahoma City Thunder",
    "SA Spurs": "San Antonio Spurs",
    "Portland Trail Blazers": "Portland Trail Blazers",
    "Cleveland Cavaliers": "Cleveland Cavaliers",
    "Milwaukee Bucks": "Milwaukee Bucks",
    "Houston Rockets": "Houston Rockets",
    "Denver Nuggets": "Denver Nuggets",
    "Dallas Mavericks": "Dallas Mavericks",
    "Los Angeles Lakers": "Los Angeles Lakers",
    "Memphis Grizzlies": "Memphis Grizzlies",
    "Boston Celtics": "Boston Celtics",
    "Philadelphia 76ers": "Philadelphia 76ers",
    "Toronto Raptors": "Toronto Raptors",
    "Miami Heat": "Miami Heat",
    "Atlanta Hawks": "Atlanta Hawks",
    "Chicago Bulls": "Chicago Bulls",
    "Sacramento Kings": "Sacramento Kings",
    "Orlando Magic": "Orlando Magic",
    "Utah Jazz": "Utah Jazz",
    "Phoenix Suns": "Phoenix Suns",
    "Minnesota Timberwolves": "Minnesota Timberwolves",
    "Washington Wizards": "Washington Wizards",
    "Indiana Pacers": "Indiana Pacers",
    "Detroit Pistons": "Detroit Pistons",
    "Charlotte Hornets": "Charlotte Hornets"
}

def normalize_name(name: str) -> str:
    # If already standard, return it.
    if name in NBA_TEAM_ALIASES.values():
        return name
    # Otherwise map alias → canonical.
    return NBA_TEAM_ALIASES.get(name, name)
