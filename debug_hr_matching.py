#!/usr/bin/env python3
"""
Debug HR matching
"""
import psycopg2
from psycopg2.extras import RealDictCursor

conn = psycopg2.connect(
    host='192.168.1.23',
    port=5432,
    database='baseball_migration_test',
    user='postgres',
    password='korn5676'
)

cursor = conn.cursor(cursor_factory=RealDictCursor)

# Get HR hitters from play_by_play for 2025-08-29
print("=== HR Hitters from play_by_play_plays on 2025-08-29 ===")
cursor.execute("""
    SELECT DISTINCT pp.batter, g.home_team, g.away_team
    FROM play_by_play_plays pp
    JOIN games g ON pp.game_id = g.game_id
    WHERE g.game_date = '2025-08-29'
      AND pp.play_result = 'Home Run'
    ORDER BY pp.batter
""")
hr_hitters = cursor.fetchall()
print(f"Found {len(hr_hitters)} HR hitters:")
for h in hr_hitters[:20]:
    print(f"  {h['batter']} ({h['away_team']} @ {h['home_team']})")

# Get Hellraiser picks for 2025-08-29
print("\n=== Hellraiser Picks on 2025-08-29 ===")
cursor.execute("""
    SELECT player_name, team, confidence_score
    FROM hellraiser_picks
    WHERE analysis_date = '2025-08-29'
    ORDER BY confidence_score DESC
    LIMIT 20
""")
picks = cursor.fetchall()
print(f"Found {len(picks)} picks (showing top 20):")
for p in picks:
    print(f"  {p['player_name']} ({p['team']}) - Conf: {p['confidence_score']}")

# Check if names match
print("\n=== Name Matching Check ===")
hr_names = {h['batter'].strip().lower() for h in hr_hitters}
pick_names = {p['player_name'].strip().lower() for p in picks}

matches = hr_names & pick_names
print(f"Exact matches: {len(matches)}")
if matches:
    print(f"  Matched: {list(matches)[:10]}")

# Try partial matching
print("\n=== Partial Name Matching ===")
matched_hr = []
for hr in hr_hitters:
    hr_name = hr['batter'].strip().lower()
    for pick in picks:
        pick_name = pick['player_name'].strip().lower()
        # Check if either contains the other or if last names match
        if hr_name in pick_name or pick_name in hr_name:
            matched_hr.append((hr['batter'], pick['player_name']))
            break
        # Check last name match
        hr_last = hr_name.split()[-1] if hr_name.split() else ''
        pick_last = pick_name.split()[-1] if pick_name.split() else ''
        if hr_last and pick_last and hr_last == pick_last:
            matched_hr.append((hr['batter'], pick['player_name']))
            break

print(f"Partial matches found: {len(matched_hr)}")
for hr, pick in matched_hr[:15]:
    print(f"  HR: '{hr}' <-> Pick: '{pick}'")

conn.close()
