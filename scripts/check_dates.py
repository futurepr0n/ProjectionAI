#!/usr/bin/env python3
"""
Check available prediction dates
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

cursor = conn.cursor()

# Check what dates have Hellraiser picks
print("📅 Available dates with Hellraiser picks:")
cursor.execute("""
    SELECT 
        analysis_date, 
        COUNT(*) as picks,
        COUNT(DISTINCT player_name) as players
    FROM hellraiser_picks 
    WHERE analysis_date IS NOT NULL
    GROUP BY analysis_date 
    ORDER BY analysis_date DESC
    LIMIT 30
""")

for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} picks, {row[2]} players")

# Check specific date 2025-09-04
print("\n🔍 Checking 2025-09-04 specifically:")
cursor.execute("""
    SELECT COUNT(*) FROM hellraiser_picks 
    WHERE analysis_date = '2025-09-04'
""")
count = cursor.fetchone()[0]
print(f"  Picks found: {count}")

# Check what dates are around 2025-09-04
print("\n📊 Dates around 2025-09-04:")
cursor.execute("""
    SELECT analysis_date, COUNT(*) 
    FROM hellraiser_picks 
    WHERE analysis_date BETWEEN '2025-09-01' AND '2025-09-10'
    GROUP BY analysis_date
    ORDER BY analysis_date
""")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} picks")

conn.close()
