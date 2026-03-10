UPDATE play_by_play_plays
SET batter = REGEXP_REPLACE(
    batter,
    '\s+(hit|struck|walk|fly|bunt|by|grounded|lined|flied|popped|singled|doubled|tripled|homered|reached|sacrifice|intentional|called|swinging)\b.*$',
    '',
    'ig'
)
WHERE batter ~ '\s+(hit|struck|walk|fly|bunt|by|grounded|lined|flied|popped|singled|doubled|tripled|homered|reached|sacrifice|intentional|called|swinging)\b';
