import logging
import os
import re
import unicodedata
from typing import Dict, List
from collections import defaultdict

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from rapidfuzz import fuzz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_REGISTRY_CACHE: Dict[int, Dict] = {}


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ''

    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode()
    name = re.sub(r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$', '', name, flags=re.IGNORECASE)
    name = name.replace('.', ' ')
    name = re.sub(r'\s+', ' ', name).strip().lower()

    if ',' in name:
        parts = name.split(',', 1)
        name = f"{parts[1].strip()} {parts[0].strip()}".strip()

    return name


def _connect_db():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.1.23'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'baseball_migration_test'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'korn5676')
    )


def fuzzy_join_names(df_left, df_right, left_col, right_col, threshold=85):
    left_norm = df_left[left_col].apply(normalize_name)
    right_norm = df_right[right_col].apply(normalize_name)

    right_lookup = dict(zip(right_norm, df_right.index))

    matched_indices = []
    for i, name in enumerate(left_norm):
        if name in right_lookup:
            matched_indices.append((i, right_lookup[name]))
            continue

        best_score = 0
        best_idx = None
        for j, rname in enumerate(right_norm):
            score = fuzz.token_sort_ratio(name, rname)
            if score > best_score:
                best_score = score
                best_idx = j

        matched_indices.append((i, best_idx if best_score >= threshold else None))

    return matched_indices


def _load_player_name_registry(conn) -> Dict:
    cache_key = id(conn)
    if cache_key in _REGISTRY_CACHE:
        return _REGISTRY_CACHE[cache_key]

    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(
        """
        SELECT canonical_name, aliases, last_name, first_name, first_initial
        FROM player_name_map
        """
    )

    exact_lookup = defaultdict(list)
    last_name_lookup = defaultdict(list)
    rows = []

    for row in cursor.fetchall():
        entry = {
            'canonical_name': row['canonical_name'],
            'aliases': row['aliases'] or [],
            'last_name': normalize_name(row['last_name'] or ''),
            'first_name': normalize_name(row['first_name'] or ''),
            'first_initial': normalize_name(row['first_initial'] or ''),
        }
        rows.append(entry)

        exact_lookup[normalize_name(entry['canonical_name'])].append(entry)
        for alias in entry['aliases']:
            normalized_alias = normalize_name(alias)
            if normalized_alias:
                exact_lookup[normalized_alias].append(entry)

        if entry['last_name']:
            last_name_lookup[entry['last_name']].append(entry)

    registry = {
        'rows': rows,
        'exact_lookup': exact_lookup,
        'last_name_lookup': last_name_lookup,
    }
    _REGISTRY_CACHE[cache_key] = registry
    return registry


def resolve_name_match(raw_name: str, conn=None) -> Dict:
    """
    Resolve a raw name to a canonical player identity and return match metadata.

    Match policy:
    - exact canonical / alias match
    - unique last-name match
    - first-name / first-initial disambiguation within same last name
    - conservative fuzzy match among same-last-name candidates only
    - otherwise unresolved
    """
    normalized_name = normalize_name(raw_name)
    result = {
        'raw_name': raw_name,
        'normalized_name': normalized_name,
        'canonical_name': normalized_name,
        'matched': False,
        'match_type': 'unresolved',
        'ambiguous': False,
        'candidate_count': 0,
        'candidates': [],
    }

    if not normalized_name:
        result['match_type'] = 'empty'
        return result

    owns_conn = conn is None
    if owns_conn:
        try:
            conn = _connect_db()
        except Exception as exc:
            logger.warning(f"Cannot resolve player name '{raw_name}' without DB: {exc}")
            result['match_type'] = 'no_db'
            return result

    try:
        registry = _load_player_name_registry(conn)
        candidates = list(registry['exact_lookup'].get(normalized_name, []))
        if not candidates:
            parts = normalized_name.split()
            if parts:
                candidates = list(registry['last_name_lookup'].get(parts[-1], []))
        result['candidate_count'] = len(candidates)
        result['candidates'] = [c['canonical_name'] for c in candidates[:5]]

        if not candidates:
            return result

        lowered_canonical = {normalize_name(c['canonical_name']): c for c in candidates}
        if normalized_name in lowered_canonical:
            canonical = lowered_canonical[normalized_name]['canonical_name']
            result.update({
                'canonical_name': canonical,
                'matched': True,
                'match_type': 'exact_canonical',
                'ambiguous': False,
            })
            return result

        for candidate in candidates:
            aliases = candidate.get('aliases') or []
            normalized_aliases = {normalize_name(alias) for alias in aliases if alias}
            if normalized_name in normalized_aliases:
                result.update({
                    'canonical_name': candidate['canonical_name'],
                    'matched': True,
                    'match_type': 'exact_alias',
                    'ambiguous': False,
                })
                return result

        if len(candidates) == 1:
            result.update({
                'canonical_name': candidates[0]['canonical_name'],
                'matched': True,
                'match_type': 'unique_last_name',
                'ambiguous': False,
            })
            return result

        parts = normalized_name.split()
        first_token = parts[0] if parts else ''
        first_initial = first_token[:1]

        exact_first = [
            candidate for candidate in candidates
            if normalize_name(candidate.get('first_name') or '') == first_token
        ]
        if len(exact_first) == 1:
            result.update({
                'canonical_name': exact_first[0]['canonical_name'],
                'matched': True,
                'match_type': 'exact_first_last',
                'ambiguous': False,
            })
            return result

        initial_matches = [
            candidate for candidate in candidates
            if normalize_name(candidate.get('first_initial') or '') == first_initial
        ]
        if len(initial_matches) == 1:
            result.update({
                'canonical_name': initial_matches[0]['canonical_name'],
                'matched': True,
                'match_type': 'first_initial_last',
                'ambiguous': False,
            })
            return result

        scored = []
        for candidate in candidates:
            options = [candidate['canonical_name'], *(candidate.get('aliases') or [])]
            best_score = max(fuzz.token_sort_ratio(normalized_name, normalize_name(option)) for option in options if option)
            scored.append((best_score, candidate['canonical_name']))

        scored.sort(reverse=True)
        if len(scored) == 1 and scored[0][0] >= 96:
            result.update({
                'canonical_name': scored[0][1],
                'matched': True,
                'match_type': 'fuzzy_same_last_name',
                'ambiguous': False,
            })
            return result

        if len(scored) >= 2 and scored[0][0] >= 96 and (scored[0][0] - scored[1][0]) >= 3:
            result.update({
                'canonical_name': scored[0][1],
                'matched': True,
                'match_type': 'fuzzy_same_last_name',
                'ambiguous': False,
            })
            return result

        result['match_type'] = 'ambiguous_last_name'
        result['ambiguous'] = True
        return result

    except Exception as exc:
        logger.warning(f"Error during canonical lookup for '{raw_name}': {exc}")
        result['match_type'] = 'lookup_error'
        return result
    finally:
        if owns_conn and conn is not None:
            conn.close()


def normalize_to_canonical(raw_name, conn=None, return_metadata=False):
    resolution = resolve_name_match(raw_name, conn=conn)
    if return_metadata:
        return resolution
    return resolution['canonical_name']


def build_name_map(conn=None):
    """
    Build a dictionary of normalized name variants to canonical names.

    Last-name-only mappings are only added when that last name is unique in the
    registry; ambiguous surnames are intentionally skipped.
    """
    owns_conn = conn is None
    if owns_conn:
        try:
            conn = _connect_db()
        except Exception as exc:
            logger.error(f"Cannot build name map without DB connection: {exc}")
            return {}

    name_map = {}

    try:
        registry = _load_player_name_registry(conn)
        rows = registry['rows']

        last_name_counts = {}
        for row in rows:
            last_name = normalize_name(row['last_name'] or '')
            if last_name:
                last_name_counts[last_name] = last_name_counts.get(last_name, 0) + 1

        for row in rows:
            canonical = row['canonical_name']
            normalized_canonical = normalize_name(canonical)
            if normalized_canonical:
                name_map[normalized_canonical] = canonical

            for alias in row['aliases'] or []:
                normalized_alias = normalize_name(alias)
                if normalized_alias:
                    name_map[normalized_alias] = canonical

            last_name = normalize_name(row['last_name'] or '')
            if last_name and last_name_counts.get(last_name, 0) == 1:
                name_map[f'last_{last_name}'] = canonical

        logger.info(f"Built name map with {len(name_map)} entries")
        return name_map

    except Exception as exc:
        logger.error(f"Error building name map: {exc}")
        return {}
    finally:
        if owns_conn and conn is not None:
            conn.close()
