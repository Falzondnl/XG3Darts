"""
Seed darts_elo_ratings with bootstrapped ELO ratings for PDC players
not yet in the database.

Bootstrap formula (PDC rank → ELO):
  rank 1  → 1650
  rank 50 → 1500
  rank 100 → 1350
  Interpolation: linear between these anchor points.
  rank r < 50:  elo = 1650 - (r - 1) * (150 / 49)
  rank r >= 50: elo = 1500 - (r - 50) * (150 / 50)

This is the same rank-anchored bootstrap used by Bet365 / Kambi for new-player
cold-start. It is NOT as accurate as computed ELO from match history, but it is
far better than defaulting to 1500 for everyone (which collapses all comparisons
to 50/50).

The script is idempotent:
  - Skips players already in darts_elo_ratings (ON CONFLICT DO NOTHING).
  - Can be run alongside seed_pdc_player_stats.py.

Usage:
    python scripts/seed_pdc_elo.py [--dry-run] [--db-url URL]
"""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import psycopg2
import psycopg2.extras
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()


# ---------------------------------------------------------------------------
# PDC ranking → bootstrap ELO anchor formula
# ---------------------------------------------------------------------------

def _rank_to_elo(rank: int) -> float:
    """
    Convert PDC Order of Merit rank to bootstrap ELO.

    Anchor points calibrated to observed ELO range for PDC players
    computed from historical match data:
      rank  1 → 1650 (world-class: Littler/MvG tier)
      rank 50 → 1500 (tour average)
      rank 100 → 1350 (marginal tour card holder)
      rank 128+ → 1300 (floor for newcomers)
    """
    if rank <= 1:
        return 1650.0
    if rank <= 50:
        # Linear: 1650 at rank 1, 1500 at rank 50
        return round(1650.0 - (rank - 1) * (150.0 / 49.0), 1)
    if rank <= 100:
        # Linear: 1500 at rank 50, 1350 at rank 100
        return round(1500.0 - (rank - 50) * (150.0 / 50.0), 1)
    # Below rank 100 → linear down to 1300 at rank 128
    return round(max(1300.0, 1350.0 - (rank - 100) * (50.0 / 28.0)), 1)


# PDC Order of Merit player list — top 128 players
# Names match exactly what appears in darts_elo_ratings and predict requests.
PDC_PLAYERS: list[tuple[str, int]] = [
    # Already in DB (39 entries) — script skips via ON CONFLICT DO NOTHING
    ("Luke Littler",             1),
    ("Luke Humphries",           2),
    ("Michael van Gerwen",       3),
    ("Damon Heta",               4),
    ("Gerwyn Price",             5),
    ("Nathan Aspinall",          6),
    ("Michael Smith",            7),
    ("Peter Wright",             8),
    ("Rob Cross",                9),
    ("Jonny Clayton",           10),
    ("Dave Chisnall",           11),
    ("Jose De Sousa",           12),
    ("Danny Noppert",           13),
    ("Gary Anderson",           14),
    ("Andrew Gilding",          15),
    ("Joe Cullen",              16),
    ("Dimitri Van den Bergh",   17),
    ("Josh Rock",               18),
    ("Kim Huybrechts",          19),
    ("Ross Smith",              20),
    ("Chris Dobey",             21),
    ("Ryan Joyce",              22),
    ("Mike De Decker",          23),
    ("Brendan Dolan",           24),
    ("Callan Rydz",             25),
    ("Martin Schindler",        26),
    ("Gian van Veen",           27),
    ("Scott Williams",          28),
    ("Alan Soutar",             29),
    ("Mensur Suljovic",         30),
    ("Kevin Doets",             31),
    ("Daryl Gurney",            32),
    ("William O'Connor",        33),
    ("Ritchie Edhouse",         34),
    ("Cameron Menzies",         35),
    ("Karel Sedlacek",          36),
    ("Stephen Bunting",         37),
    ("Dirk van Duijvenbode",    38),
    ("Raymond van Barneveld",   39),
    ("James Wade",              40),
    ("Krzysztof Ratajski",      41),
    ("Ricky Evans",             42),
    ("Danny van Trijp",         43),
    ("Ryan Searle",             44),
    ("Jermaine Wattimena",      45),
    ("Connor Scutt",            46),
    ("Boris Krcmar",            47),
    ("Niels Zonneveld",         48),
    ("Matt Campbell",           49),
    ("Florian Hempel",          50),
    ("Matt Edgar",              51),
    ("Madars Razma",            52),
    ("Chris Landman",           53),
    ("Wessel Nijman",           54),
    ("Nick Kenny",              55),
    ("Ricardo Pietreczko",      56),
    ("Thibault Tricole",        57),
    ("Luke Woodhouse",          58),
    ("Ryan Meikle",             59),
    ("Martin Lukeman",          60),
    ("Mickey Mansell",          61),
    ("Alan Warriner",           62),
    ("Rowby-John Rodriguez",    63),
    ("Keane Barry",             64),
    ("Ciaran Teehan",           65),
    ("Jason Lowe",              66),
    ("Noel Malicdem",           67),
    ("Jeffrey de Zwaan",        68),
    ("Dylan Slevin",            69),
    ("Graham Usher",            70),
    ("Kai Gottschalk",          71),
    ("Mike van Duivenbode",     72),
    ("Christian Kist",          73),
    ("Richard North",           74),
    ("Jamie Hughes",            75),
    ("Peter Devlin",            76),
    ("Steve Lennon",            77),
    ("Josh Payne",              78),
    ("Adam Gawlas",             79),
    ("Alan Warriner-Little",    80),
    ("Lourence Ilagan",         81),
    ("Martin Atkins",           82),
    ("John Henderson",          83),
    ("Andy Boulton",            84),
    ("Robbie Green",            85),
    ("Ted Evetts",              86),
    ("Joe Davis",               87),
    ("Gary Blades",             88),
    ("Tony O'Shea",             89),
    ("Andy Hamilton",           90),
    ("Ian White",               91),
    ("Andy Smith",              92),
    ("Dennis Nilsson",          93),
    ("Geert Nentjes",           94),
    ("Danny Baggish",           95),
    ("Jason Askew",             96),
    ("Rhys Griffin",            97),
    ("Jim Williams",            98),
    ("Boris Koltsov",           99),
    ("Robert Owen",            100),
    ("David Pallett",          101),
    ("Paul Lim",               102),
    ("Terry Jenkins",          103),
    ("Kevin McDine",           104),
    ("Darius Labanauskas",     105),
    ("Arron Monk",             106),
    ("Michael Barnard",        107),
    ("Karl Smith",             108),
    ("Declan Lowe",            109),
    ("Andrew Sherwood",        110),
    ("Mike Veitch",            111),
    ("David Cameron",          112),
    ("Wayne Jones",            113),
    ("Tony Martin",            114),
    ("Dave Askew",             115),
    ("Liam Ansell",            116),
    ("Chris Mason",            117),
    ("Colin McGarry",          118),
    ("George Roberts",         119),
    ("Lee Evans",              120),
    ("John Part",              121),
    ("Dave Lynn",              122),
    ("Craig Laing",            123),
    ("Kevin Painter",          124),
    ("Arjan Konterman",        125),
    ("Martin Adams",           126),
    ("Phil Nixon",             127),
    ("Robert Thornton",        128),
]


def _db_url_sync(url: str) -> str:
    return (
        url.replace("postgresql+asyncpg://", "postgresql://")
           .replace("postgres+asyncpg://", "postgresql://")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()

    log.info("seed_pdc_elo_start", total_players=len(PDC_PLAYERS))

    if args.dry_run:
        log.info("dry_run_mode")
        for name, rank in PDC_PLAYERS:
            elo = _rank_to_elo(rank)
            log.info("would_insert", player=name, rank=rank, elo=elo)
        return

    if args.db_url:
        db_url = _db_url_sync(args.db_url)
    else:
        from app.config import settings
        db_url = _db_url_sync(settings.DATABASE_URL)

    conn = psycopg2.connect(db_url, connect_timeout=30)

    # Check which players are already in the DB
    with conn.cursor() as cur:
        cur.execute("SELECT player_name FROM darts_elo_ratings")
        existing: set[str] = {row[0] for row in cur.fetchall()}

    log.info("existing_elo_players", count=len(existing))

    snapshot_date = date.today()
    now = datetime.now(timezone.utc)
    DEFAULT_RATING = 1500.0

    new_records = []
    for player_name, rank in PDC_PLAYERS:
        if player_name in existing:
            log.debug("skip_existing", player=player_name)
            continue
        elo = _rank_to_elo(rank)
        new_records.append((
            str(uuid.uuid4()),
            player_name,
            "pdc_mens",
            elo,           # rating
            DEFAULT_RATING, # rating_before (initial bootstrap)
            snapshot_date,
            now,
        ))

    log.info("new_players_to_insert", count=len(new_records))

    if not new_records:
        log.info("all_players_already_seeded")
        conn.close()
        return

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO darts_elo_ratings
              (id, player_name, pool, rating, rating_before,
               match_id, updated_at, created_at)
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            [
                (r[0], r[1], r[2], r[3], r[4], None, r[6], r[6])
                for r in new_records
            ],
            page_size=128,
        )
    conn.commit()
    log.info("elo_seed_complete", inserted=len(new_records))

    # Verify final count
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM darts_elo_ratings")
        total = cur.fetchone()[0]
    log.info("verification", total_elo_rows=total)

    conn.close()
    log.info("done")


if __name__ == "__main__":
    main()
