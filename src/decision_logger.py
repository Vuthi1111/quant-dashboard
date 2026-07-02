import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Save the DB in the 03_Data folder
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "03_Data" / "decision_log.db"

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_utc TEXT NOT NULL,
            asset TEXT NOT NULL,
            action TEXT NOT NULL,
            direction TEXT,
            prob_high REAL,
            regime_state TEXT,
            state_time_seconds INTEGER,
            gk_current REAL,
            gk_ratio REAL,
            top_driver_1 TEXT,
            top_driver_2 TEXT,
            top_driver_3 TEXT,
            macro_confluence TEXT,
            is_news_blackout INTEGER,
            entry_price REAL,
            exit_price REAL,
            trade_uid TEXT,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_decision(asset: str, action: str, snapshot: dict, direction: str = None,
                 price: float = None, trade_uid: str = None, notes: str = ""):
    """
    snapshot = dict containing:
       prob_high, regime_state, state_time_seconds, gk_current, gk_ratio,
       top_drivers (list of 3 strings), macro_confluence,
       is_news_blackout
    """
    conn = sqlite3.connect(str(DB_PATH))
    
    # Pad top_drivers if less than 3
    drivers = snapshot.get("top_drivers", [])
    while len(drivers) < 3:
        drivers.append(None)
        
    conn.execute("""
        INSERT INTO decisions
        (timestamp_utc, asset, action, direction, prob_high, regime_state,
         state_time_seconds, gk_current, gk_ratio,
         top_driver_1, top_driver_2, top_driver_3,
         macro_confluence, is_news_blackout, entry_price, trade_uid, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        asset, action, direction,
        snapshot.get("prob_high"), snapshot.get("regime_state"), snapshot.get("state_time_seconds"),
        snapshot.get("gk_current"), snapshot.get("gk_ratio"),
        drivers[0], drivers[1], drivers[2],
        snapshot.get("macro_confluence"), int(snapshot.get("is_news_blackout", 0)),
        price, trade_uid or str(uuid.uuid4()), notes,
    ))
    conn.commit()
    conn.close()
