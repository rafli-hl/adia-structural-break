"""Disk-backed evaluator loader; preserves physical per-series parquet order."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass
class Series:
    id: str
    historical: np.ndarray
    online: np.ndarray
    target: np.ndarray
    time: np.ndarray
    tau_index: int
    tau: object = None
    time_monotone: bool = True
    complete_values: object = None
    periods: object = None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def qident(name):
    return '"' + name.replace('"', '""') + '"'


def connect(cache, memory="1GB"):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(cache / "data.duckdb"))
    con.execute("SET threads=1")
    con.execute("SET memory_limit=?", [memory])
    con.execute("SET temp_directory=?", [str(cache / "spill")])
    return con


def prepare(x_path, y_path, index_path, cache, memory="1GB"):
    """Create immutable local cache. Changed sources require a new cache path."""
    paths = [Path(p).resolve() for p in (x_path, y_path, index_path)]
    source = {name: dict(sha256=sha256(p), bytes=p.stat().st_size,
                         schema=str(pq.ParquetFile(p).schema_arrow))
              for name, p in zip(("X", "y", "index"), paths)}
    cache = Path(cache)
    marker = cache / "sources.json"
    con = connect(cache, memory)
    if marker.exists():
        if json.loads(marker.read_text()) != source:
            con.close()
            raise ValueError("Source files changed; use a new cache directory")
        return con, source
    if con.execute("SHOW TABLES").fetchall():
        con.close()
        raise ValueError("Incomplete cache; use a new cache directory")
    required = [{"id", "time", "value", "period"}, {"id", "time", "target"},
                {"id", "tau_index", "tau"}]
    for path, need in zip(paths, required):
        names = set(pq.ParquetFile(path).schema_arrow.names)
        if not need <= names:
            raise ValueError(f"Missing parquet fields: {need - names}")
        if "file_row_number" in names:
            raise ValueError("Reserved field file_row_number present")
    con.execute("CREATE TABLE x AS SELECT * FROM read_parquet(?, file_row_number=true)", [str(paths[0])])
    con.execute("CREATE TABLE y AS SELECT id,time,target FROM read_parquet(?)", [str(paths[1])])
    con.execute("CREATE TABLE idx AS SELECT id,tau_index,tau FROM read_parquet(?)", [str(paths[2])])
    checks = {
        "duplicate X keys": "SELECT count(*) FROM (SELECT id,time FROM x GROUP BY id,time HAVING count(*)>1)",
        "duplicate y keys": "SELECT count(*) FROM (SELECT id,time FROM y GROUP BY id,time HAVING count(*)>1)",
        "duplicate metadata IDs": "SELECT count(*) FROM (SELECT id FROM idx GROUP BY id HAVING count(*)>1)",
        "invalid X keys/period": "SELECT count(*) FROM x WHERE id IS NULL OR time IS NULL OR period IS NULL OR period NOT IN (1,2)",
        "invalid y": "SELECT count(*) FROM y WHERE id IS NULL OR time IS NULL OR target IS NULL OR target NOT IN (0,1)",
        "invalid tau_index": "SELECT count(*) FROM idx WHERE id IS NULL OR tau_index IS NULL OR tau_index < -1 OR tau_index != floor(tau_index)",
        "labels without X": "SELECT count(*) FROM y ANTI JOIN x USING(id,time)",
        "online X without labels": "SELECT count(*) FROM (SELECT * FROM x WHERE period=2) a ANTI JOIN y USING(id,time)",
        "X without metadata": "SELECT count(*) FROM x ANTI JOIN idx USING(id)",
        "metadata without X": "SELECT count(*) FROM idx ANTI JOIN x USING(id)",
    }
    failures = {k: int(con.execute(sql).fetchone()[0]) for k, sql in checks.items()}
    if any(failures.values()):
        raise ValueError(f"Data integrity checks failed: {failures}")
    con.execute("""CREATE TABLE observations AS
        SELECT cast(x.id AS VARCHAR) AS id, x.time, x.value, x.period,
               x.file_row_number AS ordinal, y.target
        FROM x LEFT JOIN y USING(id,time)
        ORDER BY x.id, x.file_row_number""")
    con.execute("""CREATE TABLE series_index AS
        SELECT cast(id AS VARCHAR) AS id, cast(tau_index AS BIGINT) AS tau_index, tau FROM idx""")
    if con.execute("SELECT count(*) FROM (SELECT id FROM series_index GROUP BY id HAVING count(*)>1)").fetchone()[0]:
        raise ValueError("String normalization of IDs caused a collision")
    con.execute("DROP TABLE x")
    con.execute("DROP TABLE y")
    con.execute("DROP TABLE idx")
    con.execute("CHECKPOINT")
    marker.write_text(json.dumps(source, indent=2))
    return con, source


def iter_frames(con, query, key="id", batch_size=65536):
    """Yield contiguous groups, retaining only one unfinished group."""
    cursor = con.cursor()
    reader = cursor.execute(query).to_arrow_reader(batch_size)
    carry = []
    current = None
    try:
        for batch in reader:
            frame = batch.to_pandas()
            if frame.empty:
                continue
            vals = frame[key].to_numpy()
            cuts = np.r_[0, np.flatnonzero(vals[1:] != vals[:-1]) + 1, len(vals)]
            for a, b in zip(cuts[:-1], cuts[1:]):
                value = vals[a]
                if carry and value != current:
                    yield pd.concat(carry, ignore_index=True)
                    carry = []
                current = value
                carry.append(frame.iloc[a:b].copy())
        if carry:
            yield pd.concat(carry, ignore_index=True)
    finally:
        cursor.close()


def iter_series(con, batch_size=65536):
    meta = {str(sid): (int(tau_index), tau) for sid, tau_index, tau in
            con.execute("SELECT id,tau_index,tau FROM series_index").fetchall()}
    for g in iter_frames(con, "SELECT * FROM observations ORDER BY id,ordinal", batch_size=batch_size):
        sid = str(g.id.iloc[0])
        h, o = g[g.period == 1], g[g.period == 2]
        ti, tau = meta[sid]
        yield Series(sid, h.value.to_numpy(dtype=np.float64), o.value.to_numpy(dtype=np.float64),
                     o.target.to_numpy(dtype=np.int8), o.time.to_numpy(), ti, tau,
                     bool(g.time.is_monotonic_increasing), g.value.to_numpy(dtype=np.float64),
                     g.period.to_numpy(dtype=np.int64))


def get_history(con, sid):
    return np.asarray([r[0] for r in con.execute(
        "SELECT value FROM observations WHERE id=? AND period=1 ORDER BY ordinal", [sid]).fetchall()], dtype=np.float64)


def get_values(con, sid):
    return con.execute("SELECT value,period FROM observations WHERE id=? ORDER BY ordinal", [sid]).fetchnumpy()
