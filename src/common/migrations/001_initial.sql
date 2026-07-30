PRAGMA foreign_keys = ON;

CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at DATETIME NOT NULL,
    status VARCHAR(20) NOT NULL,
    duration_seconds INTEGER NOT NULL,
    media_path VARCHAR(500),
    label VARCHAR(30) NOT NULL,
    cat_name VARCHAR(100),
    error_code VARCHAR(100)
);

CREATE INDEX ix_events_created_at ON events(created_at);
CREATE INDEX ix_events_label ON events(label);
