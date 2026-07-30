PRAGMA foreign_keys = ON;

CREATE TABLE model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind VARCHAR(30) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    sha256 VARCHAR(64) NOT NULL UNIQUE,
    metadata_json TEXT NOT NULL,
    first_seen_at DATETIME NOT NULL
);

CREATE INDEX ix_model_versions_kind ON model_versions(kind);

CREATE TABLE evaluation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at DATETIME NOT NULL,
    completed_at DATETIME,
    run_kind VARCHAR(20) NOT NULL DEFAULT 'identity',
    filter_json TEXT NOT NULL,
    detector_model_version_id INTEGER REFERENCES model_versions(id),
    face_model_version_id INTEGER REFERENCES model_versions(id),
    side_model_version_id INTEGER REFERENCES model_versions(id),
    status VARCHAR(20) NOT NULL,
    metrics_json TEXT,
    error_message TEXT
);

CREATE INDEX ix_evaluation_runs_created_at ON evaluation_runs(created_at);
CREATE INDEX ix_evaluation_runs_status ON evaluation_runs(status);

CREATE TABLE event_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    model_version_id INTEGER NOT NULL REFERENCES model_versions(id),
    view VARCHAR(20) NOT NULL,
    origin VARCHAR(20) NOT NULL,
    evaluation_run_id INTEGER REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    decision VARCHAR(120) NOT NULL,
    predicted_name VARCHAR(100),
    confidence FLOAT,
    frames_used INTEGER NOT NULL DEFAULT 0,
    max_detector_confidence FLOAT,
    mean_inference_ms FLOAT,
    raw_scores_json TEXT,
    created_at DATETIME NOT NULL
);

CREATE INDEX ix_event_predictions_event_id ON event_predictions(event_id);
CREATE INDEX ix_event_predictions_model_version_id
    ON event_predictions(model_version_id);
CREATE INDEX ix_event_predictions_evaluation_run_id
    ON event_predictions(evaluation_run_id);
CREATE INDEX ix_event_predictions_origin ON event_predictions(origin);
