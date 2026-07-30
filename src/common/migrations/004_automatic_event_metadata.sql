ALTER TABLE events ADD COLUMN trigger_source VARCHAR(20) NOT NULL DEFAULT 'manual';
ALTER TABLE events ADD COLUMN initiating_camera VARCHAR(10);
ALTER TABLE events ADD COLUMN detector_confidence_face FLOAT;
ALTER TABLE events ADD COLUMN detector_confidence_side FLOAT;

CREATE INDEX ix_events_trigger_source ON events(trigger_source);
