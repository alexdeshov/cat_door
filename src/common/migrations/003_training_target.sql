ALTER TABLE events ADD COLUMN training_target VARCHAR(20) NOT NULL DEFAULT 'excluded';

CREATE INDEX ix_events_training_target ON events(training_target);
