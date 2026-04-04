-- postgres/init.sql
-- Exécuté automatiquement au premier démarrage du container

CREATE TABLE IF NOT EXISTS predictions (
    id          SERIAL PRIMARY KEY,
    features    JSONB        NOT NULL,
    prediction  INTEGER      NOT NULL,
    confidence  FLOAT        NOT NULL,
    created_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS training_runs (
    id            SERIAL PRIMARY KEY,
    n_features    INTEGER,
    n_classes     INTEGER,
    epochs_run    INTEGER,
    val_accuracy  FLOAT,
    test_accuracy FLOAT,
    test_loss     FLOAT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
