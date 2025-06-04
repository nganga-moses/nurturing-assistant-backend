CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL,
    task_details TEXT,
    priority TEXT CHECK(priority IN ('low', 'normal', 'high')) DEFAULT 'normal',
    retry_count INTEGER DEFAULT 0,
    result TEXT,
    status TEXT CHECK(status IN ('pending', 'completed', 'failed')),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);