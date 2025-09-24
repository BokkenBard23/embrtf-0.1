CREATE TABLE IF NOT EXISTS callback_phrases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phrase TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL, -- 'client_promise', 'client_uncertain', 'operator_offer'
    category INTEGER NOT NULL, -- 1=ошибка, 2=корректно, 3=ненужный звонок
    frequency INTEGER DEFAULT 1,
    verified BOOLEAN DEFAULT 0 -- 0=не проверено, 1=проверено вручную
);