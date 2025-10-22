-- Create the table that n8n will use
CREATE TABLE IF NOT EXISTS competitor_urls (
    id SERIAL PRIMARY KEY,
    normalized_url TEXT UNIQUE,
    source_site TEXT,
    date_found TIMESTAMPTZ DEFAULT NOW()
);
