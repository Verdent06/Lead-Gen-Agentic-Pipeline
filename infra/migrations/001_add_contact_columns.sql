-- Run against an existing deal_sourcing_db to add contact columns next to company_name.
ALTER TABLE target_entities ADD COLUMN IF NOT EXISTS primary_contact JSONB;
ALTER TABLE target_entities ADD COLUMN IF NOT EXISTS all_contacts JSONB;
