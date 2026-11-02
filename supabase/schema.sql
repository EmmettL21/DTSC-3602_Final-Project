-- D2 dashboard schema. Five tables, all fed by `scripts/upload_to_supabase.py`
-- from the artifacts the audit/cleaning pipeline already produces -- nothing
-- here is hand-entered. Run once against a fresh Supabase project:
--
--   psql "$SUPABASE_DB_URL" -f supabase/schema.sql
--
-- (or paste into the Supabase SQL editor).

create table if not exists postings (
    job_id              bigint primary key,
    title               text,
    company_name        text,
    location            text,
    state               text,
    formatted_work_type text,
    formatted_experience_level text,
    salary_annual       double precision,
    salary_annual_capped double precision,
    salary_outlier_flag boolean,
    remote_allowed      boolean,
    is_likely_duplicate boolean,
    duplicate_group_id  bigint,
    skill_names         jsonb,
    job_posting_url     text,
    application_url     text
);

create table if not exists resumes (
    resume_id   bigint primary key,
    category    text,
    resume_text text
);

create table if not exists data_dictionary (
    table_name    text not null,
    column_name   text not null,
    dtype         text,
    pct_missing   double precision,
    n_unique      bigint,
    example_value text,
    note          text,
    primary key (table_name, column_name)
);

create table if not exists data_quality_findings (
    finding      text primary key,
    detail       text,
    count        bigint,
    pct_of_total double precision,
    fixed        boolean
);

create table if not exists test_split_summary (
    id              int primary key default 1,
    n_pairs         bigint,
    n_resumes       bigint,
    n_jobs          bigint,
    resume_id_sha256 text,
    job_id_sha256   text,
    frozen_on       timestamptz,
    constraint single_row check (id = 1)
);
