# Security Policy

## Supported versions

Only the latest released version receives fixes.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability reporting
(repository **Security** tab, **Report a vulnerability**) and include:

- the affected version or commit;
- steps to reproduce, and a minimal input file if the problem is triggered by data;
- the impact you expect.

You can expect an acknowledgement within 7 days. Fixes are released as a patch version and noted in
`CHANGELOG.md`.

## Scope

In scope: the code in `src/`, `scripts/`, and the Streamlit dashboard, including how it handles uploaded
files (for example, spreadsheet formula injection in CSV and Excel exports).

Out of scope: the IBM Telco data itself (see `DATA.md`) and vulnerabilities in third-party dependencies
that are not reachable through this project.

## Handling data and secrets

- Never commit `.env`, customer data, or credentials. `.env` and `artifacts/` are git-ignored.
- Seed and business assumptions live in `params.yaml` and `assumptions.yaml`, not in the environment.
