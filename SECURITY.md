# Security Policy

## Supported Versions

This project is developed on a single `main` branch — there is no formal
long-term-support version matrix yet. Security fixes are applied to `main`
and released as soon as possible.

## Reporting a Vulnerability

If you discover a security vulnerability, **please do not open a public
GitHub issue**. Instead, report it privately by emailing:

**sarathj810@gmail.com**

Please include as much detail as you can:

- A description of the vulnerability and its potential impact
- Steps to reproduce, or a minimal proof-of-concept request/payload
- Any relevant logs, stack traces, or environment details

You should receive an acknowledgment within a few days. We'll work with you
to understand and validate the issue, and aim to ship a fix before any
public disclosure. We ask that you give us a reasonable amount of time to
address the issue before disclosing it publicly.

## Scope

Given this project executes LLM-generated SQL against a connected database,
areas of particular interest include (but aren't limited to):

- Prompt-injection bypasses that lead to unintended query generation
- Ways to make the agent emit or execute mutating statements (`INSERT`,
  `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, etc.) despite the query
  checker and the read-only database guard
- Ways to access tables/data outside the schema explicitly provided to the
  agent
- Any path that leaks database credentials, connection strings, or other
  secrets through logs or responses

Thank you for helping keep this project and its users safe.
