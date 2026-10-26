# Credentials and historical research

Use the empty fields in `.env.example` to configure an ignored local `.env` file. Never reuse credentials copied from an old experiment, commit, screenshot, or report. Keep exchange permissions minimal and use separate development credentials for experiments.

Earlier development included hard-coded exchange credential material. The research ledger records a rotation requirement. Removing a value from source or rewriting Git history does not revoke it or remove copies already held elsewhere; the account owner must revoke or rotate exposed credentials through the provider.

The repository's live and testnet integration code is historical research infrastructure. Its presence does not establish that any strategy is validated for live capital. Review `docs/STATUS.md` before interpreting old results or running exchange-connected workflows.
