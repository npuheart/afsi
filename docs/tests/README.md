# Tests — Overview

This directory documents every test file in `afsic/tests/` to aid future
understanding and review.

---

## `tests/test_email.py`

**Purpose:** Integration test for the `afsic.send_email` / `afsic.EmailInfo`
nanobind bindings (backed by libcurl SMTP).

**What it tests:**

| Test | Description |
|---|---|
| `test_send_plain_email` | Sends a plain-text email to `Pengfei.Ma@glasgow.ac.uk` and asserts `send_email` returns `True`. |
| `test_send_html_email` | Sends an HTML email to the same address and asserts success. |

**Credentials:** Read from environment variables — tests are automatically
**skipped** if any variable is missing (no hard-coded secrets):

| Variable | Example value |
|---|---|
| `AFSIC_SMTP_URL` | `smtp://smtp.qq.com:587` |
| `AFSIC_SMTP_USERNAME` | `499908174@qq.com` |
| `AFSIC_SMTP_PASSWORD` | QQ SMTP authorisation code |
| `AFSIC_SMTP_FROM` | `499908174@qq.com` |

**How to run:**

```bash
source ~/.bashrc          # loads the AFSIC_SMTP_* variables
cd /home/staff4/pma/afsi/afsic
pytest tests/test_email.py -v
```

---

*Add a new section here whenever a new test file is created.*
