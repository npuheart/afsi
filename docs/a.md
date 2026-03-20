# afsic_ext — Email Module

`afsic_ext` exposes a thin nanobind wrapper around libcurl SMTP so Python
scripts can send notification emails without any extra Python dependency.

---

## Configuration

Credentials are stored as shell environment variables.  
Add the following to `~/.bashrc` (already done on this machine):

```bash
export AFSIC_SMTP_URL="smtp://smtp.qq.com:587"
export AFSIC_SMTP_USERNAME="499908174@qq.com"
export AFSIC_SMTP_PASSWORD="<authorisation-code>"   # QQ SMTP authorisation code, not login password
export AFSIC_SMTP_FROM="499908174@qq.com"
```

> **Never hard-code credentials in source files.**  
> The `password` field is the QQ-mail SMTP *authorisation code*
> (授权码), not the account login password.

---

## API

```python
import afsic_ext

info = afsic_ext.EmailInfo()
info.smtp_url = "smtp://smtp.qq.com:587"
info.username = "499908174@qq.com"
info.password = "<authorisation-code>"
info.From     = "499908174@qq.com"
info.to       = "recipient@example.com"
info.subject  = "Hello from afsic"
info.body     = "Plain-text body."
info.is_html  = False          # set True for HTML body

ok = afsic_ext.send_email(info)
print("sent:", ok)
```

### `EmailInfo` fields

| Field       | Type   | Description                              |
|-------------|--------|------------------------------------------|
| `smtp_url`  | `str`  | SMTP URL, e.g. `smtp://smtp.qq.com:587`  |
| `username`  | `str`  | SMTP login username                      |
| `password`  | `str`  | SMTP authorisation code                  |
| `From`      | `str`  | Sender address                           |
| `to`        | `str`  | Recipient address                        |
| `subject`   | `str`  | Email subject line                       |
| `body`      | `str`  | Email body (plain text or HTML)          |
| `is_html`   | `bool` | `True` → `Content-Type: text/html`       |

### `send_email(info) → bool`

Sends the email described by `info`.  
Returns `True` on success, `False` on failure (curl error printed to stderr).

---

## Running the tests

```bash
# source credentials first if not already in the shell
source ~/.bashrc

cd /home/staff4/pma/afsi/afsic
pytest tests/test_email.py -v
```

Both `test_send_plain_email` and `test_send_html_email` will be skipped
automatically if the environment variables are not set.

