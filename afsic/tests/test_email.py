"""
Test afsic_ext email functionality.

Credentials are read from environment variables (set in ~/.bashrc):
    AFSIC_SMTP_URL      e.g. smtp://smtp.qq.com:587
    AFSIC_SMTP_USERNAME sender account
    AFSIC_SMTP_PASSWORD SMTP authorisation code
    AFSIC_SMTP_FROM     sender address
"""

import os
import pytest
import afsic


def _make_info(to: str, subject: str, body: str, is_html: bool = False) -> afsic.EmailInfo:
    info = afsic.EmailInfo()
    info.smtp_url = os.environ["AFSIC_SMTP_URL"]
    info.username = os.environ["AFSIC_SMTP_USERNAME"]
    info.password = os.environ["AFSIC_SMTP_PASSWORD"]
    info.From     = os.environ["AFSIC_SMTP_FROM"]
    info.to       = to
    info.subject  = subject
    info.body     = body
    info.is_html  = is_html
    return info


@pytest.mark.skipif(
    not all(k in os.environ for k in (
        "AFSIC_SMTP_URL", "AFSIC_SMTP_USERNAME",
        "AFSIC_SMTP_PASSWORD", "AFSIC_SMTP_FROM",
    )),
    reason="Email credentials not set in environment",
)
def test_send_plain_email():
    info = _make_info(
        to      = "Pengfei.Ma@glasgow.ac.uk",
        subject = "[afsic] test_send_plain_email",
        body    = "This is an automated test email from afsic_ext.send_email().",
    )
    assert afsic.send_email(info), "send_email returned False"


@pytest.mark.skipif(
    not all(k in os.environ for k in (
        "AFSIC_SMTP_URL", "AFSIC_SMTP_USERNAME",
        "AFSIC_SMTP_PASSWORD", "AFSIC_SMTP_FROM",
    )),
    reason="Email credentials not set in environment",
)
def test_send_html_email():
    info = _make_info(
        to      = "Pengfei.Ma@glasgow.ac.uk",
        subject = "[afsic] test_send_html_email",
        body    = "<h1>afsic</h1><p>HTML email test from <b>afsic_ext</b>.</p>",
        is_html = True,
    )
    assert afsic.send_email(info), "send_email returned False"
