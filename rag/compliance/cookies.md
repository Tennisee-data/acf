# Cookie Compliance Guide

*Last updated: 2026-01-09*

This document summarizes cookie consent requirements across major jurisdictions. Always consult legal counsel for your specific situation.

---

## Cookie Categories

Before implementing consent, categorize your cookies:

| Category | Description | Examples | Consent Required |
|----------|-------------|----------|------------------|
| **Strictly Necessary** | Essential for site function | Session ID, CSRF token, load balancer | No (but must disclose) |
| **Functional** | Remember preferences | Language, theme, font size | Varies by region |
| **Analytics** | Usage statistics | Google Analytics, Mixpanel | Yes (EU), Opt-out (US) |
| **Marketing** | Advertising/tracking | Facebook Pixel, Google Ads | Yes (everywhere) |

---

## European Union (GDPR + ePrivacy)

### Requirements

1. **Prior Consent**: Must obtain consent BEFORE setting non-essential cookies
2. **Informed**: User must understand what they're consenting to
3. **Specific**: Separate consent for each purpose/category
4. **Freely Given**: No cookie walls (can't block content for refusing)
5. **Easy Withdrawal**: As easy to withdraw as to give consent
6. **Documented**: Must store proof of consent

### Implementation Checklist

- [ ] No non-essential cookies before consent
- [ ] Clear banner explaining cookie use
- [ ] Granular options (accept all / reject all / customize)
- [ ] "Reject All" as prominent as "Accept All"
- [ ] Link to detailed cookie policy
- [ ] Cookie policy lists all cookies, purposes, durations
- [ ] Consent stored with timestamp
- [ ] Re-consent after 12 months (recommended)
- [ ] Honor DNT/GPC signals (recommended)

### Code Pattern

```python
# EU-compliant cookie consent flow

def set_analytics_cookie(response, user_consent):
    # Check consent BEFORE setting
    if not user_consent.get("analytics", False):
        return  # Don't set cookie

    response.set_cookie(
        key="analytics_id",
        value=generate_id(),
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=365 * 24 * 3600,  # 1 year
    )

# Consent storage
def store_consent(user_id, consent_data):
    """Store consent with proof for GDPR compliance."""
    db.consent_records.insert({
        "user_id": user_id,
        "timestamp": datetime.utcnow(),
        "ip_hash": hash_ip(request.remote_addr),  # Anonymized
        "consent": consent_data,  # {necessary: true, analytics: false, ...}
        "version": COOKIE_POLICY_VERSION,
    })
```

### Official Guidance

- **EDPB Guidelines**: [Guidelines 05/2020 on consent](https://edpb.europa.eu/our-work-tools/documents/public-consultations/2020/guidelines-052020-consent-under-regulation_en)
- **ICO (UK)**: [Cookies guidance](https://ico.org.uk/for-organisations/guide-to-pecr/cookies-and-similar-technologies/)
- **CNIL (France)**: [Cookie guidelines](https://www.cnil.fr/en/cookies-and-other-tracking-devices-cnil-publishes-new-guidelines)

---

## United States

### California (CCPA/CPRA)

- **Opt-out model**: Can set cookies by default (except for minors)
- **"Do Not Sell" link**: Required if selling/sharing personal data
- **Global Privacy Control**: Must honor GPC browser signal
- **Privacy policy**: Must disclose cookie use

### Other States

- **Virginia (VCDPA)**: Opt-out for targeted advertising
- **Colorado (CPA)**: Opt-out, must honor universal opt-out
- **Connecticut (CTDPA)**: Similar to Virginia

### Implementation Checklist

- [ ] Privacy policy discloses cookie use
- [ ] "Do Not Sell My Personal Information" link (if applicable)
- [ ] Honor GPC signals
- [ ] Opt-out mechanism for analytics/marketing
- [ ] No discrimination for opting out

### Code Pattern

```python
# US opt-out model

def should_set_tracking_cookie(request):
    # Check for Global Privacy Control signal
    gpc = request.headers.get("Sec-GPC", "0")
    if gpc == "1":
        return False

    # Check for opt-out cookie
    if request.cookies.get("opted_out_tracking"):
        return False

    return True  # Default: can set (opt-out model)

# Opt-out endpoint
@app.post("/privacy/opt-out")
def opt_out(response):
    response.set_cookie(
        key="opted_out_tracking",
        value="1",
        httponly=True,
        secure=True,
        max_age=365 * 24 * 3600 * 10,  # 10 years
    )
    # Also: update user record, stop data sharing
    return {"status": "opted_out"}
```

---

## Canada (PIPEDA)

- **Implied consent**: Okay for necessary cookies
- **Express consent**: Required for tracking/marketing
- **Clear disclosure**: Must explain what cookies do
- **Opt-out option**: Must provide way to withdraw consent

---

## Practical Implementation

### Minimum Viable Compliance (Global)

If targeting users globally, implement EU standard (strictest):

```python
COOKIE_CATEGORIES = {
    "necessary": {
        "consent_required": False,
        "description": "Essential for website to function",
    },
    "functional": {
        "consent_required": True,
        "description": "Remember your preferences",
    },
    "analytics": {
        "consent_required": True,
        "description": "Help us improve our website",
    },
    "marketing": {
        "consent_required": True,
        "description": "Personalized advertisements",
    },
}

def get_allowed_categories(consent_cookie):
    """Determine which cookie categories are allowed."""
    if not consent_cookie:
        # No consent yet - only necessary cookies
        return {"necessary"}

    consent = json.loads(consent_cookie)
    allowed = {"necessary"}  # Always allowed

    for category, granted in consent.items():
        if granted and category in COOKIE_CATEGORIES:
            allowed.add(category)

    return allowed

def can_set_cookie(cookie_name, consent_cookie):
    """Check if a specific cookie can be set."""
    cookie_category = COOKIE_REGISTRY.get(cookie_name, "marketing")
    allowed = get_allowed_categories(consent_cookie)
    return cookie_category in allowed
```

### Cookie Banner Best Practices

1. **First layer**: Simple accept/reject/customize
2. **Second layer**: Detailed category toggles
3. **No dark patterns**: Equal prominence for all options
4. **Persistent settings**: Remember choice for returning users
5. **Re-consent**: When policy changes or after ~12 months

---

## References

### EU/GDPR
- [GDPR Official Text](https://eur-lex.europa.eu/eli/reg/2016/679/oj)
- [ePrivacy Directive](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32002L0058)
- [EDPB Guidelines](https://edpb.europa.eu/our-work-tools/general-guidance/guidelines-recommendations-best-practices_en)

### US
- [CCPA Official Text](https://oag.ca.gov/privacy/ccpa)
- [CPRA Regulations](https://cppa.ca.gov/regulations/)
- [Global Privacy Control](https://globalprivacycontrol.org/)

### Technical
- [MDN: HTTP Cookies](https://developer.mozilla.org/en-US/docs/Web/HTTP/Cookies)
- [OWASP Session Management](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)

---

## Summary Table

| Region | Model | Necessary | Analytics | Marketing |
|--------|-------|-----------|-----------|-----------|
| EU (GDPR) | Opt-in | Auto | Consent | Consent |
| UK | Opt-in | Auto | Consent | Consent |
| California | Opt-out | Auto | Opt-out | Opt-out |
| Canada | Hybrid | Implied | Express | Express |

**When in doubt**: Implement EU-style opt-in consent. It's stricter but satisfies all jurisdictions.
