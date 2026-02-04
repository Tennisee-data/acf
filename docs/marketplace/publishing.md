# Publishing Guide

Submit your extension to the ACF Marketplace.

## Before You Submit

### Checklist

- [ ] Extension works locally (`acf extensions list` shows it)
- [ ] All required manifest fields are present
- [ ] Code follows security guidelines
- [ ] README.md documents usage
- [ ] Tested with at least one full pipeline run

### Test Locally

```bash
# Verify extension loads
acf extensions list

# Check for conflicts
acf extensions check-conflicts

# Validate manifest
acf extensions validate ./my-extension

# Run a test pipeline
acf run "Build a simple feature" --auto-approve
```

## Packaging

### Create Package

```bash
cd my-extension
tar -czvf my-extension-1.0.0.tar.gz .
```

Package contents:
```
my-extension-1.0.0.tar.gz
├── manifest.yaml
├── skill.py (or agent.py, profile.py, retriever.py)
├── requirements.txt (optional)
└── README.md (recommended)
```

### Verify Package

```bash
# Extract and check
mkdir /tmp/verify && cd /tmp/verify
tar -xzvf /path/to/my-extension-1.0.0.tar.gz
cat manifest.yaml
```

## Submit

### Command Line

```bash
acf marketplace submit ./my-extension-1.0.0.tar.gz \
  --name "my-extension" \
  --version "1.0.0" \
  --type skill \
  --description "What it does"

# For commercial extensions
acf marketplace submit ./my-extension-1.0.0.tar.gz \
  --name "my-extension" \
  --version "1.0.0" \
  --type skill \
  --description "What it does" \
  --price 19.00
```

### Required Information

| Field | Description |
|-------|-------------|
| `name` | Unique extension name |
| `version` | Semantic version |
| `type` | skill, agent, profile, or rag |
| `description` | Short description |
| `price` | Price in USD (omit for free) |

## Review Process

### 1. Automated Security Scan

Your extension is automatically scanned for:
- Malicious code patterns
- Unsafe operations (eval, exec, etc.)
- Network calls to suspicious endpoints
- File operations outside project scope

### 2. Manual Review

The ACF team reviews:
- Code quality and best practices
- Documentation completeness
- Functionality as described
- Security considerations

### 3. Approval or Feedback

- **Approved**: Extension goes live on marketplace
- **Changes Requested**: You'll receive specific feedback

Typical review time: 2-5 business days

## Pricing Guidelines

### Free Extensions

Great for:
- Building reputation
- Simple utilities
- Community contributions

### Commercial Extensions

| Complexity | Suggested Price |
|------------|-----------------|
| Simple utility | $5 - $10 |
| Standard feature | $10 - $25 |
| Advanced feature | $25 - $49 |
| Enterprise-grade | $49+ |

### Revenue Split

- **You keep: 82.35%**
- Platform fee: 17.65% (covers Stripe/PayPal fees + infrastructure)

Example: $29.00 extension = $23.88 to you

## Updates

### Version Updates

```bash
# Update version in manifest.yaml
# Then submit new package

acf marketplace submit ./my-extension-1.1.0.tar.gz \
  --name "my-extension" \
  --version "1.1.0"
```

### Changelog

Include a CHANGELOG.md:

```markdown
# Changelog

## 1.1.0 - 2024-01-15
### Added
- New feature X
### Fixed
- Bug in Y

## 1.0.0 - 2024-01-01
- Initial release
```

## Security Requirements

### Do

- Use safe file operations
- Validate all inputs
- Handle errors gracefully
- Document any network calls
- Keep dependencies updated

### Don't

- Use `eval()` or `exec()` with user input
- Make undocumented network requests
- Access files outside project directory
- Store credentials in code
- Use deprecated/vulnerable packages

### Example: Safe File Operations

```python
# Good - stays within project
from pathlib import Path

def read_file(project_dir: Path, filename: str) -> str:
    filepath = (project_dir / filename).resolve()
    # Verify still within project
    if not str(filepath).startswith(str(project_dir.resolve())):
        raise ValueError("Path traversal detected")
    return filepath.read_text()
```

## Tips for Success

1. **Solve a Real Problem**
   - Focus on common pain points
   - Look at GitHub issues for ideas

2. **Write Good Documentation**
   - Clear README with examples
   - Document all options
   - Include troubleshooting tips

3. **Start Free**
   - Build reputation first
   - Get feedback from users
   - Add commercial features later

4. **Respond to Feedback**
   - Monitor reviews
   - Fix issues quickly
   - Thank users for reports

5. **Keep It Updated**
   - Fix security issues promptly
   - Support new ACF versions
   - Add requested features

## Support

Questions about publishing?

- [GitHub Discussions](https://github.com/Tennisee-data/acf/discussions)
- [Issues](https://github.com/Tennisee-data/acf/issues)
