"""Security utilities for safe file handling.

YAML Deserialization Attacks:
- yaml.load() without Loader can execute arbitrary Python code
- Attackers can inject payloads like: !!python/object/apply:os.system ['rm -rf /']
- Even docker-compose.yml files can be vectors if parsed unsafely

Safe Practices:
- ALWAYS use yaml.safe_load() or yaml.load(data, Loader=yaml.SafeLoader)
- NEVER use yaml.load() without explicit SafeLoader
- Validate file paths to prevent path traversal
- Sanitize user input before writing to config files
"""

import re
from pathlib import Path
from typing import Any


# Dangerous YAML patterns that indicate potential exploits
YAML_EXPLOIT_PATTERNS = [
    r'!!python/',           # Python object instantiation
    r'!!ruby/',             # Ruby object instantiation
    r'!!perl/',             # Perl object instantiation
    r'!!java/',             # Java object instantiation
    r'!!\w+/object',        # Generic object tags
    r'__reduce__',          # Pickle-style attacks
    r'subprocess',          # Command execution
    r'os\.system',          # Shell execution
    r'eval\s*\(',           # Code evaluation
    r'exec\s*\(',           # Code execution
    r'__import__',          # Dynamic imports
    r'getattr\s*\(',        # Attribute access exploits
]


def safe_yaml_load(content: str) -> Any:
    """Safely load YAML content.

    Uses yaml.safe_load which only allows basic Python types:
    - strings, integers, floats, booleans, None
    - lists, dicts
    - dates (as strings)

    Does NOT allow:
    - Arbitrary Python object instantiation
    - Function calls
    - Custom tags

    Args:
        content: YAML string to parse

    Returns:
        Parsed YAML data (dict, list, or scalar)

    Raises:
        ValueError: If content contains suspicious patterns
        yaml.YAMLError: If YAML is malformed
    """
    # Pre-scan for exploit patterns
    for pattern in YAML_EXPLOIT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            raise ValueError(f"Suspicious YAML pattern detected: {pattern}")

    try:
        import yaml
        return yaml.safe_load(content)
    except ImportError:
        # Fallback: refuse to parse if PyYAML not installed
        raise ImportError(
            "PyYAML not installed. Install with: pip install PyYAML"
        )


def safe_yaml_load_file(file_path: Path | str) -> Any:
    """Safely load YAML from a file.

    Args:
        file_path: Path to YAML file

    Returns:
        Parsed YAML data

    Raises:
        ValueError: If path traversal detected or suspicious content
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path).resolve()

    # Prevent path traversal
    if ".." in str(file_path):
        raise ValueError(f"Path traversal detected in: {file_path}")

    content = path.read_text(encoding='utf-8')
    return safe_yaml_load(content)


def sanitize_yaml_value(value: str) -> str:
    """Sanitize a string value for safe YAML inclusion.

    Escapes or removes potentially dangerous characters/sequences.

    Args:
        value: String to sanitize

    Returns:
        Sanitized string safe for YAML
    """
    if not isinstance(value, str):
        return str(value)

    # Remove YAML special characters that could alter structure
    dangerous_chars = [':', '{', '}', '[', ']', '&', '*', '!', '|', '>', "'", '"']

    result = value
    for char in dangerous_chars:
        result = result.replace(char, '')

    # Remove any YAML tags
    result = re.sub(r'!!\w+', '', result)

    # Collapse whitespace
    result = ' '.join(result.split())

    return result.strip()


def validate_docker_compose(content: str) -> tuple[bool, list[str]]:
    """Validate docker-compose.yml for security issues.

    Checks for:
    - Privileged containers
    - Host network mode
    - Dangerous volume mounts
    - Exposed sensitive paths
    - Environment variable leaks

    Args:
        content: docker-compose.yml content

    Returns:
        Tuple of (is_safe, list of warnings)
    """
    warnings = []

    # Check for privileged mode
    if re.search(r'privileged:\s*true', content, re.IGNORECASE):
        warnings.append("CRITICAL: Container runs in privileged mode")

    # Check for host network
    if re.search(r'network_mode:\s*["\']?host', content, re.IGNORECASE):
        warnings.append("WARNING: Container uses host network")

    # Check for dangerous volume mounts
    dangerous_mounts = [
        r'/etc:/\w+',           # Host /etc
        r'/var/run/docker',     # Docker socket
        r'/root:/\w+',          # Root home
        r'/:/\w+:rw',           # Root filesystem writable
        r'\.ssh:/\w+',          # SSH keys
    ]
    for mount in dangerous_mounts:
        if re.search(mount, content):
            warnings.append(f"CRITICAL: Dangerous volume mount pattern: {mount}")

    # Check for cap_add
    if re.search(r'cap_add:', content):
        warnings.append("WARNING: Container adds Linux capabilities")
        if re.search(r'SYS_ADMIN|ALL', content):
            warnings.append("CRITICAL: Container has SYS_ADMIN or ALL capabilities")

    # Check for user: root
    if re.search(r'user:\s*["\']?root', content, re.IGNORECASE):
        warnings.append("WARNING: Container runs as root user")

    # Check for secrets in environment
    env_secrets = re.findall(
        r'(\w*(?:PASSWORD|SECRET|KEY|TOKEN|CREDENTIAL)\w*)\s*[:=]',
        content,
        re.IGNORECASE
    )
    if env_secrets:
        warnings.append(
            f"INFO: Sensitive env vars detected (ensure not hardcoded): {', '.join(set(env_secrets))}"
        )

    is_safe = not any(w.startswith("CRITICAL") for w in warnings)
    return is_safe, warnings


def validate_dockerfile(content: str) -> tuple[bool, list[str]]:
    """Validate Dockerfile for security issues.

    Args:
        content: Dockerfile content

    Returns:
        Tuple of (is_safe, list of warnings)
    """
    warnings = []

    # Check for root user
    if not re.search(r'USER\s+\w+', content) or re.search(r'USER\s+root', content):
        warnings.append("WARNING: Container may run as root (no USER directive)")

    # Check for latest tag
    if re.search(r'FROM\s+\w+:latest', content, re.IGNORECASE):
        warnings.append("WARNING: Using :latest tag - pin to specific version")

    # Check for curl | bash patterns
    if re.search(r'curl.*\|\s*(?:ba)?sh', content):
        warnings.append("CRITICAL: curl piped to shell - potential RCE vector")

    # Check for ADD with URL (prefer COPY)
    if re.search(r'ADD\s+https?://', content):
        warnings.append("WARNING: ADD with URL - prefer COPY + explicit download")

    # Check for secrets in build args
    if re.search(r'ARG\s+\w*(?:PASSWORD|SECRET|KEY|TOKEN)\w*', content, re.IGNORECASE):
        warnings.append("CRITICAL: Secrets in build args are cached in layers")

    # Check for COPY of sensitive files
    if re.search(r'COPY.*(?:\.env|\.ssh|\.aws|credentials)', content, re.IGNORECASE):
        warnings.append("CRITICAL: Copying sensitive files into image")

    is_safe = not any(w.startswith("CRITICAL") for w in warnings)
    return is_safe, warnings


def scan_generated_files(project_dir: Path) -> dict[str, list[str]]:
    """Scan generated project files for security issues.

    Args:
        project_dir: Path to project directory

    Returns:
        Dict mapping filename to list of warnings
    """
    results = {}

    # Check docker-compose.yml
    compose_file = project_dir / "docker-compose.yml"
    if compose_file.exists():
        _, warnings = validate_docker_compose(compose_file.read_text())
        if warnings:
            results["docker-compose.yml"] = warnings

    # Check Dockerfile
    dockerfile = project_dir / "Dockerfile"
    if dockerfile.exists():
        _, warnings = validate_dockerfile(dockerfile.read_text())
        if warnings:
            results["Dockerfile"] = warnings

    # Check .env files for hardcoded secrets
    for env_file in project_dir.glob(".env*"):
        if env_file.name == ".env.example":
            continue
        content = env_file.read_text()
        if re.search(r'(?:PASSWORD|SECRET|KEY)=\S+', content):
            results[env_file.name] = [
                "WARNING: Secrets detected in env file - ensure not committed"
            ]

    return results


def run_bandit_scan(project_dir: Path) -> dict[str, Any]:
    """Run bandit security scanner on Python files.

    Bandit is a tool designed to find common security issues in Python code.

    Args:
        project_dir: Path to project directory

    Returns:
        Dict with scan results including issues found
    """
    import subprocess
    import json

    try:
        # Check if bandit is available
        check = subprocess.run(
            ["bandit", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode != 0:
            return {
                "success": False,
                "error": "bandit not installed. Install with: pip install bandit",
                "issues": [],
            }

        # Run bandit with JSON output
        result = subprocess.run(
            [
                "bandit",
                "-r",  # Recursive
                str(project_dir),
                "-f", "json",  # JSON format
                "-ll",  # Only medium and above
                "--exclude", "**/tests/**,**/*_test.py,**/test_*.py",  # Skip tests
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        # Parse JSON output
        try:
            output = json.loads(result.stdout) if result.stdout else {}
        except json.JSONDecodeError:
            output = {}

        issues = output.get("results", [])

        # Format issues
        formatted_issues = []
        for issue in issues:
            severity = issue.get("issue_severity", "UNKNOWN")
            confidence = issue.get("issue_confidence", "UNKNOWN")
            text = issue.get("issue_text", "")
            filename = issue.get("filename", "")
            line = issue.get("line_number", 0)

            formatted_issues.append({
                "severity": severity,
                "confidence": confidence,
                "message": text,
                "file": filename,
                "line": line,
                "code": issue.get("code", ""),
                "test_id": issue.get("test_id", ""),
            })

        return {
            "success": True,
            "issues": formatted_issues,
            "metrics": output.get("metrics", {}),
            "high_count": sum(1 for i in formatted_issues if i["severity"] == "HIGH"),
            "medium_count": sum(1 for i in formatted_issues if i["severity"] == "MEDIUM"),
            "low_count": sum(1 for i in formatted_issues if i["severity"] == "LOW"),
        }

    except FileNotFoundError:
        return {
            "success": False,
            "error": "bandit not found. Install with: pip install bandit",
            "issues": [],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "issues": [],
        }


def check_dependency_deprecation(requirements_file: Path) -> list[dict]:
    """Check if dependencies are deprecated or have known vulnerabilities.

    Uses pip-audit or safety to check packages.

    Args:
        requirements_file: Path to requirements.txt

    Returns:
        List of deprecation/vulnerability warnings
    """
    import subprocess

    warnings = []

    if not requirements_file.exists():
        return warnings

    # Parse requirements
    packages = []
    for line in requirements_file.read_text().split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            # Extract package name (before ==, >=, etc.)
            match = re.match(r'^([a-zA-Z0-9_-]+)', line)
            if match:
                packages.append(match.group(1))

    # Known deprecated packages
    deprecated_packages = {
        "pycrypto": "Use pycryptodome instead",
        "pylint": None,  # Not deprecated, just example
        "nose": "Use pytest instead",
        "distribute": "Use setuptools instead",
        "argparse": "Built into Python 3, no need to install",
        "typing": "Built into Python 3.5+, no need to install",
        "pathlib": "Built into Python 3.4+, no need to install",
    }

    for pkg in packages:
        pkg_lower = pkg.lower()
        if pkg_lower in deprecated_packages:
            msg = deprecated_packages[pkg_lower]
            if msg:
                warnings.append({
                    "package": pkg,
                    "type": "deprecated",
                    "message": msg,
                })

    # Try pip-audit for vulnerability scanning
    try:
        result = subprocess.run(
            ["pip-audit", "--requirement", str(requirements_file), "--format", "json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

        if result.returncode == 0 and result.stdout:
            import json
            try:
                audit_data = json.loads(result.stdout)
                for vuln in audit_data.get("dependencies", []):
                    for v in vuln.get("vulns", []):
                        warnings.append({
                            "package": vuln.get("name"),
                            "type": "vulnerability",
                            "message": f"{v.get('id')}: {v.get('description', '')}",
                            "fix_versions": v.get("fix_versions", []),
                        })
            except json.JSONDecodeError:
                pass

    except (FileNotFoundError, subprocess.TimeoutExpired):
        # pip-audit not available or timed out
        pass

    return warnings


def full_security_scan(project_dir: Path) -> dict[str, Any]:
    """Run a comprehensive security scan on a project.

    Includes:
    - Docker/compose file validation
    - Bandit Python security scan
    - Dependency vulnerability check

    Args:
        project_dir: Path to project directory

    Returns:
        Complete scan results
    """
    project_dir = Path(project_dir)

    results = {
        "docker_issues": scan_generated_files(project_dir),
        "bandit_scan": run_bandit_scan(project_dir),
        "dependency_warnings": [],
        "summary": {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        },
    }

    # Check dependencies
    req_file = project_dir / "requirements.txt"
    if req_file.exists():
        results["dependency_warnings"] = check_dependency_deprecation(req_file)

    # Count severities
    for file_warnings in results["docker_issues"].values():
        for w in file_warnings:
            if "CRITICAL" in w:
                results["summary"]["critical"] += 1
            elif "WARNING" in w:
                results["summary"]["medium"] += 1
            else:
                results["summary"]["info"] += 1

    bandit = results["bandit_scan"]
    if bandit.get("success"):
        results["summary"]["high"] += bandit.get("high_count", 0)
        results["summary"]["medium"] += bandit.get("medium_count", 0)
        results["summary"]["low"] += bandit.get("low_count", 0)

    for dep_warn in results["dependency_warnings"]:
        if dep_warn["type"] == "vulnerability":
            results["summary"]["high"] += 1
        else:
            results["summary"]["medium"] += 1

    return results
