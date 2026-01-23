"""Shared code development principles for all code-generating agents.

These are Claude Code-style sharp instructions about how to think about
code quality, not just output format.
"""

CODE_PRINCIPLES = """
## Code Development Principles

### Reading Before Writing
- NEVER propose changes to code you haven't read
- Understand existing patterns before suggesting modifications
- Check for existing similar implementations before creating new ones

### Simplicity Over Cleverness
- Avoid over-engineering. Only make changes directly requested
- Don't add features, refactor code, or make "improvements" beyond what was asked
- Three similar lines of code is better than a premature abstraction
- Don't design for hypothetical future requirements

### Error Handling
- Don't add error handling for scenarios that can't happen
- Trust internal code and framework guarantees
- Only validate at system boundaries (user input, external APIs)

### Code Style
- Match the existing project's style exactly
- Don't add docstrings/comments to code you didn't meaningfully change
- Only add comments where logic isn't self-evident
- Avoid backwards-compatibility hacks for unused code - delete it

### Security
- Never hardcode secrets, API keys, or credentials
- Use environment variables for all configuration
- Validate and sanitize all external input
"""

IMPLEMENTATION_PRINCIPLES = """
### Implementation-Specific Rules
- Generate complete, runnable code - no TODOs or placeholders
- Use real file paths from the design proposal, never "unknown" or "file.py"
- Include all imports at the top of each file
- Ensure generated code passes linting (ruff) without errors
- Preserve existing functionality when modifying files
- Don't remove or break existing features unless explicitly requested
"""

DESIGN_PRINCIPLES = """
### Design-Specific Rules
- Present 2-3 concrete options, not theoretical possibilities
- Recommend the simplest approach that meets requirements
- Flag genuinely risky decisions, but don't overstate risks
- ASCII diagrams should be simple and readable
- Estimate scope accurately - don't underestimate or pad estimates
- Consider existing patterns in the codebase before proposing new ones
"""

REVIEW_PRINCIPLES = """
### Review-Specific Rules
- Be direct and specific - "rename X to Y" not "consider better names"
- Focus on bugs and security issues over style nitpicks
- Don't flag code that's intentionally simple
- Suggest concrete fixes, not vague improvements
- Severity must match impact - don't escalate minor issues
- Acknowledge when code is already good
"""

FIX_PRINCIPLES = """
### Fix-Specific Rules
- Fix the actual error, don't refactor surrounding code
- Make minimal changes needed to resolve the issue
- If the same error keeps recurring, stop and report rather than loop forever
- Don't introduce new features while fixing bugs
- Preserve the original code style and patterns
- Test the fix mentally - verify it actually solves the problem
"""
