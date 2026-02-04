# Getting Started

Create your first ACF extension in 10 minutes.

## Prerequisites

- ACF installed (`pip install acf` or from source)
- Basic Python knowledge
- A text editor

## Your First Skill: Hello World

Let's create a simple skill that adds a header comment to Python files.

### Step 1: Create the Directory

```bash
mkdir -p ~/.coding-factory/extensions/skills/hello-header
cd ~/.coding-factory/extensions/skills/hello-header
```

### Step 2: Create manifest.yaml

```yaml
name: hello-header
version: 1.0.0
type: skill
author: Your Name
description: Adds a hello header comment to Python files
license: free

skill_class: HelloHeaderSkill
input_type: files
output_type: modified_files
file_patterns: ["*.py"]

keywords:
  - hello
  - header
  - comment
```

### Step 3: Create skill.py

```python
"""Hello Header Skill - Adds header comments to Python files."""

from pathlib import Path


class HelloHeaderSkill:
    """Adds a hello header comment to Python files."""

    name = "hello-header"

    def __init__(self, llm=None, **kwargs):
        """Initialize the skill.

        Args:
            llm: Optional LLM backend (not needed for this skill)
        """
        self.llm = llm

    def run(self, files: list[Path], dry_run: bool = False) -> dict:
        """Add header to files.

        Args:
            files: List of file paths to process
            dry_run: If True, show changes without applying

        Returns:
            Results dictionary
        """
        results = {"modified": [], "skipped": []}

        header = "# Hello from my first ACF skill!\n\n"

        for file_path in files:
            content = file_path.read_text()

            # Skip if already has header
            if content.startswith("# Hello"):
                results["skipped"].append(str(file_path))
                continue

            new_content = header + content

            if not dry_run:
                file_path.write_text(new_content)

            results["modified"].append(str(file_path))

        return results
```

### Step 4: Verify Installation

```bash
acf extensions list
```

You should see:

```
Skills:
  • hello-header v1.0.0 - Adds a hello header comment to Python files
```

### Step 5: Test It

```bash
# Create a test file
echo "print('hello')" > /tmp/test.py

# Run the skill
acf skill hello-header /tmp/test.py

# Check the result
cat /tmp/test.py
```

Output:
```python
# Hello from my first ACF skill!

print('hello')
```

## Next Steps

You've created a working skill! Now explore:

- [Creating Skills](creating/skills.md) - Advanced skill features
- [Creating Agents](creating/agents.md) - Hook into pipeline stages
- [Creating Profiles](creating/profiles.md) - Define tech stack templates
- [Creating RAG Kits](creating/rag.md) - Custom code retrieval
- [Publishing Guide](publishing.md) - Share your extension

## Extension Types at a Glance

| Type | Use When | Runs |
|------|----------|------|
| **Skill** | You want to transform code files | On demand via `acf skill` |
| **Agent** | You want to add pipeline analysis | Automatically at hook points |
| **Profile** | You want to define project conventions | At pipeline start |
| **RAG Kit** | You want custom code search | During context gathering |

Choose the type that fits your use case, then follow the specific guide.
