"""Rust Stack Profile - Expert guidance for code generation.

This profile is injected into the implementation prompt when the user
selects Rust as their tech stack. It provides modern Rust patterns,
error handling best practices, and common crate usage.
"""

PROFILE_NAME = "rust"
PROFILE_VERSION = "1.0"

# Technologies covered by this profile
TECHNOLOGIES = ["rust", "cargo", "tokio", "axum", "actix"]

# Injected into the system prompt for implementation
SYSTEM_GUIDANCE = """
## Rust Expert Guidelines

You are generating Rust code. Follow these patterns exactly:

### Project Structure

```
my_project/
├── Cargo.toml
├── src/
│   ├── main.rs           # Binary entry point
│   ├── lib.rs            # Library root (if also a lib)
│   ├── error.rs          # Custom error types
│   ├── config.rs         # Configuration
│   └── modules/
│       ├── mod.rs
│       └── handler.rs
├── tests/
│   └── integration_test.rs
└── benches/
    └── benchmark.rs
```

### Error Handling (CRITICAL - Use thiserror)

```rust
// error.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },

    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Not found: {entity} with id {id}")]
    NotFound { entity: &'static str, id: String },

    #[error("Validation error: {0}")]
    Validation(String),
}

// Use Result type alias
pub type Result<T> = std::result::Result<T, AppError>;
```

### Result Propagation (Use ? operator)

```rust
// CORRECT: Use ? for error propagation
fn read_config(path: &Path) -> Result<Config> {
    let content = std::fs::read_to_string(path)?;
    let config: Config = toml::from_str(&content)?;
    Ok(config)
}

// WRONG: Manual error handling
fn read_config_bad(path: &Path) -> Result<Config> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return Err(e.into()),
    };
    // Don't do this - use ?
}
```

### Structs and Derive Macros

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: i64,
    pub email: String,
    pub name: String,
    #[serde(skip_serializing)]
    pub password_hash: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

// Builder pattern for complex construction
#[derive(Debug, Default)]
pub struct UserBuilder {
    email: Option<String>,
    name: Option<String>,
}

impl UserBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn email(mut self, email: impl Into<String>) -> Self {
        self.email = Some(email.into());
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn build(self) -> Result<User> {
        Ok(User {
            id: 0,
            email: self.email.ok_or(AppError::Validation("email required".into()))?,
            name: self.name.ok_or(AppError::Validation("name required".into()))?,
            password_hash: String::new(),
            created_at: chrono::Utc::now(),
        })
    }
}
```

### Async with Tokio

```rust
// main.rs
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Load config
    let config = Config::load()?;

    // Run application
    run(config).await
}

// Async functions
async fn fetch_user(client: &reqwest::Client, id: i64) -> Result<User> {
    let response = client
        .get(format!("https://api.example.com/users/{}", id))
        .send()
        .await?
        .error_for_status()?;

    let user: User = response.json().await?;
    Ok(user)
}

// Concurrent operations
async fn fetch_all_users(client: &reqwest::Client, ids: Vec<i64>) -> Result<Vec<User>> {
    let futures: Vec<_> = ids
        .into_iter()
        .map(|id| fetch_user(client, id))
        .collect();

    let results = futures::future::try_join_all(futures).await?;
    Ok(results)
}
```

### Axum Web Framework

```rust
// main.rs
use axum::{
    routing::{get, post},
    Router, Json, Extension,
    extract::{Path, State},
    http::StatusCode,
};
use std::sync::Arc;
use tokio::sync::RwLock;

// Application state
struct AppState {
    db: sqlx::PgPool,
    config: Config,
}

#[tokio::main]
async fn main() -> Result<()> {
    let pool = sqlx::PgPool::connect(&std::env::var("DATABASE_URL")?).await?;

    let state = Arc::new(AppState {
        db: pool,
        config: Config::load()?,
    });

    let app = Router::new()
        .route("/users", get(list_users).post(create_user))
        .route("/users/:id", get(get_user).delete(delete_user))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// Handler functions
async fn list_users(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<User>>, AppError> {
    let users = sqlx::query_as!(User, "SELECT * FROM users")
        .fetch_all(&state.db)
        .await?;
    Ok(Json(users))
}

async fn get_user(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> Result<Json<User>, AppError> {
    let user = sqlx::query_as!(User, "SELECT * FROM users WHERE id = $1", id)
        .fetch_optional(&state.db)
        .await?
        .ok_or(AppError::NotFound { entity: "User", id: id.to_string() })?;
    Ok(Json(user))
}

// Error response implementation
impl axum::response::IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match &self {
            AppError::NotFound { .. } => (StatusCode::NOT_FOUND, self.to_string()),
            AppError::Validation(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "Internal error".to_string()),
        };

        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}
```

### CLI with Clap

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "myapp")]
#[command(about = "My awesome CLI application")]
struct Cli {
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Config file path
    #[arg(short, long, default_value = "config.toml")]
    config: std::path::PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Process input files
    Process {
        /// Input files to process
        #[arg(required = true)]
        files: Vec<std::path::PathBuf>,

        /// Output directory
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
    },
    /// Initialize a new project
    Init {
        /// Directory to initialize
        #[arg(default_value = ".")]
        directory: std::path::PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Process { files, output } => {
            process_files(&files, output.as_deref())?;
        }
        Commands::Init { directory } => {
            init_project(&directory)?;
        }
    }
    Ok(())
}
```

### Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_builder() {
        let user = UserBuilder::new()
            .email("test@example.com")
            .name("Test User")
            .build()
            .unwrap();

        assert_eq!(user.email, "test@example.com");
    }

    #[tokio::test]
    async fn test_async_function() {
        let result = some_async_function().await;
        assert!(result.is_ok());
    }
}
```

### Common Mistakes to Avoid

1. **Using unwrap() in production** - Use ? or proper error handling
2. **Not using #[derive]** - Always derive Debug, Clone when appropriate
3. **Ignoring clippy warnings** - Run `cargo clippy` and fix all warnings
4. **Not using lifetimes correctly** - Prefer owned types for simplicity
5. **Missing documentation** - Use `///` doc comments on public items
6. **Not using Result/Option combinators** - Use map, and_then, ok_or

```rust
// WRONG: Unnecessary unwrap
let value = some_option.unwrap();

// CORRECT: Handle the None case
let value = some_option.ok_or(AppError::NotFound { .. })?;

// CORRECT: Provide default
let value = some_option.unwrap_or_default();
```

### Cargo.toml

```toml
[package]
name = "myapp"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.35", features = ["full"] }
axum = "0.7"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
sqlx = { version = "0.7", features = ["runtime-tokio", "postgres"] }
clap = { version = "4.4", features = ["derive"] }
chrono = { version = "0.4", features = ["serde"] }
reqwest = { version = "0.11", features = ["json"] }
```
"""

DEPENDENCIES = [
    "tokio",
    "serde",
    "serde_json",
    "thiserror",
    "anyhow",
    "tracing",
    "tracing-subscriber",
]

OPTIONAL_DEPENDENCIES = {
    "web": ["axum", "tower", "tower-http"],
    "database": ["sqlx"],
    "cli": ["clap"],
    "http": ["reqwest"],
}

TRIGGER_KEYWORDS = [
    "rust",
    "cargo",
    "tokio",
    "axum",
    "actix",
    "warp",
    "rocket",
    "clap",
]


def should_apply(tech_stack: list[str] | None, prompt: str) -> bool:
    """Determine if this profile should be applied."""
    prompt_lower = prompt.lower()

    # Check explicit tech stack selection
    if tech_stack:
        tech_lower = [t.lower() for t in tech_stack]
        if any(kw in tech_lower for kw in ["rust", "cargo", "tokio", "axum"]):
            return True

    # Check prompt keywords
    return any(kw in prompt_lower for kw in TRIGGER_KEYWORDS)


def get_guidance() -> str:
    """Get the full guidance text to inject into prompts."""
    return SYSTEM_GUIDANCE


def get_dependencies(features: list[str] | None = None) -> list[str]:
    """Get recommended dependencies."""
    deps = DEPENDENCIES.copy()

    if features:
        for feature in features:
            if feature in OPTIONAL_DEPENDENCIES:
                deps.extend(OPTIONAL_DEPENDENCIES[feature])

    return deps
