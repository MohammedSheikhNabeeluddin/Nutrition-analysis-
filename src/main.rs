use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use argon2::{
    Argon2,
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString, rand_core::OsRng},
};
use axum::{
    Extension, Json, Router,
    body::Body,
    extract::{Multipart, Request, State},
    http::{HeaderMap, StatusCode, header},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use chrono::{Duration, NaiveDate, Utc};
use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey, Header, Validation, decode, encode};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::{Row, SqlitePool, sqlite::SqlitePoolOptions};
use thiserror::Error;
use tokio::net::TcpListener;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::{error, info};

#[derive(Clone)]
struct AppState {
    pool: SqlitePool,
    jwt_secret: Arc<String>,
    vision: VisionClient,
    usda: UsdaClient,
    ai: AiClient,
    limiter: Arc<RateLimiter>,
    max_upload_size: usize,
}

#[derive(Debug, Error)]
enum AppError {
    #[error("{0}")]
    Validation(String),
    #[error("{0}")]
    Unauthorized(String),
    #[error("{0}")]
    External(String),
    #[error("too many requests")]
    TooManyRequests,
    #[error("internal server error")]
    Internal,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            AppError::Validation(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            AppError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, msg.clone()),
            AppError::External(msg) => (StatusCode::BAD_GATEWAY, msg.clone()),
            AppError::TooManyRequests => (StatusCode::TOO_MANY_REQUESTS, self.to_string()),
            AppError::Internal => {
                error!("internal error returned to client");
                (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
            }
        };

        (status, Json(json!({ "error": message }))).into_response()
    }
}

impl From<sqlx::Error> for AppError {
    fn from(err: sqlx::Error) -> Self {
        error!("sqlx error: {err}");
        AppError::Internal
    }
}

impl From<reqwest::Error> for AppError {
    fn from(err: reqwest::Error) -> Self {
        error!("reqwest error: {err}");
        AppError::External("external service request failed".to_string())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: i64,
    exp: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CurrentUser {
    id: i64,
    username: String,
}

#[derive(Debug, Deserialize)]
struct AuthPayload {
    username: String,
    password: String,
}

#[derive(Debug, Serialize)]
struct AuthResponse {
    token: String,
    expires_at: usize,
}

#[derive(Debug, Serialize, Clone)]
struct MacroTotals {
    calories: f64,
    protein_g: f64,
    carbs_g: f64,
    fat_g: f64,
}

impl Default for MacroTotals {
    fn default() -> Self {
        Self {
            calories: 0.0,
            protein_g: 0.0,
            carbs_g: 0.0,
            fat_g: 0.0,
        }
    }
}

#[derive(Debug, Serialize)]
struct AnalyzeResponse {
    foods: Vec<String>,
    macros: MacroTotals,
    score: i64,
    streak: i64,
    suggestion: String,
}

#[derive(Debug, Serialize)]
struct LeaderboardEntry {
    username: String,
    total_score: i64,
    streak: i64,
}

#[derive(Clone)]
struct VisionClient {
    http: Client,
    api_url: String,
    api_key: Option<String>,
}

impl VisionClient {
    async fn detect_foods(&self, image_b64: &str, mime: &str) -> Result<Vec<String>, AppError> {
        let api_key = self
            .api_key
            .as_ref()
            .ok_or_else(|| AppError::External("VISION_API_KEY not configured".to_string()))?;

        let response = self
            .http
            .post(&self.api_url)
            .header(header::AUTHORIZATION, format!("Bearer {api_key}"))
            .json(&json!({
                "image": {
                    "mime_type": mime,
                    "base64": image_b64
                }
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            error!("vision api failed with {status}: {body}");
            return Err(AppError::External("vision api failed".to_string()));
        }

        let value = response.json::<Value>().await?;
        parse_detected_foods(&value)
    }
}

#[derive(Clone)]
struct UsdaClient {
    http: Client,
    api_url: String,
    api_key: Option<String>,
}

impl UsdaClient {
    async fn fetch_macros_for_foods(&self, foods: &[String]) -> Result<MacroTotals, AppError> {
        let key = self
            .api_key
            .as_ref()
            .ok_or_else(|| AppError::External("USDA_API_KEY not configured".to_string()))?;

        let mut totals = MacroTotals::default();
        for food in foods {
            let response = self
                .http
                .get(format!("{}?api_key={}", self.api_url, key))
                .query(&[("query", food.as_str()), ("pageSize", "1")])
                .send()
                .await?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                error!("usda api failed with {status}: {body}");
                return Err(AppError::External("usda api failed".to_string()));
            }

            let value = response.json::<Value>().await?;
            let Some(first) = value
                .get("foods")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
            else {
                continue;
            };

            let nutrients = first
                .get("foodNutrients")
                .and_then(|n| n.as_array())
                .cloned()
                .unwrap_or_default();

            for nutrient in nutrients {
                let name = nutrient
                    .get("nutrientName")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_lowercase();
                let value = nutrient
                    .get("value")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);

                if name.contains("protein") {
                    totals.protein_g += value;
                } else if name.contains("carbohydrate") {
                    totals.carbs_g += value;
                } else if name.contains("lipid") || name.contains("fat") {
                    totals.fat_g += value;
                } else if name.contains("energy") || name.contains("calorie") {
                    totals.calories += value;
                }
            }
        }

        Ok(totals)
    }
}

#[derive(Clone)]
struct AiClient {
    http: Client,
    api_url: String,
    api_key: Option<String>,
    model: String,
}

impl AiClient {
    async fn suggest(
        &self,
        goal: Option<String>,
        macros: &MacroTotals,
        deficiencies: &[String],
    ) -> Result<String, AppError> {
        let key = self
            .api_key
            .as_ref()
            .ok_or_else(|| AppError::External("OPENROUTER_API_KEY not configured".to_string()))?;

        let goal = goal
            .map(|g| g.trim().to_string())
            .filter(|g| !g.is_empty())
            .unwrap_or_else(|| "general health".to_string());

        let deficiency_text = if deficiencies.is_empty() {
            "none identified".to_string()
        } else {
            deficiencies.join(", ")
        };

        let prompt = format!(
            "You are a nutrition coach.\nGoal: {goal}\nProtein(g): {:.1}\nCalories: {:.1}\nDeficiencies: {}\nProvide 3 concise actionable suggestions.",
            macros.protein_g, macros.calories, deficiency_text
        );

        let response = self
            .http
            .post(&self.api_url)
            .header(header::AUTHORIZATION, format!("Bearer {key}"))
            .json(&json!({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Respond with practical nutrition guidance."},
                    {"role": "user", "content": prompt}
                ]
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            error!("openrouter api failed with {status}: {body}");
            return Err(AppError::External("ai generation failed".to_string()));
        }

        let value = response.json::<Value>().await?;
        let content = value
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|msg| msg.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.trim().to_string())
            .unwrap_or_default();

        if content.is_empty() {
            return Err(AppError::External(
                "ai generation returned empty response".to_string(),
            ));
        }

        Ok(content)
    }
}

#[derive(Clone)]
struct RateLimiter {
    records: Arc<Mutex<HashMap<String, Vec<i64>>>>,
    max_requests: usize,
    window_secs: i64,
}

impl RateLimiter {
    fn new(max_requests: usize, window_secs: i64) -> Self {
        Self {
            records: Arc::new(Mutex::new(HashMap::new())),
            max_requests,
            window_secs,
        }
    }

    fn check_and_record(&self, key: &str) -> bool {
        let now = current_unix_time();
        let cutoff = now - self.window_secs;
        let mut lock = self.records.lock().expect("rate limiter mutex poisoned");
        let events = lock.entry(key.to_string()).or_default();
        events.retain(|ts| *ts >= cutoff);
        if events.len() >= self.max_requests {
            return false;
        }
        events.push(now);
        true
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "nutrition_analysis=info,tower_http=info".to_string()),
        )
        .init();

    let database_url =
        std::env::var("DATABASE_URL").unwrap_or_else(|_| "sqlite://nutrition.db".to_string());
    let jwt_secret =
        std::env::var("JWT_SECRET").unwrap_or_else(|_| "replace-me-in-production".to_string());
    let bind = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:3000".to_string());

    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    init_db(&pool).await?;

    let state = AppState {
        pool,
        jwt_secret: Arc::new(jwt_secret),
        vision: VisionClient {
            http: Client::new(),
            api_url: std::env::var("VISION_API_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1/responses".to_string()),
            api_key: std::env::var("VISION_API_KEY").ok(),
        },
        usda: UsdaClient {
            http: Client::new(),
            api_url: std::env::var("USDA_API_URL")
                .unwrap_or_else(|_| "https://api.nal.usda.gov/fdc/v1/foods/search".to_string()),
            api_key: std::env::var("USDA_API_KEY").ok(),
        },
        ai: AiClient {
            http: Client::new(),
            api_url: std::env::var("OPENROUTER_API_URL")
                .unwrap_or_else(|_| "https://openrouter.ai/api/v1/chat/completions".to_string()),
            api_key: std::env::var("OPENROUTER_API_KEY").ok(),
            model: std::env::var("OPENROUTER_MODEL")
                .unwrap_or_else(|_| "openai/gpt-4o-mini".to_string()),
        },
        limiter: Arc::new(RateLimiter::new(20, 60)),
        max_upload_size: std::env::var("MAX_UPLOAD_SIZE_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(5 * 1024 * 1024),
    };

    let protected = Router::new()
        .route("/api/analyze", post(analyze_meal))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/style.css", get(style_handler))
        .route("/script.js", get(script_handler))
        .route("/api/health", get(|| async { Json(json!({"ok": true})) }))
        .route("/api/signup", post(signup))
        .route("/api/login", post(login))
        .route("/api/leaderboard", get(leaderboard))
        .merge(protected)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = TcpListener::bind(&bind).await?;
    let bound = listener.local_addr()?;
    info!("server listening on {bound}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../index.html"))
}

async fn style_handler() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("../style.css"),
    )
}

async fn script_handler() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        include_str!("../script.js"),
    )
}

async fn signup(
    State(state): State<AppState>,
    Json(payload): Json<AuthPayload>,
) -> Result<Json<AuthResponse>, AppError> {
    validate_auth_payload(&payload)?;

    let existing = sqlx::query("SELECT id FROM users WHERE username = ?")
        .bind(&payload.username)
        .fetch_optional(&state.pool)
        .await?;
    if existing.is_some() {
        return Err(AppError::Validation(
            "username is already taken".to_string(),
        ));
    }

    let hashed = hash_password(&payload.password)?;
    let result = sqlx::query(
        "INSERT INTO users (username, password_hash, streak, last_scan_date) VALUES (?, ?, 0, NULL)",
    )
    .bind(&payload.username)
    .bind(&hashed)
    .execute(&state.pool)
    .await?;

    let user_id = result.last_insert_rowid();
    let (token, expires_at) = issue_jwt(user_id, &state.jwt_secret)?;

    Ok(Json(AuthResponse { token, expires_at }))
}

async fn login(
    State(state): State<AppState>,
    Json(payload): Json<AuthPayload>,
) -> Result<Json<AuthResponse>, AppError> {
    validate_auth_payload(&payload)?;

    let user = sqlx::query("SELECT id, password_hash FROM users WHERE username = ?")
        .bind(&payload.username)
        .fetch_optional(&state.pool)
        .await?
        .ok_or_else(|| AppError::Unauthorized("invalid credentials".to_string()))?;

    let user_id: i64 = user.get("id");
    let hash: String = user.get("password_hash");

    verify_password(&payload.password, &hash)?;
    let (token, expires_at) = issue_jwt(user_id, &state.jwt_secret)?;

    Ok(Json(AuthResponse { token, expires_at }))
}

async fn auth_middleware(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut request: Request<Body>,
    next: Next,
) -> Result<Response, AppError> {
    let auth_header = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| AppError::Unauthorized("missing authorization header".to_string()))?;

    let token = auth_header
        .strip_prefix("Bearer ")
        .ok_or_else(|| AppError::Unauthorized("invalid authorization scheme".to_string()))?;

    let decoded = decode::<Claims>(
        token,
        &DecodingKey::from_secret(state.jwt_secret.as_bytes()),
        &Validation::new(Algorithm::HS256),
    )
    .map_err(|_| AppError::Unauthorized("invalid or expired token".to_string()))?;

    let row = sqlx::query("SELECT id, username FROM users WHERE id = ?")
        .bind(decoded.claims.sub)
        .fetch_optional(&state.pool)
        .await?
        .ok_or_else(|| AppError::Unauthorized("user not found".to_string()))?;

    let user = CurrentUser {
        id: row.get("id"),
        username: row.get("username"),
    };
    request.extensions_mut().insert(user);

    Ok(next.run(request).await)
}

async fn analyze_meal(
    State(state): State<AppState>,
    Extension(user): Extension<CurrentUser>,
    mut multipart: Multipart,
) -> Result<Json<AnalyzeResponse>, AppError> {
    if !state
        .limiter
        .check_and_record(&format!("upload:{}", user.id))
    {
        return Err(AppError::TooManyRequests);
    }

    let mut image_bytes: Option<Vec<u8>> = None;
    let mut image_mime: Option<String> = None;
    let mut goal: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|_| AppError::Validation("invalid multipart body".to_string()))?
    {
        let name = field.name().unwrap_or_default().to_string();
        if name == "image" {
            let content_type = field.content_type().unwrap_or_default().to_lowercase();
            validate_image_content_type(&content_type)?;
            let bytes = field
                .bytes()
                .await
                .map_err(|_| AppError::Validation("failed to read image bytes".to_string()))?;
            if bytes.len() > state.max_upload_size {
                return Err(AppError::Validation(format!(
                    "image exceeds max size of {} bytes",
                    state.max_upload_size
                )));
            }
            image_bytes = Some(bytes.to_vec());
            image_mime = Some(content_type);
        } else if name == "goal" {
            let value = field.text().await.unwrap_or_default();
            if !value.trim().is_empty() {
                goal = Some(value.trim().chars().take(120).collect());
            }
        }
    }

    let image =
        image_bytes.ok_or_else(|| AppError::Validation("image field is required".to_string()))?;
    let image_mime = image_mime
        .ok_or_else(|| AppError::Validation("image content type is required".to_string()))?;
    let image_b64 = BASE64.encode(image);

    let foods = state.vision.detect_foods(&image_b64, &image_mime).await?;
    if foods.is_empty() {
        return Err(AppError::Validation("no foods detected".to_string()));
    }

    let macros = state.usda.fetch_macros_for_foods(&foods).await?;
    let deficiencies = infer_deficiencies(&macros);

    if !state.limiter.check_and_record(&format!("ai:{}", user.id)) {
        return Err(AppError::TooManyRequests);
    }
    let suggestion = state
        .ai
        .suggest(goal.clone(), &macros, &deficiencies)
        .await?;

    let score = ((macros.protein_g * 2.0) + (macros.calories / 10.0) - (macros.fat_g * 0.5))
        .max(0.0) as i64;
    let today = Utc::now().date_naive();

    let row = sqlx::query("SELECT streak, last_scan_date FROM users WHERE id = ?")
        .bind(user.id)
        .fetch_one(&state.pool)
        .await?;

    let current_streak: i64 = row.get("streak");
    let last_scan_date = row
        .get::<Option<String>, _>("last_scan_date")
        .and_then(|v| NaiveDate::parse_from_str(&v, "%Y-%m-%d").ok());

    let new_streak = calculate_new_streak(current_streak, last_scan_date, today);

    sqlx::query("UPDATE users SET streak = ?, last_scan_date = ? WHERE id = ?")
        .bind(new_streak)
        .bind(today.to_string())
        .bind(user.id)
        .execute(&state.pool)
        .await?;

    sqlx::query(
        "INSERT INTO food_logs (user_id, foods, calories, protein_g, carbs_g, fat_g, score, suggestion, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(user.id)
    .bind(serde_json::to_string(&foods).unwrap_or_else(|_| "[]".to_string()))
    .bind(macros.calories)
    .bind(macros.protein_g)
    .bind(macros.carbs_g)
    .bind(macros.fat_g)
    .bind(score)
    .bind(&suggestion)
    .bind(Utc::now().to_rfc3339())
    .execute(&state.pool)
    .await?;

    Ok(Json(AnalyzeResponse {
        foods,
        macros,
        score,
        streak: new_streak,
        suggestion,
    }))
}

async fn leaderboard(
    State(state): State<AppState>,
) -> Result<Json<Vec<LeaderboardEntry>>, AppError> {
    let rows = sqlx::query(
        "SELECT u.username, COALESCE(SUM(fl.score), 0) AS total_score, u.streak
         FROM users u
         LEFT JOIN food_logs fl ON fl.user_id = u.id
         GROUP BY u.id, u.username, u.streak
         ORDER BY total_score DESC, u.streak DESC, u.username ASC
         LIMIT 20",
    )
    .fetch_all(&state.pool)
    .await?;

    let items = rows
        .into_iter()
        .map(|row| LeaderboardEntry {
            username: row.get("username"),
            total_score: row.get("total_score"),
            streak: row.get("streak"),
        })
        .collect::<Vec<_>>();

    Ok(Json(items))
}

async fn init_db(pool: &SqlitePool) -> Result<(), AppError> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            streak INTEGER NOT NULL DEFAULT 0,
            last_scan_date TEXT NULL
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS food_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            foods TEXT NOT NULL,
            calories REAL NOT NULL,
            protein_g REAL NOT NULL,
            carbs_g REAL NOT NULL,
            fat_g REAL NOT NULL,
            score INTEGER NOT NULL,
            suggestion TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )",
    )
    .execute(pool)
    .await?;

    Ok(())
}

fn validate_auth_payload(payload: &AuthPayload) -> Result<(), AppError> {
    if payload.username.trim().len() < 3 || payload.username.trim().len() > 50 {
        return Err(AppError::Validation(
            "username must be between 3 and 50 characters".to_string(),
        ));
    }
    if payload.password.len() < 8 {
        return Err(AppError::Validation(
            "password must be at least 8 characters".to_string(),
        ));
    }
    Ok(())
}

fn validate_image_content_type(content_type: &str) -> Result<(), AppError> {
    let allowed = ["image/jpeg", "image/png", "image/webp"];
    if !allowed.iter().any(|ct| *ct == content_type) {
        return Err(AppError::Validation(
            "unsupported file type; use JPEG, PNG, or WEBP".to_string(),
        ));
    }
    Ok(())
}

fn hash_password(password: &str) -> Result<String, AppError> {
    let salt = SaltString::generate(&mut OsRng);
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|hash| hash.to_string())
        .map_err(|_| AppError::Internal)
}

fn verify_password(password: &str, hash: &str) -> Result<(), AppError> {
    let parsed = PasswordHash::new(hash)
        .map_err(|_| AppError::Unauthorized("invalid credentials".to_string()))?;
    Argon2::default()
        .verify_password(password.as_bytes(), &parsed)
        .map_err(|_| AppError::Unauthorized("invalid credentials".to_string()))
}

fn issue_jwt(user_id: i64, secret: &str) -> Result<(String, usize), AppError> {
    let expires_at = (Utc::now() + Duration::hours(24)).timestamp() as usize;
    let claims = Claims {
        sub: user_id,
        exp: expires_at,
    };
    let token = encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .map_err(|_| AppError::Internal)?;

    Ok((token, expires_at))
}

fn parse_detected_foods(value: &Value) -> Result<Vec<String>, AppError> {
    if let Some(foods) = value.get("foods").and_then(|v| v.as_array()) {
        let items = foods
            .iter()
            .filter_map(|f| f.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();
        if !items.is_empty() {
            return Ok(items);
        }
    }

    if let Some(content) = value
        .get("output_text")
        .and_then(|v| v.as_str())
        .or_else(|| {
            value
                .get("choices")
                .and_then(|c| c.as_array())
                .and_then(|arr| arr.first())
                .and_then(|choice| choice.get("message"))
                .and_then(|msg| msg.get("content"))
                .and_then(|c| c.as_str())
        })
    {
        let items = content
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();
        if !items.is_empty() {
            return Ok(items);
        }
    }

    Err(AppError::External(
        "vision api returned unrecognized response".to_string(),
    ))
}

fn infer_deficiencies(macros: &MacroTotals) -> Vec<String> {
    let mut result = Vec::new();
    if macros.protein_g < 20.0 {
        result.push("low protein".to_string());
    }
    if macros.calories < 300.0 {
        result.push("low energy".to_string());
    }
    if macros.carbs_g < 20.0 {
        result.push("low carbohydrates".to_string());
    }
    result
}

fn calculate_new_streak(
    current_streak: i64,
    last_scan_date: Option<NaiveDate>,
    today: NaiveDate,
) -> i64 {
    match last_scan_date {
        Some(last) if last == today => current_streak,
        Some(last) if last == (today - Duration::days(1)) => current_streak + 1,
        _ => 1,
    }
}

fn current_unix_time() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streak_increments_for_yesterday() {
        let today = NaiveDate::from_ymd_opt(2026, 3, 27).unwrap();
        let yesterday = today - Duration::days(1);
        assert_eq!(calculate_new_streak(3, Some(yesterday), today), 4);
    }

    #[test]
    fn streak_stays_for_same_day() {
        let today = NaiveDate::from_ymd_opt(2026, 3, 27).unwrap();
        assert_eq!(calculate_new_streak(5, Some(today), today), 5);
    }

    #[test]
    fn streak_resets_after_gap() {
        let today = NaiveDate::from_ymd_opt(2026, 3, 27).unwrap();
        let older = today - Duration::days(3);
        assert_eq!(calculate_new_streak(7, Some(older), today), 1);
    }

    #[test]
    fn rate_limiter_blocks_when_exceeded() {
        let limiter = RateLimiter::new(2, 60);
        assert!(limiter.check_and_record("u1"));
        assert!(limiter.check_and_record("u1"));
        assert!(!limiter.check_and_record("u1"));
    }

    #[test]
    fn parse_vision_foods_array() {
        let value = json!({"foods": ["rice", "chicken"]});
        let foods = parse_detected_foods(&value).unwrap();
        assert_eq!(foods, vec!["rice", "chicken"]);
    }
}
