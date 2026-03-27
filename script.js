const TOKEN_KEY = 'nutrition_token';
const TOKEN_EXP_KEY = 'nutrition_token_exp';

const signupForm = document.getElementById('signup-form');
const loginForm = document.getElementById('login-form');
const analyzeForm = document.getElementById('analyze-form');
const logoutBtn = document.getElementById('logout-btn');
const authState = document.getElementById('auth-state');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loading-text');
const toast = document.getElementById('toast');
const resultsCard = document.getElementById('results-card');
const refreshLeaderboardBtn = document.getElementById('refresh-leaderboard');
const leaderboardEl = document.getElementById('leaderboard');

function showToast(message, isError = false) {
  toast.textContent = message;
  toast.classList.remove('hidden', 'error');
  if (isError) toast.classList.add('error');
  setTimeout(() => toast.classList.add('hidden'), 3500);
}

function setLoading(isLoading, text = 'Loading...') {
  loading.classList.toggle('hidden', !isLoading);
  loadingText.textContent = text;
}

function persistToken(token, expiresAt) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(TOKEN_EXP_KEY, String(expiresAt));
  updateAuthState();
}

function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(TOKEN_EXP_KEY);
  updateAuthState();
}

function getToken() {
  const token = localStorage.getItem(TOKEN_KEY);
  const exp = Number(localStorage.getItem(TOKEN_EXP_KEY) || 0);
  if (!token || !exp) return null;
  if (Math.floor(Date.now() / 1000) >= exp) {
    clearToken();
    showToast('Session expired. Please login again.', true);
    return null;
  }
  return token;
}

function updateAuthState() {
  const token = localStorage.getItem(TOKEN_KEY);
  authState.textContent = token ? 'Logged in' : 'Not logged in';
}

async function apiRequest(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = getToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);

  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const data = await response.json();
      if (data?.error) message = data.error;
    } catch {}

    if (response.status === 401) {
      clearToken();
      showToast('Session expired. Please login again.', true);
    }
    throw new Error(message);
  }

  return response.json();
}

signupForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const username = document.getElementById('signup-username').value.trim();
  const password = document.getElementById('signup-password').value;

  try {
    const data = await apiRequest('/api/signup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    persistToken(data.token, data.expires_at);
    showToast('Signup successful.');
  } catch (err) {
    showToast(err.message, true);
  }
});

loginForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const username = document.getElementById('login-username').value.trim();
  const password = document.getElementById('login-password').value;

  try {
    const data = await apiRequest('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    persistToken(data.token, data.expires_at);
    showToast('Login successful.');
  } catch (err) {
    showToast(err.message, true);
  }
});

logoutBtn.addEventListener('click', () => {
  clearToken();
  showToast('Logged out.');
});

analyzeForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const token = getToken();
  if (!token) {
    showToast('Please login first.', true);
    return;
  }

  const imageInput = document.getElementById('meal-image');
  const goal = document.getElementById('goal').value.trim();
  if (!imageInput.files?.length) {
    showToast('Please select an image.', true);
    return;
  }

  const file = imageInput.files[0];
  const allowed = ['image/jpeg', 'image/png', 'image/webp'];
  if (!allowed.includes(file.type)) {
    showToast('Unsupported image type. Use JPEG, PNG, or WEBP.', true);
    return;
  }

  const form = new FormData();
  form.append('image', file);
  if (goal) form.append('goal', goal);

  try {
    setLoading(true, 'Processing image...');
    const data = await apiRequest('/api/analyze', {
      method: 'POST',
      body: form,
    });

    document.getElementById('foods').textContent = data.foods.join(', ');
    document.getElementById('calories').textContent = Number(data.macros.calories || 0).toFixed(1);
    document.getElementById('protein').textContent = Number(data.macros.protein_g || 0).toFixed(1);
    document.getElementById('carbs').textContent = Number(data.macros.carbs_g || 0).toFixed(1);
    document.getElementById('fat').textContent = Number(data.macros.fat_g || 0).toFixed(1);
    document.getElementById('score').textContent = data.score;
    document.getElementById('streak').textContent = data.streak;
    document.getElementById('suggestion').textContent = data.suggestion;

    resultsCard.classList.remove('hidden');
    showToast('Analysis complete.');
    await loadLeaderboard();
  } catch (err) {
    showToast(err.message, true);
  } finally {
    setLoading(false);
  }
});

async function loadLeaderboard() {
  try {
    const data = await apiRequest('/api/leaderboard');
    leaderboardEl.innerHTML = '';
    data.forEach((entry) => {
      const li = document.createElement('li');
      li.textContent = `${entry.username} — ${entry.total_score} pts (streak: ${entry.streak})`;
      leaderboardEl.appendChild(li);
    });
  } catch (err) {
    showToast(err.message, true);
  }
}

refreshLeaderboardBtn.addEventListener('click', loadLeaderboard);

updateAuthState();
loadLeaderboard();
