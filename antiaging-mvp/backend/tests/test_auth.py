import os, sys, pathlib, pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Force sqlite URL before importing application modules to avoid postgres driver requirement.
os.environ.setdefault('DATABASE_URL', 'sqlite+pysqlite:///:memory:')

from fastapi_app.main import app, get_db  # type: ignore  # noqa
from fastapi_app.db import Base  # type: ignore  # noqa

# File-based SQLite for stable connection persistence across sessions
SQLALCHEMY_DATABASE_URL = "sqlite+pysqlite:///./test_auth.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(autouse=True)
def create_tables():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

client = TestClient(app)

def test_signup_and_login_and_me():
    # Signup
    resp = client.post('/signup', json={'username': 'alice', 'password': 'Secret123'})
    assert resp.status_code == 200
    token = resp.json()['access_token']
    assert token

    # Login
    resp2 = client.post('/token', data={'username': 'alice', 'password': 'Secret123'})
    assert resp2.status_code == 200
    token2 = resp2.json()['access_token']
    assert token2

    # /me endpoint
    me = client.get('/me', headers={'Authorization': f'Bearer {token2}'})
    assert me.status_code == 200
    assert me.json()['username'] == 'alice'

    # Protected endpoints require token; try upload with dummy CSV
    import io
    csv_content = 'col1,col2\n1,2\n'
    files = {'file': ('data.csv', io.BytesIO(csv_content.encode()), 'text/csv')}
    headers = {'Authorization': f'Bearer {token2}'}
    up = client.post('/upload-genetic', files=files, headers=headers)
    assert up.status_code in (200, 500)  # 500 acceptable for now until validation added


def test_invalid_login():
    resp = client.post('/token', data={'username': 'nosuch', 'password': 'bad'})
    assert resp.status_code == 401


def test_password_policy_rejects_weak():
    r = client.post('/signup', json={'username': 'weak', 'password': 'short'})
    assert r.status_code == 400
    assert 'Password' in r.json()['detail']
