import os, sys, pathlib, io
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault('DATABASE_URL', 'sqlite+pysqlite:///:memory:')

from fastapi_app.main import app, get_db, SCHEMA_COLUMNS  # type: ignore  # noqa
from fastapi_app.db import Base  # type: ignore  # noqa

SQLALCHEMY_DATABASE_URL = "sqlite+pysqlite:///./test_issue4.db"
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


def auth_token():
    r = client.post('/signup', json={'username': 'bob', 'password': 'Strong123'})
    assert r.status_code == 200
    return r.json()['access_token']


def build_single_row_csv(columns, fill_value=1):
    data = {c: [fill_value] for c in columns}
    # Some known non-numeric fields from schema need string placeholders
    for cat_col in [c for c in columns if c.startswith(('APOE_', 'FOXO3_', 'SIRT1_', 'TP53_', 'CDKN2A_', 'TERT_', 'TERC_', 'IGF1_', 'KLOTHO_'))]:
        data[cat_col] = ["AA"]
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def test_upload_genetic_success_and_retrieval():
    token = auth_token()
    # Build CSV with exact schema columns
    cols = [c for c in SCHEMA_COLUMNS if c != 'user_id'] if 'user_id' in SCHEMA_COLUMNS else SCHEMA_COLUMNS
    csv_text = build_single_row_csv(cols, fill_value=1)
    files = {'file': ('genetic.csv', io.BytesIO(csv_text.encode()), 'text/csv')}
    headers = {'Authorization': f'Bearer {token}'}
    resp = client.post('/upload-genetic', files=files, headers=headers)
    assert resp.status_code == 200, resp.text
    gid = resp.json()['id']
    assert isinstance(gid, int)

    # Retrieve latest genetic profile
    g = client.get('/genetic-profile', headers=headers)
    assert g.status_code == 200
    body = g.json()
    assert body['id'] == gid
    assert isinstance(body['snp_data'], dict)

    # Submit and retrieve habits
    h_post = client.post('/submit-habits', json={
        'exercises_per_week': 3,
        'daily_calories': 2000,
        'alcohol_doses_per_week': 2.0,
        'years_smoking': 0,
        'hours_of_sleep': 7.5,
        'stress_level': 5
    }, headers=headers)
    assert h_post.status_code == 200

    h_get = client.get('/habits', headers=headers)
    assert h_get.status_code == 200
    h = h_get.json()
    assert h['exercises_per_week'] == 3


def test_upload_missing_column_rejected():
    token = auth_token()
    # Remove one required column
    if len(SCHEMA_COLUMNS) < 2:
        pytest.skip("Schema insufficient for test")
    subset = SCHEMA_COLUMNS[:-1]
    csv_text = build_single_row_csv(subset, fill_value=1)
    files = {'file': ('genetic.csv', io.BytesIO(csv_text.encode()), 'text/csv')}
    headers = {'Authorization': f'Bearer {token}'}
    resp = client.post('/upload-genetic', files=files, headers=headers)
    assert resp.status_code == 400
    assert 'Missing required columns' in resp.text


def test_upload_rejects_extras():
    token = auth_token()
    cols = SCHEMA_COLUMNS[:]
    extra_cols = cols + ['unexpected_col']
    csv_text = build_single_row_csv(extra_cols, fill_value=1)
    files = {'file': ('genetic.csv', io.BytesIO(csv_text.encode()), 'text/csv')}
    headers = {'Authorization': f'Bearer {token}'}
    resp = client.post('/upload-genetic', files=files, headers=headers)
    assert resp.status_code == 400
    assert 'Unexpected columns' in resp.text
