from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import pandas as pd
from .schemas import UserIn, Token, HabitsIn, UploadResponse, PredictResponse
from .auth import (
    create_access_token,
    get_password_hash,
    verify_password,
    decode_access_token,
    validate_password_strength,
)
from .db import get_db, User, GeneticProfile, Habits
from .ml.predict import predict_with_explain

app = FastAPI(title="Anti-Aging ML API", version="0.1.0", description="Endpoints for authentication, data upload, and aging prediction.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "service": "anti-aging-api"}

@app.post("/signup", response_model=Token, tags=["auth"], summary="Register new user")
def signup(user_in: UserIn, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == user_in.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="User exists")
    pw_error = validate_password_strength(user_in.password)
    if pw_error:
        raise HTTPException(status_code=400, detail=pw_error)
    hashed = get_password_hash(user_in.password)
    db_user = User(username=user_in.username, hashed_password=hashed)
    db.add(db_user)
    db.commit()
    token = create_access_token(subject=user_in.username)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/token", response_model=Token, tags=["auth"], summary="Obtain JWT bearer token")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(subject=form.username)
    return {"access_token": token, "token_type": "bearer"}

# Note: Minimal token dependency stub; real implementation would decode and fetch user.

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    username = decode_access_token(token)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

@app.get("/me", tags=["auth"], summary="Return current authenticated user")
def read_current_user(current_user: User = Depends(get_current_user)):
    return {"id": current_user.id, "username": current_user.username}

@app.post("/upload-genetic", response_model=UploadResponse, tags=["data"], summary="Upload genetic CSV")
def upload_genetic(file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        df = pd.read_csv(file.file)
        snp_data = df.iloc[0].to_dict() if len(df) > 0 else {}
        profile = GeneticProfile(user_id=current_user.id, snp_data=snp_data)
        db.add(profile)
        db.commit()
        return {"status": "success", "id": profile.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit-habits", response_model=UploadResponse, tags=["data"], summary="Submit lifestyle habits")
def submit_habits(habits: HabitsIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db_habits = Habits(user_id=current_user.id, **habits.dict())
    db.add(db_habits)
    db.commit()
    return {"status": "success", "id": db_habits.id}

@app.get("/predict", response_model=PredictResponse, tags=["ml"], summary="Run model prediction")
def predict(model_type: str = "rf", current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    genetic = db.query(GeneticProfile).filter(GeneticProfile.user_id == current_user.id).order_by(GeneticProfile.upload_date.desc()).first()
    habits = db.query(Habits).filter(Habits.user_id == current_user.id).order_by(Habits.entry_date.desc()).first()
    if not genetic or not habits:
        raise HTTPException(status_code=400, detail="Missing genetic or habits data")
    data_dict = {**(genetic.snp_data or {}), **{k: getattr(habits, k) for k in ['exercises_per_week','daily_calories','alcohol_doses_per_week','years_smoking','hours_of_sleep','stress_level']}}
    df = pd.DataFrame([data_dict])
    pred, explanations = predict_with_explain(df, model_type=model_type)
    return {"prediction": float(pred[0]), "model": model_type, "explanations": explanations}
