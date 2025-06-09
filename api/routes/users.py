from fastapi import APIRouter, Depends, HTTPException, Query, File, UploadFile
from sqlalchemy.orm import Session
from typing import List
from database.session import get_db
from data.models.user import User
from data.models.student_profile import StudentProfile
from api.auth.supabase import check_admin, get_current_user, get_supabase_client, get_supabase_user, check_admin_or_vp, check_recruiter_or_admin
import csv
from io import StringIO
from pydantic import BaseModel

router = APIRouter(prefix="/users", tags=["users"])

SUPER_ADMIN_ROLE = "super_admin"

# Helper: get users visible to current user

def get_visible_users(db, current_user):
    if current_user.role == SUPER_ADMIN_ROLE:
        return db.query(User).all()
    elif current_user.role == "admin":
        return db.query(User).filter(User.role != SUPER_ADMIN_ROLE).all()
    else:
        return db.query(User).filter(User.role == current_user.role).all()

@router.get("/me", response_model=dict)
def get_me(current_user=Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=403, detail="User is deactivated")
    return current_user.to_dict()

@router.get("", response_model=List[dict])
def list_users(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    users = get_visible_users(db, current_user)
    return [u.to_dict() for u in users]

@router.get("/{user_id}", response_model=dict)
def get_user(user_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    user_obj = db.query(User).filter(User.id == user_id).first()
    if not user_obj:
        raise HTTPException(status_code=404, detail="User not found")
    # Only allow if visible to current user
    visible_ids = {u.id for u in get_visible_users(db, current_user)}
    if user_id not in visible_ids:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return user_obj.to_dict()

@router.post("", response_model=dict)
def create_user(
    username: str = Query(...),
    email: str = Query(...),
    name: str = Query(None),
    role: str = Query("recruiter"),
    db: Session = Depends(get_db),
    current_user=Depends(check_admin)
):
    # Create user in Supabase
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=500, detail="Authentication service unavailable")
    resp = supabase.auth.admin.create_user({"email": email, "password": "TempPass123!", "email_confirm": False})
    if not resp.user:
        raise HTTPException(status_code=400, detail="Failed to create user in Supabase")
    supabase_id = resp.user.id
    # Save in local DB
    new_user = User(
        supabase_id=supabase_id,
        username=username,
        email=email,
        role=role,
        is_active=True
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    # Optionally: trigger password reset email
    try:
        if supabase:
            supabase.auth.admin.invite_user_by_email(email)
    except Exception:
        pass
    return new_user.to_dict()

@router.put("/{user_id}", response_model=dict)
def update_user(
    user_id: int,
    username: str = Query(None),
    email: str = Query(None),
    role: str = Query(None),
    is_active: bool = Query(None),
    db: Session = Depends(get_db),
    current_user=Depends(check_admin)
):
    user_obj = db.query(User).filter(User.id == user_id).first()
    if not user_obj:
        raise HTTPException(status_code=404, detail="User not found")
    # Only allow if current user outranks target
    if current_user.role != SUPER_ADMIN_ROLE and user_obj.role == SUPER_ADMIN_ROLE:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    if username is not None:
        user_obj.username = username
    if email is not None:
        user_obj.email = email
    if role is not None and (current_user.role == SUPER_ADMIN_ROLE or role != SUPER_ADMIN_ROLE):
        user_obj.role = role
    if is_active is not None:
        user_obj.is_active = is_active
    db.commit()
    return user_obj.to_dict()

@router.delete("/{user_id}", response_model=dict)
def delete_user(user_id: int, db: Session = Depends(get_db), current_user=Depends(check_admin)):
    user_obj = db.query(User).filter(User.id == user_id).first()
    if not user_obj:
        raise HTTPException(status_code=404, detail="User not found")
    if current_user.role != SUPER_ADMIN_ROLE and user_obj.role == SUPER_ADMIN_ROLE:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    db.delete(user_obj)
    db.commit()
    return {"success": True, "user_id": user_id}

@router.post("/{user_id}/activate", response_model=dict)
def activate_user(user_id: int, db: Session = Depends(get_db), current_user=Depends(check_admin)):
    user_obj = db.query(User).filter(User.id == user_id).first()
    if not user_obj:
        raise HTTPException(status_code=404, detail="User not found")
    user_obj.is_active = True
    db.commit()
    return user_obj.to_dict()

@router.post("/{user_id}/deactivate", response_model=dict)
def deactivate_user(user_id: int, db: Session = Depends(get_db), current_user=Depends(check_admin)):
    user_obj = db.query(User).filter(User.id == user_id).first()
    if not user_obj:
        raise HTTPException(status_code=404, detail="User not found")
    user_obj.is_active = False
    db.commit()
    return user_obj.to_dict()

@router.post("/import", response_model=dict)
def import_users(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(check_admin)
):
    content = file.file.read().decode("utf-8")
    reader = csv.DictReader(StringIO(content))
    created = 0
    errors = []
    for row in reader:
        try:
            email = row["email"]
            username = row.get("username") or email.split("@")[0]
            role = row.get("role", "recruiter")
            # Create in Supabase
            supabase = get_supabase_client()
            if not supabase:
                errors.append(f"Authentication service unavailable for {email}")
                continue
            resp = supabase.auth.admin.create_user({"email": email, "password": "TempPass123!", "email_confirm": False})
            if not resp.user:
                errors.append(f"Supabase error for {email}")
                continue
            supabase_id = resp.user.id
            # Save in local DB
            new_user = User(
                supabase_id=supabase_id,
                username=username,
                email=email,
                role=role,
                is_active=True
            )
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            # Trigger password reset email
            try:
                if supabase:
                    supabase.auth.admin.invite_user_by_email(email)
            except Exception:
                pass
            created += 1
        except Exception as e:
            errors.append(str(e))
    return {"created": created, "errors": errors}

@router.post("/register", response_model=dict)
def register_user(
    db: Session = Depends(get_db),
    supabase_user=Depends(get_supabase_user),
    body: dict = None
):
    # Check if user already exists
    user_obj = db.query(User).filter(User.supabase_id == supabase_user['supabase_id']).first()
    if user_obj:
        raise HTTPException(status_code=400, detail="User already registered")
    # Use names from request body if provided, else from JWT
    first_name = (body or {}).get('first_name') or supabase_user.get('first_name')
    last_name = (body or {}).get('last_name') or supabase_user.get('last_name')
    # Create user with recruiter role
    new_user = User(
        supabase_id=supabase_user['supabase_id'],
        username=supabase_user['username'] or supabase_user['email'].split("@")[0],
        email=supabase_user['email'],
        first_name=first_name,
        last_name=last_name,
        role="recruiter",
        is_active=True
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user.to_dict()

class SignupRequest(BaseModel):
    email: str
    password: str
    first_name: str = None
    last_name: str = None

@router.post("/signup", response_model=dict)
async def signup(
    data: SignupRequest,
    db: Session = Depends(get_db)
):
    email = data.email
    password = data.password
    first_name = data.first_name
    last_name = data.last_name
    try:
        # Create user in Supabase
        supabase = get_supabase_client()
        if not supabase:
            raise HTTPException(status_code=500, detail="Authentication service unavailable")
        auth_response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "display_name": f"{first_name or ''} {last_name or ''}".strip(),
                    "first_name": first_name,
                    "last_name": last_name
                }
            }
        })
        if not auth_response.user:
            raise HTTPException(status_code=400, detail="Error creating user in Supabase")
        # Create user in our database
        user = User(
            supabase_id=auth_response.user.id,
            email=email,
            username=email.split("@")[0],
            first_name=first_name,
            last_name=last_name,
            role="admissions_assistant",
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return {
            "message": "User created successfully",
            "user": user.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating user: {e}")

# Recruiter-specific endpoints
@router.get("/recruiters", response_model=List[dict])
def list_recruiters(
    db: Session = Depends(get_db),
    user=Depends(check_admin_or_vp)
):
    """
    List all recruiters. Only accessible by admin/VP.
    """
    recruiters = db.query(User).filter(User.role == "recruiter").all()
    return [user.to_dict() for user in recruiters]

@router.get("/recruiters/{recruiter_id}/students")
def get_recruiter_students(
    recruiter_id: str,
    db: Session = Depends(get_db),
    user=Depends(check_recruiter_or_admin)
):
    """
    List students assigned to a recruiter (by CRM user ID). Accessible by admin/VP or the recruiter themselves.
    """
    students = db.query(StudentProfile).filter(StudentProfile.enrollment_agent_id == recruiter_id).all()
    return [s.to_dict() for s in students]

@router.post("/recruiters/import")
def import_recruiters(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user=Depends(check_admin_or_vp)
):
    """
    Import recruiter accounts from an Excel/CSV file. Only accessible by admin/VP.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    content = file.file.read().decode('utf-8')
    reader = csv.DictReader(StringIO(content))
    created = 0
    for row in reader:
        # Expecting columns: first_name, last_name, email, crm_user_id, role
        if not all(k in row for k in ("first_name", "last_name", "email", "crm_user_id", "role")):
            continue
        # Check if user already exists
        existing = db.query(User).filter_by(crm_user_id=row["crm_user_id"]).first()
        if existing:
            continue
        user = User(
            first_name=row["first_name"],
            last_name=row["last_name"],
            email=row["email"],
            crm_user_id=row["crm_user_id"],
            role=row["role"],
            is_active=True
        )
        db.add(user)
        created += 1
    db.commit()
    return {"created": created} 