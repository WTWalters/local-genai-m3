"""
Test endpoints for RBAC system without full authentication
Temporary endpoints for POC demonstration
"""

from fastapi import APIRouter, Depends, Request
from typing import Dict, Any

from middleware.rbac import Permission, MedicalRole
from core.dependencies import CurrentUser

router = APIRouter()

def create_mock_user(role: str, permissions: set) -> CurrentUser:
    """Create a mock user for testing RBAC."""
    return CurrentUser(
        user_id="test-user-123",
        username="test_user",
        role=role,
        can_access_phi=role in ["attending_physician", "resident", "nurse"],
        session_id="test-session-123",
        permissions=permissions
    )

def mock_admin_user() -> CurrentUser:
    """Mock admin user for testing."""
    return create_mock_user("admin", MedicalRole.ADMIN)

def mock_physician_user() -> CurrentUser:
    """Mock attending physician user for testing."""
    return create_mock_user("attending_physician", MedicalRole.ATTENDING_PHYSICIAN)

def mock_resident_user() -> CurrentUser:
    """Mock resident user for testing."""
    return create_mock_user("resident", MedicalRole.RESIDENT)

def mock_nurse_user() -> CurrentUser:
    """Mock nurse user for testing."""
    return create_mock_user("nurse", MedicalRole.NURSE)

def mock_readonly_user() -> CurrentUser:
    """Mock read-only user for testing."""
    return create_mock_user("read_only", MedicalRole.READ_ONLY)

@router.get("/admin-only")
async def admin_only_endpoint(current_user: CurrentUser = Depends(mock_admin_user)) -> Dict[str, Any]:
    """Test endpoint that requires admin permissions."""
    return {
        "message": "Access granted to admin endpoint",
        "user_role": current_user.role,
        "permissions": list(current_user.permissions)[:10],  # Show first 10 permissions
        "is_admin": current_user.is_admin(),
        "can_access_phi": current_user.can_access_phi
    }

@router.get("/physician-only")
async def physician_only_endpoint(current_user: CurrentUser = Depends(mock_physician_user)) -> Dict[str, Any]:
    """Test endpoint for attending physicians."""
    return {
        "message": "Access granted to physician endpoint",
        "user_role": current_user.role,
        "permissions": list(current_user.permissions)[:10],
        "is_physician": current_user.is_physician(),
        "can_access_phi": current_user.can_access_phi
    }

@router.get("/phi-read")
async def phi_read_endpoint(current_user: CurrentUser = Depends(mock_physician_user)) -> Dict[str, Any]:
    """Test endpoint that requires PHI read access."""
    if not current_user.has_permission(Permission.PHI_READ):
        return {"error": "Insufficient permissions - PHI_READ required"}
    
    return {
        "message": "PHI data access granted",
        "user_role": current_user.role,
        "can_access_patient_data": current_user.can_access_patient_data(),
        "sample_phi_data": {
            "patient_id": "PHI-PROTECTED-DATA",
            "note": "This would contain protected health information"
        }
    }

@router.get("/test-permissions/{test_role}")
async def test_role_permissions(test_role: str) -> Dict[str, Any]:
    """Test endpoint to show permissions for different roles."""
    from middleware.rbac import RBACMiddleware
    
    role_permissions = {
        "admin": MedicalRole.ADMIN,
        "attending_physician": MedicalRole.ATTENDING_PHYSICIAN,
        "resident": MedicalRole.RESIDENT,
        "nurse": MedicalRole.NURSE,
        "read_only": MedicalRole.READ_ONLY
    }
    
    if test_role not in role_permissions:
        return {"error": f"Unknown role: {test_role}"}
    
    permissions = role_permissions[test_role]
    
    return {
        "role": test_role,
        "permissions": sorted(list(permissions)),
        "permission_count": len(permissions),
        "can_access_phi": test_role in ["attending_physician", "resident", "nurse"],
        "key_permissions": {
            "PHI_READ": Permission.PHI_READ in permissions,
            "PHI_WRITE": Permission.PHI_WRITE in permissions,
            "USER_DELETE": Permission.USER_DELETE in permissions,
            "ADMIN_WRITE": Permission.ADMIN_WRITE in permissions
        }
    }

@router.get("/rbac-demo")
async def rbac_demo() -> Dict[str, Any]:
    """Comprehensive RBAC demonstration endpoint."""
    return {
        "message": "RBAC System Demonstration",
        "available_roles": [
            "admin",
            "attending_physician", 
            "resident",
            "nurse",
            "read_only"
        ],
        "test_endpoints": [
            "/api/v1/test-rbac/admin-only",
            "/api/v1/test-rbac/physician-only",
            "/api/v1/test-rbac/phi-read",
            "/api/v1/test-rbac/test-permissions/{role}"
        ],
        "permission_categories": [
            "PHI Access (PHI_READ, PHI_WRITE, PHI_EXPORT)",
            "User Management (USER_READ, USER_WRITE, USER_DELETE)",
            "System Administration (SYSTEM_ADMIN, AUDIT_READ)",
            "Query Operations (QUERY_BASIC, QUERY_ADVANCED)"
        ],
        "security_notes": [
            "This is a test endpoint for POC demonstration",
            "Full authentication and session management will be integrated later",
            "All medical data access will require proper MFA in production"
        ]
    }

@router.get("/test-patient-api")
async def test_patient_api() -> Dict[str, Any]:
    """Test patient API endpoints - bypasses authentication for development testing"""
    from database import db_manager
    from database.patient_models import Patient
    from sqlalchemy import select
    
    async with db_manager.get_session() as db:
        # Get first few patients for testing
        result = await db.execute(
            select(Patient).limit(3)
        )
        patients = result.scalars().all()
        
        patient_list = []
        for patient in patients:
            patient_list.append({
                "patient_id": str(patient.patient_id),
                "medical_record_number": patient.medical_record_number,
                "first_name": patient.first_name,
                "last_name": patient.last_name,
                "date_of_birth": patient.date_of_birth.isoformat(),
                "gender": patient.gender,
                "status": patient.status
            })
    
    return {
        "message": "Patient API test endpoint",
        "patients_found": len(patient_list),
        "sample_patients": patient_list,
        "endpoints_available": [
            "GET /api/v1/patients/ - List patients with pagination",
            "GET /api/v1/patients/{patient_id} - Get patient details",
            "GET /api/v1/patients/{patient_id}/comprehensive - Get complete patient record",
            "PATCH /api/v1/patients/{patient_id} - Update patient information"
        ],
        "note": "All patient endpoints require PHI access permissions in production"
    }

@router.get("/test-patient-count")
async def test_patient_count() -> Dict[str, Any]:
    """Test patient database count and verify data exists"""
    from database import db_manager
    from database.patient_models import Patient, Appointment, Procedure
    from sqlalchemy import select, func
    
    async with db_manager.get_session() as db:
        # Get patient count
        patient_count_result = await db.execute(
            select(func.count(Patient.patient_id))
        )
        patient_count = patient_count_result.scalar()
        
        # Get appointment count
        appointment_count_result = await db.execute(
            select(func.count(Appointment.appointment_id))
        )
        appointment_count = appointment_count_result.scalar()
        
        # Get procedure count
        procedure_count_result = await db.execute(
            select(func.count(Procedure.procedure_id))
        )
        procedure_count = procedure_count_result.scalar()
        
        # Get sample patient data
        sample_patients_result = await db.execute(
            select(Patient).limit(2)
        )
        sample_patients = sample_patients_result.scalars().all()
        
        patient_samples = []
        for patient in sample_patients:
            patient_samples.append({
                "patient_id": str(patient.patient_id),
                "mrn": patient.medical_record_number,
                "name": f"{patient.first_name} {patient.last_name}",
                "dob": patient.date_of_birth.isoformat(),
                "status": patient.status
            })
    
    return {
        "success": True,
        "message": "Patient database connectivity test successful",
        "counts": {
            "patients": patient_count,
            "appointments": appointment_count,
            "procedures": procedure_count
        },
        "sample_patients": patient_samples,
        "api_status": "Patient API endpoints are ready and connected to database"
    }

@router.post("/test-patient-search")
async def test_patient_search() -> Dict[str, Any]:
    """Test advanced patient search functionality"""
    from api.patients import search_patients
    from schemas.patients import PatientSearch
    from middleware.rbac import MedicalRole
    from core.dependencies import CurrentUser
    
    # Mock user with PHI access for testing
    mock_user = CurrentUser(
        user_id="test-user-id",
        username="test_physician",
        role=MedicalRole.ATTENDING_PHYSICIAN,
        can_access_phi=True,
        session_id="test-session",
        permissions={"PHI_READ", "PHI_WRITE"}
    )
    
    # Test different search scenarios
    test_results = {}
    
    try:
        # Test 1: Basic name search
        search_1 = PatientSearch(search_term="Margaret")
        result_1 = await search_patients(
            search_params=search_1,
            page=1,
            per_page=10,
            sort_by="last_name",
            sort_order="asc",
            current_user=mock_user
        )
        test_results["name_search"] = {
            "search_term": "Margaret",
            "found": result_1.total,
            "patients": [f"{p.first_name} {p.last_name}" for p in result_1.patients]
        }
        
        # Test 2: Gender filter
        search_2 = PatientSearch(gender="female")
        result_2 = await search_patients(
            search_params=search_2,
            page=1,
            per_page=10,
            sort_by="last_name",
            sort_order="asc",
            current_user=mock_user
        )
        test_results["gender_filter"] = {
            "gender": "female",
            "found": result_2.total,
            "patients": [f"{p.first_name} {p.last_name}" for p in result_2.patients]
        }
        
        # Test 3: Age range search
        search_3 = PatientSearch(age_min=60, age_max=80)
        result_3 = await search_patients(
            search_params=search_3,
            page=1,
            per_page=10,
            sort_by="last_name",
            sort_order="asc",
            current_user=mock_user
        )
        test_results["age_range"] = {
            "age_range": "60-80",
            "found": result_3.total,
            "patients": [f"{p.first_name} {p.last_name} (age {p.age})" for p in result_3.patients]
        }
        
        # Test 4: City search
        search_4 = PatientSearch(city="Boston")
        result_4 = await search_patients(
            search_params=search_4,
            page=1,
            per_page=10,
            sort_by="last_name",
            sort_order="asc",
            current_user=mock_user
        )
        test_results["city_search"] = {
            "city": "Boston",
            "found": result_4.total,
            "patients": [f"{p.first_name} {p.last_name}" for p in result_4.patients]
        }
        
        # Test 5: Combined search
        search_5 = PatientSearch(gender="male", age_min=50)
        result_5 = await search_patients(
            search_params=search_5,
            page=1,
            per_page=10,
            sort_by="last_name",
            sort_order="asc",
            current_user=mock_user
        )
        test_results["combined_search"] = {
            "criteria": "male, age 50+",
            "found": result_5.total,
            "patients": [f"{p.first_name} {p.last_name} (age {p.age})" for p in result_5.patients]
        }
        
        return {
            "success": True,
            "message": "Advanced patient search tests completed successfully",
            "test_results": test_results,
            "search_capabilities": [
                "Basic text search (name, MRN, phone)",
                "Demographic filtering (gender, age range, DOB)",
                "Contact information search (phone, email, address)",
                "Medical information search (allergies, primary care physician)",
                "Treatment status filtering (appointments, procedures)",
                "Date range filtering (created dates, visit dates)",
                "Flexible sorting and pagination"
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Patient search test failed: {str(e)}",
            "error_type": type(e).__name__,
            "completed_tests": list(test_results.keys())
        }

@router.post("/test-medical-search")
async def test_medical_search() -> Dict[str, Any]:
    """Test medical condition and procedure search functionality"""
    from api.patients import search_patients
    from schemas.patients import PatientSearch
    from middleware.rbac import MedicalRole
    from core.dependencies import CurrentUser
    
    # Mock user with PHI access for testing
    mock_user = CurrentUser(
        user_id="test-user-id",
        username="test_physician",
        role=MedicalRole.ATTENDING_PHYSICIAN,
        can_access_phi=True,
        session_id="test-session",
        permissions={"PHI_READ", "PHI_WRITE"}
    )
    
    # Test medical condition and procedure searches
    test_results = {}
    
    try:
        # Test 1: Search for knee procedures
        search_1 = PatientSearch(procedure_name="knee")
        result_1 = await search_patients(
            search_params=search_1,
            page=1,
            per_page=10,
            sort_by="last_name",
            sort_order="asc",
            current_user=mock_user
        )
        test_results["knee_procedures"] = {
            "search_term": "knee",
            "found": result_1.total,
            "patients": [f"{p.first_name} {p.last_name}" for p in result_1.patients]
        }
        
        # Test 2: Search for completed procedures
        search_2 = PatientSearch(procedure_status="completed")
        result_2 = await search_patients(
            search_params=search_2,
            page=1,
            per_page=10,
            sort_by="last_name",
            sort_order="asc",
            current_user=mock_user
        )
        test_results["completed_procedures"] = {
            "status": "completed",
            "found": result_2.total,
            "patients": [f"{p.first_name} {p.last_name}" for p in result_2.patients]
        }
        
        # Test 3: Search for shoulder procedures
        search_3 = PatientSearch(procedure_name="shoulder")
        result_3 = await search_patients(
            search_params=search_3,
            page=1,
            per_page=10,
            sort_by="last_name",
            sort_order="asc",
            current_user=mock_user
        )
        test_results["shoulder_procedures"] = {
            "search_term": "shoulder",
            "found": result_3.total,
            "patients": [f"{p.first_name} {p.last_name}" for p in result_3.patients]
        }
        
        # Test 4: Search by chief complaint
        search_4 = PatientSearch(chief_complaint="pain")
        result_4 = await search_patients(
            search_params=search_4,
            page=1,
            per_page=10,
            sort_by="last_name",
            sort_order="asc",
            current_user=mock_user
        )
        test_results["pain_complaints"] = {
            "chief_complaint": "pain",
            "found": result_4.total,
            "patients": [f"{p.first_name} {p.last_name}" for p in result_4.patients]
        }
        
        # Test 5: Search for fractures
        search_5 = PatientSearch(diagnosis_search="fracture")
        result_5 = await search_patients(
            search_params=search_5,
            page=1,
            per_page=10,
            sort_by="last_name",
            sort_order="asc",
            current_user=mock_user
        )
        test_results["fracture_diagnosis"] = {
            "diagnosis": "fracture",
            "found": result_5.total,
            "patients": [f"{p.first_name} {p.last_name}" for p in result_5.patients]
        }
        
        # Test 6: Search for arthritis in medical history
        search_6 = PatientSearch(medical_condition="arthritis")
        result_6 = await search_patients(
            search_params=search_6,
            page=1,
            per_page=10,
            sort_by="last_name",
            sort_order="asc",
            current_user=mock_user
        )
        test_results["arthritis_history"] = {
            "condition": "arthritis",
            "found": result_6.total,
            "patients": [f"{p.first_name} {p.last_name}" for p in result_6.patients]
        }
        
        return {
            "success": True,
            "message": "Medical condition and procedure search tests completed successfully",
            "test_results": test_results,
            "search_capabilities": [
                "Procedure name search (knee, shoulder, ACL, etc.)",
                "Procedure status filtering (completed, scheduled, in_progress)",
                "CPT procedure code search",
                "ICD-10 diagnosis code search",
                "Chief complaint search across appointments and notes",
                "Diagnosis search in pre/post-op notes and assessments",
                "Medical history search (medical, family, social history)",
                "Medical note content search (assessment and plan)",
                "Procedure date range filtering"
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Medical search test failed: {str(e)}",
            "error_type": type(e).__name__,
            "completed_tests": list(test_results.keys())
        }