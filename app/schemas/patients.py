"""
Pydantic schemas for patient data API responses
HIPAA-compliant data models with proper field validation
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum

# Response Models for Patient Data
class PatientSummary(BaseModel):
    """Patient summary for list views - limited PHI"""
    patient_id: str
    medical_record_number: str
    first_name: str = Field(..., description="Patient first name")
    last_name: str = Field(..., description="Patient last name")
    date_of_birth: date
    gender: str
    status: str
    primary_care_physician: Optional[str] = None
    created_at: datetime
    
    # Calculated fields
    age: Optional[int] = None
    
    @validator('age', always=True)
    def calculate_age(cls, v, values):
        if 'date_of_birth' in values:
            today = date.today()
            dob = values['date_of_birth']
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return None

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "patient_id": "550e8400-e29b-41d4-a716-446655440000",
                "medical_record_number": "MRN001234",
                "first_name": "Margaret",
                "last_name": "Thompson",
                "date_of_birth": "1955-03-15",
                "gender": "female",
                "status": "active",
                "age": 68
            }
        }

class PatientDetail(PatientSummary):
    """Complete patient information - requires PHI access"""
    middle_name: Optional[str] = None
    phone_primary: Optional[str] = None
    phone_secondary: Optional[str] = None
    email: Optional[str] = None
    
    # Address information
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    
    # Medical information
    allergies: Optional[str] = None
    current_medications: Optional[List[Dict[str, Any]]] = None
    medical_history: Optional[str] = None
    family_history: Optional[str] = None
    social_history: Optional[str] = None
    
    # Emergency contact
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    emergency_contact_relationship: Optional[str] = None
    
    updated_at: datetime

class InsuranceInfo(BaseModel):
    """Patient insurance information"""
    insurance_id: str
    insurance_type: str
    insurance_company: str
    policy_number: str
    group_number: Optional[str] = None
    subscriber_name: str
    is_primary: bool
    effective_date: Optional[date] = None
    expiration_date: Optional[date] = None
    copay_amount: Optional[Decimal] = None
    deductible_amount: Optional[Decimal] = None
    active: bool

    class Config:
        from_attributes = True

class AppointmentInfo(BaseModel):
    """Appointment information"""
    appointment_id: str
    appointment_type: str
    appointment_date: date
    appointment_time: time
    duration_minutes: int
    status: str
    chief_complaint: Optional[str] = None
    notes: Optional[str] = None
    provider_name: Optional[str] = None
    
    class Config:
        from_attributes = True

class ProcedureInfo(BaseModel):
    """Procedure information"""
    procedure_id: str
    procedure_name: str
    procedure_code: Optional[str] = None
    icd10_diagnosis_codes: Optional[List[str]] = None
    scheduled_date: Optional[datetime] = None
    completed_date: Optional[datetime] = None
    status: str
    pre_op_diagnosis: Optional[str] = None
    post_op_diagnosis: Optional[str] = None
    provider_name: Optional[str] = None
    
    class Config:
        from_attributes = True

class MedicalNoteInfo(BaseModel):
    """Medical note information"""
    note_id: str
    note_type: Optional[str] = None
    note_date: datetime
    chief_complaint: Optional[str] = None
    assessment: Optional[str] = None
    plan: Optional[str] = None
    provider_name: Optional[str] = None
    
    class Config:
        from_attributes = True

class ImagingStudyInfo(BaseModel):
    """Imaging study information"""
    study_id: str
    study_type: str
    body_part: Optional[str] = None
    study_date: Optional[datetime] = None
    clinical_indication: Optional[str] = None
    findings: Optional[str] = None
    impression: Optional[str] = None
    provider_name: Optional[str] = None
    
    class Config:
        from_attributes = True

class BillingInfo(BaseModel):
    """Billing record information"""
    billing_id: str
    service_date: date
    cpt_code: str
    description: str
    charge_amount: Decimal
    allowed_amount: Optional[Decimal] = None
    paid_amount: Optional[Decimal] = None
    patient_responsibility: Optional[Decimal] = None
    status: str
    
    class Config:
        from_attributes = True

class PhysicalTherapyInfo(BaseModel):
    """Physical therapy session information"""
    pt_session_id: str
    session_date: date
    session_number: Optional[int] = None
    total_sessions_planned: Optional[int] = None
    pain_level_before: Optional[int] = None
    pain_level_after: Optional[int] = None
    session_notes: Optional[str] = None
    patient_compliance: Optional[str] = None
    
    class Config:
        from_attributes = True

class PatientComprehensive(PatientDetail):
    """Complete patient record with all related data"""
    insurance_plans: List[InsuranceInfo] = []
    recent_appointments: List[AppointmentInfo] = []
    procedures: List[ProcedureInfo] = []
    recent_notes: List[MedicalNoteInfo] = []
    imaging_studies: List[ImagingStudyInfo] = []
    billing_records: List[BillingInfo] = []
    pt_sessions: List[PhysicalTherapyInfo] = []

# Request Models
class PatientSearch(BaseModel):
    """Advanced patient search parameters"""
    # Basic search
    search_term: Optional[str] = Field(None, description="Search in name, MRN, or phone")
    medical_record_number: Optional[str] = Field(None, description="Exact MRN search")
    
    # Demographics
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    gender: Optional[str] = None
    status: Optional[str] = None
    age_min: Optional[int] = Field(None, ge=0, le=150)
    age_max: Optional[int] = Field(None, ge=0, le=150)
    date_of_birth: Optional[date] = None
    
    # Contact information
    phone: Optional[str] = Field(None, description="Search in primary or secondary phone")
    email: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    
    # Medical information
    has_allergies: Optional[bool] = None
    allergy_search: Optional[str] = Field(None, description="Search in allergies text")
    primary_care_physician: Optional[str] = None
    
    # Treatment status
    has_active_treatment: Optional[bool] = None
    has_upcoming_appointments: Optional[bool] = None
    has_recent_procedures: Optional[bool] = None
    
    # Date ranges
    created_after: Optional[date] = None
    created_before: Optional[date] = None
    last_visit_after: Optional[date] = None
    last_visit_before: Optional[date] = None
    
    # Medical procedures and conditions
    procedure_name: Optional[str] = Field(None, description="Search for patients with specific procedure")
    procedure_code: Optional[str] = Field(None, description="Search by CPT procedure code")
    icd10_code: Optional[str] = Field(None, description="Search by ICD-10 diagnosis code")
    procedure_status: Optional[str] = Field(None, description="Filter by procedure status")
    procedure_date_after: Optional[date] = None
    procedure_date_before: Optional[date] = None
    
    # Medical conditions and notes
    medical_condition: Optional[str] = Field(None, description="Search in medical history or assessment")
    chief_complaint: Optional[str] = Field(None, description="Search in chief complaints")
    diagnosis_search: Optional[str] = Field(None, description="Search in diagnosis fields")
    note_content: Optional[str] = Field(None, description="Search in medical notes content")
    
class PatientListResponse(BaseModel):
    """Paginated patient list response"""
    patients: List[PatientSummary]
    total: int
    page: int
    per_page: int
    total_pages: int

class PatientUpdate(BaseModel):
    """Patient update model - limited fields"""
    phone_primary: Optional[str] = None
    phone_secondary: Optional[str] = None
    email: Optional[str] = None
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    emergency_contact_relationship: Optional[str] = None
    allergies: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "phone_primary": "555-0123",
                "email": "patient@email.com",
                "allergies": "Penicillin (rash)"
            }
        }

# Summary and Analytics Models
class PatientStatistics(BaseModel):
    """Patient statistics for dashboard"""
    total_patients: int
    active_patients: int
    patients_by_gender: Dict[str, int]
    patients_by_age_group: Dict[str, int]
    recent_registrations: int
    
class TreatmentSummary(BaseModel):
    """Treatment summary for a patient"""
    patient_id: str
    total_procedures: int
    total_appointments: int
    total_pt_sessions: int
    last_visit_date: Optional[date] = None
    next_appointment_date: Optional[date] = None
    active_treatments: List[str] = []
    
class BillingSummary(BaseModel):
    """Billing summary for a patient"""
    patient_id: str
    total_charges: Decimal
    total_paid: Decimal
    outstanding_balance: Decimal
    insurance_paid: Decimal
    last_payment_date: Optional[date] = None