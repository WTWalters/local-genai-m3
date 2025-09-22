"""
Patient data models for Orthopedic EMR system
Comprehensive patient lifecycle management from diagnosis to follow-up
"""

from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, Numeric, ForeignKey, Date, Time
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from enum import Enum
import uuid

from .connection import Base

# Enums for standardized medical data
class PatientStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DECEASED = "deceased"
    TRANSFERRED = "transferred"

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"

class InsuranceType(str, Enum):
    PRIVATE = "private"
    MEDICARE = "medicare"
    MEDICAID = "medicaid"
    SELF_PAY = "self_pay"
    WORKERS_COMP = "workers_comp"
    AUTO_INSURANCE = "auto_insurance"

class AppointmentStatus(str, Enum):
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"

class AppointmentType(str, Enum):
    CONSULTATION = "consultation"
    FOLLOW_UP = "follow_up"
    SURGERY = "surgery"
    PHYSICAL_THERAPY = "physical_therapy"
    IMAGING = "imaging"
    PROCEDURE = "procedure"

class ProcedureStatus(str, Enum):
    PLANNED = "planned"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    POSTPONED = "postponed"

class BillingStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    DENIED = "denied"
    PAID = "paid"
    WRITE_OFF = "write_off"

class ImagingType(str, Enum):
    XRAY = "xray"
    MRI = "mri"
    CT = "ct"
    ULTRASOUND = "ultrasound"
    BONE_SCAN = "bone_scan"

# Core Patient Models
class Patient(Base):
    __tablename__ = "patients"
    
    patient_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    medical_record_number = Column(String(20), unique=True, nullable=False)
    
    # Demographics
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    middle_name = Column(String(100))
    date_of_birth = Column(Date, nullable=False)
    gender = Column(String(20), nullable=False)
    
    # Contact Information
    phone_primary = Column(String(20))
    phone_secondary = Column(String(20))
    email = Column(String(255))
    
    # Address
    address_line1 = Column(String(255))
    address_line2 = Column(String(255))
    city = Column(String(100))
    state = Column(String(50))
    zip_code = Column(String(10))
    
    # Medical Information
    primary_care_physician = Column(String(255))
    allergies = Column(Text)
    current_medications = Column(JSONB)
    medical_history = Column(Text)
    family_history = Column(Text)
    social_history = Column(Text)
    
    # Emergency Contact
    emergency_contact_name = Column(String(255))
    emergency_contact_phone = Column(String(20))
    emergency_contact_relationship = Column(String(100))
    
    # Vector embedding for patient summary semantic search
    summary_embedding = Column(Vector(384))  # 384 dimensions for all-MiniLM-L6-v2
    
    # System Fields
    status = Column(String(20), default=PatientStatus.ACTIVE)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    
    # Relationships
    insurance_plans = relationship("PatientInsurance", back_populates="patient")
    appointments = relationship("Appointment", back_populates="patient")
    procedures = relationship("Procedure", back_populates="patient")
    billing_records = relationship("BillingRecord", back_populates="patient")
    medical_notes = relationship("MedicalNote", back_populates="patient")
    imaging_studies = relationship("ImagingStudy", back_populates="patient")

class PatientInsurance(Base):
    __tablename__ = "patient_insurance"
    
    insurance_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.patient_id'), nullable=False)
    
    # Insurance Details
    insurance_type = Column(String(50), nullable=False)
    insurance_company = Column(String(255), nullable=False)
    policy_number = Column(String(100), nullable=False)
    group_number = Column(String(100))
    subscriber_name = Column(String(255))
    subscriber_id = Column(String(100))
    
    # Coverage Details
    is_primary = Column(Boolean, default=True)
    effective_date = Column(Date)
    expiration_date = Column(Date)
    copay_amount = Column(Numeric(10, 2))
    deductible_amount = Column(Numeric(10, 2))
    out_of_pocket_max = Column(Numeric(10, 2))
    
    # System Fields
    active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    patient = relationship("Patient", back_populates="insurance_plans")

class Appointment(Base):
    __tablename__ = "appointments"
    
    appointment_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.patient_id'), nullable=False)
    provider_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=False)
    
    # Appointment Details
    appointment_type = Column(String(50), nullable=False)
    appointment_date = Column(Date, nullable=False)
    appointment_time = Column(Time, nullable=False)
    duration_minutes = Column(Integer, default=30)
    
    # Status and Notes
    status = Column(String(20), default=AppointmentStatus.SCHEDULED)
    chief_complaint = Column(Text)
    notes = Column(Text)
    
    # Location
    room_number = Column(String(20))
    facility = Column(String(255))
    
    # System Fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    
    # Relationships
    patient = relationship("Patient", back_populates="appointments")

class Procedure(Base):
    __tablename__ = "procedures"
    
    procedure_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.patient_id'), nullable=False)
    provider_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=False)
    
    # Procedure Details
    procedure_name = Column(String(255), nullable=False)
    procedure_code = Column(String(20))  # CPT code
    icd10_diagnosis_codes = Column(ARRAY(String))
    
    # Scheduling
    scheduled_date = Column(DateTime(timezone=True))
    completed_date = Column(DateTime(timezone=True))
    duration_minutes = Column(Integer)
    
    # Clinical Details
    pre_op_diagnosis = Column(Text)
    post_op_diagnosis = Column(Text)
    procedure_notes = Column(Text)
    complications = Column(Text)
    anesthesia_type = Column(String(100))
    
    # Status
    status = Column(String(20), default=ProcedureStatus.PLANNED)
    
    # Vector embedding for semantic search
    content_embedding = Column(Vector(384))  # 384 dimensions for all-MiniLM-L6-v2
    
    # System Fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    
    # Relationships
    patient = relationship("Patient", back_populates="procedures")
    billing_records = relationship("BillingRecord", back_populates="procedure")

class MedicalNote(Base):
    __tablename__ = "medical_notes"
    
    note_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.patient_id'), nullable=False)
    provider_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=False)
    appointment_id = Column(UUID(as_uuid=True), ForeignKey('appointments.appointment_id'))
    
    # Note Details
    note_type = Column(String(100))  # Progress Note, Operative Note, Discharge Summary, etc.
    chief_complaint = Column(Text)
    history_present_illness = Column(Text)
    review_of_systems = Column(Text)
    physical_exam = Column(Text)
    assessment = Column(Text)
    plan = Column(Text)
    
    # Structured Data
    vital_signs = Column(JSONB)
    medications_prescribed = Column(JSONB)
    
    # Vector embedding for semantic search
    content_embedding = Column(Vector(384))  # 384 dimensions for all-MiniLM-L6-v2
    
    # System Fields
    note_date = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    patient = relationship("Patient", back_populates="medical_notes")

class ImagingStudy(Base):
    __tablename__ = "imaging_studies"
    
    study_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.patient_id'), nullable=False)
    ordering_provider_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=False)
    
    # Study Details
    study_type = Column(String(50), nullable=False)
    body_part = Column(String(100))
    study_date = Column(DateTime(timezone=True))
    
    # Clinical Information
    clinical_indication = Column(Text)
    findings = Column(Text)
    impression = Column(Text)
    
    # Technical Details
    modality = Column(String(50))
    study_uid = Column(String(255))  # DICOM Study Instance UID
    accession_number = Column(String(50))
    
    # Vector embedding for semantic search
    content_embedding = Column(Vector(384))  # 384 dimensions for all-MiniLM-L6-v2
    
    # System Fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    patient = relationship("Patient", back_populates="imaging_studies")

class BillingRecord(Base):
    __tablename__ = "billing_records"
    
    billing_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.patient_id'), nullable=False)
    procedure_id = Column(UUID(as_uuid=True), ForeignKey('procedures.procedure_id'))
    
    # Billing Details
    service_date = Column(Date, nullable=False)
    cpt_code = Column(String(20), nullable=False)
    description = Column(String(255), nullable=False)
    units = Column(Integer, default=1)
    
    # Financial Information
    charge_amount = Column(Numeric(10, 2), nullable=False)
    allowed_amount = Column(Numeric(10, 2))
    paid_amount = Column(Numeric(10, 2))
    patient_responsibility = Column(Numeric(10, 2))
    
    # Insurance Processing
    primary_insurance_paid = Column(Numeric(10, 2))
    secondary_insurance_paid = Column(Numeric(10, 2))
    adjustment_amount = Column(Numeric(10, 2))
    
    # Status and Processing
    status = Column(String(20), default=BillingStatus.PENDING)
    claim_number = Column(String(50))
    date_submitted = Column(Date)
    date_paid = Column(Date)
    
    # System Fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    patient = relationship("Patient", back_populates="billing_records")
    procedure = relationship("Procedure", back_populates="billing_records")

class TreatmentPlan(Base):
    __tablename__ = "treatment_plans"
    
    plan_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.patient_id'), nullable=False)
    provider_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=False)
    
    # Plan Details
    diagnosis = Column(Text, nullable=False)
    treatment_goals = Column(Text)
    treatment_timeline = Column(Text)
    
    # Plan Components
    surgical_plan = Column(Text)
    rehabilitation_plan = Column(Text)
    medication_plan = Column(JSONB)
    follow_up_schedule = Column(JSONB)
    
    # Status
    active = Column(Boolean, default=True)
    start_date = Column(Date)
    completion_date = Column(Date)
    
    # System Fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class PhysicalTherapy(Base):
    __tablename__ = "physical_therapy"
    
    pt_session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.patient_id'), nullable=False)
    therapist_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=False)
    
    # Session Details
    session_date = Column(Date, nullable=False)
    session_number = Column(Integer)
    total_sessions_planned = Column(Integer)
    
    # Treatment Details
    exercises_performed = Column(JSONB)
    range_of_motion = Column(JSONB)
    pain_level_before = Column(Integer)  # 1-10 scale
    pain_level_after = Column(Integer)   # 1-10 scale
    
    # Progress Notes
    session_notes = Column(Text)
    patient_compliance = Column(String(50))
    home_exercise_program = Column(Text)
    
    # System Fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())