"""
Database package for Orthopedics EMR RAG System.
"""

from .connection import DatabaseManager, db_manager, get_db_session, check_database_health
from .models import (
    Base, Role, User, Session, AuditLog, SystemSetting, SecurityEvent,
    UserStatus, AuditAction, ResourceType, SecuritySeverity, SecurityStatus
)
from .patient_models import (
    Patient, PatientInsurance, Appointment, Procedure, MedicalNote, 
    ImagingStudy, BillingRecord, TreatmentPlan, PhysicalTherapy,
    PatientStatus, Gender, InsuranceType, AppointmentStatus, AppointmentType,
    ProcedureStatus, BillingStatus, ImagingType
)

__all__ = [
    "DatabaseManager",
    "db_manager", 
    "get_db_session",
    "check_database_health",
    "Base",
    "Role",
    "User", 
    "Session",
    "AuditLog",
    "SystemSetting",
    "SecurityEvent",
    "UserStatus",
    "AuditAction", 
    "ResourceType",
    "SecuritySeverity",
    "SecurityStatus",
    # Patient Models
    "Patient",
    "PatientInsurance", 
    "Appointment",
    "Procedure",
    "MedicalNote",
    "ImagingStudy",
    "BillingRecord",
    "TreatmentPlan",
    "PhysicalTherapy",
    # Patient Enums
    "PatientStatus",
    "Gender",
    "InsuranceType",
    "AppointmentStatus",
    "AppointmentType",
    "ProcedureStatus", 
    "BillingStatus",
    "ImagingType",
]