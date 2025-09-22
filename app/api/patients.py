"""
Patient API endpoints for Orthopedic EMR system
HIPAA-compliant patient data access with RBAC protection
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Depends, Query
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.orm import selectinload
from datetime import datetime, date
import logging

from core.dependencies import CurrentUser, get_current_user, require_phi_access
from middleware.rbac import Permission
from schemas.patients import (
    PatientSummary, PatientDetail, PatientComprehensive, PatientListResponse,
    PatientSearch, PatientUpdate, PatientStatistics, TreatmentSummary, BillingSummary
)
from database import (
    db_manager, Patient, PatientInsurance, Appointment, Procedure, 
    MedicalNote, ImagingStudy, BillingRecord, PhysicalTherapy, User
)

# Simple error response class for this module
class ErrorResponse:
    def __init__(self, error: str, message: str):
        self.error = error
        self.message = message
    
    def model_dump(self):
        return {"error": self.error, "message": self.message}

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get(
    "/",
    response_model=PatientListResponse,
    summary="List patients",
    description="Get paginated list of patients with search and filtering"
)
async def list_patients(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search in name or MRN"),
    gender: Optional[str] = Query(None, description="Filter by gender"),
    status: Optional[str] = Query(None, description="Filter by status"),
    current_user: CurrentUser = Depends(require_phi_access())
):
    """
    List patients with pagination and filtering.
    Requires PHI access permissions.
    """
    try:
        async with db_manager.get_session() as session:
            # Build base query
            query = select(Patient)
            count_query = select(func.count(Patient.patient_id))
            
            # Apply filters
            conditions = []
            
            if search:
                search_condition = or_(
                    Patient.first_name.ilike(f"%{search}%"),
                    Patient.last_name.ilike(f"%{search}%"),
                    Patient.medical_record_number.ilike(f"%{search}%")
                )
                conditions.append(search_condition)
            
            if gender:
                conditions.append(Patient.gender == gender)
                
            if status:
                conditions.append(Patient.status == status)
            
            if conditions:
                query = query.where(and_(*conditions))
                count_query = count_query.where(and_(*conditions))
            
            # Get total count
            total_result = await session.execute(count_query)
            total = total_result.scalar()
            
            # Apply pagination and ordering
            query = query.order_by(Patient.last_name, Patient.first_name)
            query = query.offset((page - 1) * per_page).limit(per_page)
            
            # Execute query
            result = await session.execute(query)
            patients = result.scalars().all()
            
            # Convert to response models
            patient_summaries = []
            for patient in patients:
                summary = PatientSummary.model_validate(patient)
                patient_summaries.append(summary)
            
            total_pages = (total + per_page - 1) // per_page
            
            return PatientListResponse(
                patients=patient_summaries,
                total=total,
                page=page,
                per_page=per_page,
                total_pages=total_pages
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve patients: {str(e)}"
        )

@router.get(
    "/{patient_id}",
    response_model=PatientDetail,
    summary="Get patient details",
    description="Get detailed patient information by ID"
)
async def get_patient(
    patient_id: str,
    current_user: CurrentUser = Depends(require_phi_access())
):
    """
    Get detailed patient information.
    Requires PHI access permissions.
    """
    try:
        async with db_manager.get_session() as session:
            # Get patient with insurance information
            query = select(Patient).where(Patient.patient_id == patient_id)
            result = await session.execute(query)
            patient = result.scalar_one_or_none()
            
            if not patient:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Patient not found"
                )
            
            return PatientDetail.model_validate(patient)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve patient: {str(e)}"
        )

@router.get(
    "/{patient_id}/comprehensive",
    response_model=PatientComprehensive,
    summary="Get comprehensive patient record",
    description="Get complete patient information with all related medical data"
)
async def get_patient_comprehensive(
    patient_id: str,
    current_user: CurrentUser = Depends(require_phi_access())
):
    """
    Get comprehensive patient record with all related data.
    Requires PHI access permissions.
    """
    try:
        async with db_manager.get_session() as session:
            # Get patient
            patient_query = select(Patient).where(Patient.patient_id == patient_id)
            patient_result = await session.execute(patient_query)
            patient = patient_result.scalar_one_or_none()
            
            if not patient:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Patient not found"
                )
            
            # Get insurance plans
            insurance_query = select(PatientInsurance).where(
                PatientInsurance.patient_id == patient_id,
                PatientInsurance.active == True
            )
            insurance_result = await session.execute(insurance_query)
            insurance_plans = insurance_result.scalars().all()
            
            # Get recent appointments (last 6 months)
            six_months_ago = date.today().replace(month=date.today().month-6 if date.today().month > 6 else date.today().month+6)
            appointments_query = select(Appointment, User.first_name, User.last_name).join(
                User, Appointment.provider_id == User.user_id
            ).where(
                Appointment.patient_id == patient_id,
                Appointment.appointment_date >= six_months_ago
            ).order_by(desc(Appointment.appointment_date))
            appointments_result = await session.execute(appointments_query)
            appointments_data = appointments_result.all()
            
            # Get procedures
            procedures_query = select(Procedure, User.first_name, User.last_name).join(
                User, Procedure.provider_id == User.user_id
            ).where(
                Procedure.patient_id == patient_id
            ).order_by(desc(Procedure.scheduled_date))
            procedures_result = await session.execute(procedures_query)
            procedures_data = procedures_result.all()
            
            # Get recent medical notes (last 3 months)
            three_months_ago = datetime.now().replace(month=datetime.now().month-3 if datetime.now().month > 3 else datetime.now().month+9)
            notes_query = select(MedicalNote, User.first_name, User.last_name).join(
                User, MedicalNote.provider_id == User.user_id
            ).where(
                MedicalNote.patient_id == patient_id,
                MedicalNote.note_date >= three_months_ago
            ).order_by(desc(MedicalNote.note_date))
            notes_result = await session.execute(notes_query)
            notes_data = notes_result.all()
            
            # Get imaging studies
            imaging_query = select(ImagingStudy, User.first_name, User.last_name).join(
                User, ImagingStudy.ordering_provider_id == User.user_id
            ).where(
                ImagingStudy.patient_id == patient_id
            ).order_by(desc(ImagingStudy.study_date))
            imaging_result = await session.execute(imaging_query)
            imaging_data = imaging_result.all()
            
            # Get billing records
            billing_query = select(BillingRecord).where(
                BillingRecord.patient_id == patient_id
            ).order_by(desc(BillingRecord.service_date))
            billing_result = await session.execute(billing_query)
            billing_records = billing_result.scalars().all()
            
            # Get PT sessions
            pt_query = select(PhysicalTherapy).where(
                PhysicalTherapy.patient_id == patient_id
            ).order_by(desc(PhysicalTherapy.session_date))
            pt_result = await session.execute(pt_query)
            pt_sessions = pt_result.scalars().all()
            
            # Build comprehensive response
            patient_detail = PatientDetail.model_validate(patient)
            
            # Convert related data with provider names
            recent_appointments = []
            for appt, first_name, last_name in appointments_data:
                appt_data = appt.__dict__.copy()
                appt_data['provider_name'] = f"{first_name} {last_name}"
                recent_appointments.append(appt_data)
            
            procedures = []
            for proc, first_name, last_name in procedures_data:
                proc_data = proc.__dict__.copy()
                proc_data['provider_name'] = f"{first_name} {last_name}"
                procedures.append(proc_data)
            
            recent_notes = []
            for note, first_name, last_name in notes_data:
                note_data = note.__dict__.copy()
                note_data['provider_name'] = f"{first_name} {last_name}"
                recent_notes.append(note_data)
            
            imaging_studies = []
            for study, first_name, last_name in imaging_data:
                study_data = study.__dict__.copy()
                study_data['provider_name'] = f"{first_name} {last_name}"
                imaging_studies.append(study_data)
            
            return PatientComprehensive(
                **patient_detail.model_dump(),
                insurance_plans=[plan.__dict__ for plan in insurance_plans],
                recent_appointments=recent_appointments,
                procedures=procedures,
                recent_notes=recent_notes,
                imaging_studies=imaging_studies,
                billing_records=[record.__dict__ for record in billing_records],
                pt_sessions=[session.__dict__ for session in pt_sessions]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve comprehensive patient data: {str(e)}"
        )

@router.put(
    "/{patient_id}",
    response_model=PatientDetail,
    summary="Update patient information",
    description="Update patient demographic and contact information"
)
async def update_patient(
    patient_id: str,
    patient_update: PatientUpdate,
    current_user: CurrentUser = Depends(require_phi_access())
):
    """
    Update patient information.
    Requires PHI access permissions.
    """
    # Check if user has write permissions
    if not current_user.has_permission(Permission.PHI_WRITE):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="PHI write permission required"
        )
    
    try:
        async with db_manager.get_session() as session:
            # Get patient
            query = select(Patient).where(Patient.patient_id == patient_id)
            result = await session.execute(query)
            patient = result.scalar_one_or_none()
            
            if not patient:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Patient not found"
                )
            
            # Update fields
            update_data = patient_update.model_dump(exclude_unset=True)
            for field, value in update_data.items():
                setattr(patient, field, value)
            
            # Update timestamp
            patient.updated_at = datetime.now()
            
            await session.commit()
            await session.refresh(patient)
            
            return PatientDetail.model_validate(patient)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update patient: {str(e)}"
        )

@router.get(
    "/{patient_id}/treatment-summary",
    response_model=TreatmentSummary,
    summary="Get patient treatment summary",
    description="Get summary of patient's treatments and care"
)
async def get_treatment_summary(
    patient_id: str,
    current_user: CurrentUser = Depends(require_phi_access())
):
    """
    Get treatment summary for a patient.
    Requires PHI access permissions.
    """
    try:
        async with db_manager.get_session() as session:
            # Verify patient exists
            patient_query = select(Patient).where(Patient.patient_id == patient_id)
            patient_result = await session.execute(patient_query)
            patient = patient_result.scalar_one_or_none()
            
            if not patient:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Patient not found"
                )
            
            # Get counts
            procedures_count = await session.execute(
                select(func.count(Procedure.procedure_id)).where(Procedure.patient_id == patient_id)
            )
            total_procedures = procedures_count.scalar()
            
            appointments_count = await session.execute(
                select(func.count(Appointment.appointment_id)).where(Appointment.patient_id == patient_id)
            )
            total_appointments = appointments_count.scalar()
            
            pt_count = await session.execute(
                select(func.count(PhysicalTherapy.pt_session_id)).where(PhysicalTherapy.patient_id == patient_id)
            )
            total_pt_sessions = pt_count.scalar()
            
            # Get last visit date
            last_visit = await session.execute(
                select(func.max(Appointment.appointment_date)).where(
                    Appointment.patient_id == patient_id,
                    Appointment.status == 'completed'
                )
            )
            last_visit_date = last_visit.scalar()
            
            # Get next appointment
            next_appt = await session.execute(
                select(func.min(Appointment.appointment_date)).where(
                    Appointment.patient_id == patient_id,
                    Appointment.appointment_date >= date.today(),
                    Appointment.status.in_(['scheduled', 'confirmed'])
                )
            )
            next_appointment_date = next_appt.scalar()
            
            # Get active treatments
            active_procedures = await session.execute(
                select(Procedure.procedure_name).where(
                    Procedure.patient_id == patient_id,
                    Procedure.status.in_(['planned', 'scheduled', 'in_progress'])
                )
            )
            active_treatments = [proc[0] for proc in active_procedures.all()]
            
            return TreatmentSummary(
                patient_id=patient_id,
                total_procedures=total_procedures,
                total_appointments=total_appointments,
                total_pt_sessions=total_pt_sessions,
                last_visit_date=last_visit_date,
                next_appointment_date=next_appointment_date,
                active_treatments=active_treatments
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve treatment summary: {str(e)}"
        )

@router.get(
    "/{patient_id}/billing-summary",
    response_model=BillingSummary,
    summary="Get patient billing summary",
    description="Get summary of patient's billing and payments"
)
async def get_billing_summary(
    patient_id: str,
    current_user: CurrentUser = Depends(require_phi_access())
):
    """
    Get billing summary for a patient.
    Requires PHI access permissions.
    """
    try:
        async with db_manager.get_session() as session:
            # Verify patient exists
            patient_query = select(Patient).where(Patient.patient_id == patient_id)
            patient_result = await session.execute(patient_query)
            patient = patient_result.scalar_one_or_none()
            
            if not patient:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Patient not found"
                )
            
            # Get billing aggregates
            billing_summary = await session.execute(
                select(
                    func.sum(BillingRecord.charge_amount).label('total_charges'),
                    func.sum(BillingRecord.paid_amount).label('total_paid'),
                    func.sum(BillingRecord.patient_responsibility).label('outstanding_balance'),
                    func.sum(BillingRecord.primary_insurance_paid + func.coalesce(BillingRecord.secondary_insurance_paid, 0)).label('insurance_paid'),
                    func.max(BillingRecord.date_paid).label('last_payment_date')
                ).where(BillingRecord.patient_id == patient_id)
            )
            
            summary_result = billing_summary.first()
            
            return BillingSummary(
                patient_id=patient_id,
                total_charges=summary_result.total_charges or 0,
                total_paid=summary_result.total_paid or 0,
                outstanding_balance=summary_result.outstanding_balance or 0,
                insurance_paid=summary_result.insurance_paid or 0,
                last_payment_date=summary_result.last_payment_date
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve billing summary: {str(e)}"
        )

@router.get(
    "/statistics/overview",
    response_model=PatientStatistics,
    summary="Get patient statistics",
    description="Get overall patient statistics for dashboard"
)
async def get_patient_statistics(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get patient statistics.
    Requires basic user access.
    """
    try:
        async with db_manager.get_session() as session:
            # Total patients
            total_count = await session.execute(select(func.count(Patient.patient_id)))
            total_patients = total_count.scalar()
            
            # Active patients
            active_count = await session.execute(
                select(func.count(Patient.patient_id)).where(Patient.status == 'active')
            )
            active_patients = active_count.scalar()
            
            # Patients by gender
            gender_stats = await session.execute(
                select(Patient.gender, func.count(Patient.patient_id)).group_by(Patient.gender)
            )
            patients_by_gender = {gender: count for gender, count in gender_stats.all()}
            
            # Patients by age group (simplified)
            age_groups = {
                "0-18": 0, "19-30": 0, "31-50": 0, "51-70": 0, "70+": 0
            }
            
            patients_with_ages = await session.execute(select(Patient.date_of_birth))
            for (dob,) in patients_with_ages.all():
                if dob:
                    age = date.today().year - dob.year - ((date.today().month, date.today().day) < (dob.month, dob.day))
                    if age <= 18:
                        age_groups["0-18"] += 1
                    elif age <= 30:
                        age_groups["19-30"] += 1
                    elif age <= 50:
                        age_groups["31-50"] += 1
                    elif age <= 70:
                        age_groups["51-70"] += 1
                    else:
                        age_groups["70+"] += 1
            
            # Recent registrations (last 30 days)
            thirty_days_ago = datetime.now() - datetime.timedelta(days=30)
            recent_count = await session.execute(
                select(func.count(Patient.patient_id)).where(Patient.created_at >= thirty_days_ago)
            )
            recent_registrations = recent_count.scalar()
            
            return PatientStatistics(
                total_patients=total_patients,
                active_patients=active_patients,
                patients_by_gender=patients_by_gender,
                patients_by_age_group=age_groups,
                recent_registrations=recent_registrations
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve patient statistics: {str(e)}"
        )

@router.post(
    "/search",
    response_model=PatientListResponse,
    summary="Advanced patient search",
    description="Search patients with multiple criteria and filters"
)
async def search_patients(
    search_params: PatientSearch,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("last_name", description="Sort field: last_name, first_name, created_at, medical_record_number"),
    sort_order: str = Query("asc", description="Sort order: asc or desc"),
    current_user: CurrentUser = Depends(require_phi_access())
):
    """
    Advanced patient search with multiple criteria.
    Supports searching by demographics, contact info, medical conditions, and date ranges.
    Requires PHI access permissions.
    """
    try:
        async with db_manager.get_session() as session:
            # Import required models for joins
            from database.patient_models import MedicalNote, ImagingStudy
            
            # Build base query with joins for related data
            query = select(Patient).outerjoin(Appointment).outerjoin(Procedure).outerjoin(MedicalNote)
            count_query = select(func.count(Patient.patient_id.distinct()))
            
            # Apply search conditions
            conditions = []
            
            # Basic search across multiple fields
            if search_params.search_term:
                search_term = search_params.search_term
                basic_search = or_(
                    Patient.first_name.ilike(f"%{search_term}%"),
                    Patient.last_name.ilike(f"%{search_term}%"),
                    Patient.medical_record_number.ilike(f"%{search_term}%"),
                    Patient.phone_primary.ilike(f"%{search_term}%"),
                    Patient.phone_secondary.ilike(f"%{search_term}%")
                )
                conditions.append(basic_search)
            
            # Exact MRN search
            if search_params.medical_record_number:
                conditions.append(Patient.medical_record_number == search_params.medical_record_number)
            
            # Demographics
            if search_params.first_name:
                conditions.append(Patient.first_name.ilike(f"%{search_params.first_name}%"))
            if search_params.last_name:
                conditions.append(Patient.last_name.ilike(f"%{search_params.last_name}%"))
            if search_params.gender:
                conditions.append(Patient.gender == search_params.gender)
            if search_params.status:
                conditions.append(Patient.status == search_params.status)
            if search_params.date_of_birth:
                conditions.append(Patient.date_of_birth == search_params.date_of_birth)
            
            # Age range calculations
            if search_params.age_min is not None or search_params.age_max is not None:
                from datetime import timedelta
                today = date.today()
                
                if search_params.age_min is not None:
                    max_birth_date = today - timedelta(days=search_params.age_min * 365.25)
                    conditions.append(Patient.date_of_birth <= max_birth_date)
                
                if search_params.age_max is not None:
                    min_birth_date = today - timedelta(days=(search_params.age_max + 1) * 365.25)
                    conditions.append(Patient.date_of_birth > min_birth_date)
            
            # Contact information
            if search_params.phone:
                phone_search = or_(
                    Patient.phone_primary.ilike(f"%{search_params.phone}%"),
                    Patient.phone_secondary.ilike(f"%{search_params.phone}%")
                )
                conditions.append(phone_search)
            if search_params.email:
                conditions.append(Patient.email.ilike(f"%{search_params.email}%"))
            if search_params.city:
                conditions.append(Patient.city.ilike(f"%{search_params.city}%"))
            if search_params.state:
                conditions.append(Patient.state.ilike(f"%{search_params.state}%"))
            if search_params.zip_code:
                conditions.append(Patient.zip_code == search_params.zip_code)
            
            # Medical information
            if search_params.has_allergies is not None:
                if search_params.has_allergies:
                    conditions.append(Patient.allergies.isnot(None))
                    conditions.append(Patient.allergies != "")
                else:
                    conditions.append(or_(Patient.allergies.is_(None), Patient.allergies == ""))
            
            if search_params.allergy_search:
                conditions.append(Patient.allergies.ilike(f"%{search_params.allergy_search}%"))
            
            if search_params.primary_care_physician:
                conditions.append(Patient.primary_care_physician.ilike(f"%{search_params.primary_care_physician}%"))
            
            # Date ranges
            if search_params.created_after:
                conditions.append(Patient.created_at >= search_params.created_after)
            if search_params.created_before:
                conditions.append(Patient.created_at <= search_params.created_before)
            
            # Treatment status filters (using subqueries for better performance)
            if search_params.has_upcoming_appointments is not None:
                upcoming_appt_subquery = select(Appointment.patient_id).where(
                    Appointment.appointment_date > date.today()
                ).where(Appointment.status.in_(["scheduled", "confirmed"]))
                
                if search_params.has_upcoming_appointments:
                    conditions.append(Patient.patient_id.in_(upcoming_appt_subquery))
                else:
                    conditions.append(~Patient.patient_id.in_(upcoming_appt_subquery))
            
            if search_params.has_recent_procedures is not None:
                from datetime import timedelta
                recent_date = date.today() - timedelta(days=90)
                recent_proc_subquery = select(Procedure.patient_id).where(
                    Procedure.completed_date >= recent_date
                )
                
                if search_params.has_recent_procedures:
                    conditions.append(Patient.patient_id.in_(recent_proc_subquery))
                else:
                    conditions.append(~Patient.patient_id.in_(recent_proc_subquery))
            
            # Medical procedures and conditions filtering
            if search_params.procedure_name:
                procedure_name_subquery = select(Procedure.patient_id).where(
                    Procedure.procedure_name.ilike(f"%{search_params.procedure_name}%")
                )
                conditions.append(Patient.patient_id.in_(procedure_name_subquery))
            
            if search_params.procedure_code:
                procedure_code_subquery = select(Procedure.patient_id).where(
                    Procedure.procedure_code.ilike(f"%{search_params.procedure_code}%")
                )
                conditions.append(Patient.patient_id.in_(procedure_code_subquery))
            
            if search_params.icd10_code:
                icd10_subquery = select(Procedure.patient_id).where(
                    func.array_to_string(Procedure.icd10_diagnosis_codes, ',').ilike(f"%{search_params.icd10_code}%")
                )
                conditions.append(Patient.patient_id.in_(icd10_subquery))
            
            if search_params.procedure_status:
                procedure_status_subquery = select(Procedure.patient_id).where(
                    Procedure.status == search_params.procedure_status
                )
                conditions.append(Patient.patient_id.in_(procedure_status_subquery))
            
            if search_params.procedure_date_after:
                proc_date_after_subquery = select(Procedure.patient_id).where(
                    Procedure.scheduled_date >= search_params.procedure_date_after
                )
                conditions.append(Patient.patient_id.in_(proc_date_after_subquery))
            
            if search_params.procedure_date_before:
                proc_date_before_subquery = select(Procedure.patient_id).where(
                    Procedure.scheduled_date <= search_params.procedure_date_before
                )
                conditions.append(Patient.patient_id.in_(proc_date_before_subquery))
            
            # Medical conditions and notes filtering
            if search_params.medical_condition:
                medical_condition_search = or_(
                    Patient.medical_history.ilike(f"%{search_params.medical_condition}%"),
                    Patient.family_history.ilike(f"%{search_params.medical_condition}%"),
                    Patient.social_history.ilike(f"%{search_params.medical_condition}%")
                )
                conditions.append(medical_condition_search)
            
            if search_params.chief_complaint:
                chief_complaint_subquery = select(func.distinct(MedicalNote.patient_id)).where(
                    MedicalNote.chief_complaint.ilike(f"%{search_params.chief_complaint}%")
                )
                appointment_complaint_subquery = select(func.distinct(Appointment.patient_id)).where(
                    Appointment.chief_complaint.ilike(f"%{search_params.chief_complaint}%")
                )
                
                complaint_search = or_(
                    Patient.patient_id.in_(chief_complaint_subquery),
                    Patient.patient_id.in_(appointment_complaint_subquery)
                )
                conditions.append(complaint_search)
            
            if search_params.diagnosis_search:
                diagnosis_subquery_proc = select(Procedure.patient_id).where(
                    or_(
                        Procedure.pre_op_diagnosis.ilike(f"%{search_params.diagnosis_search}%"),
                        Procedure.post_op_diagnosis.ilike(f"%{search_params.diagnosis_search}%")
                    )
                )
                diagnosis_subquery_note = select(MedicalNote.patient_id).where(
                    MedicalNote.assessment.ilike(f"%{search_params.diagnosis_search}%")
                )
                
                diagnosis_search = or_(
                    Patient.patient_id.in_(diagnosis_subquery_proc),
                    Patient.patient_id.in_(diagnosis_subquery_note)
                )
                conditions.append(diagnosis_search)
            
            if search_params.note_content:
                note_content_subquery = select(MedicalNote.patient_id).where(
                    or_(
                        MedicalNote.assessment.ilike(f"%{search_params.note_content}%"),
                        MedicalNote.plan.ilike(f"%{search_params.note_content}%")
                    )
                )
                conditions.append(Patient.patient_id.in_(note_content_subquery))
            
            # Apply all conditions
            if conditions:
                query = query.where(and_(*conditions))
                count_query = count_query.where(and_(*conditions))
            
            # Make queries distinct to avoid duplicates from joins
            query = query.distinct()
            
            # Get total count
            total_result = await session.execute(count_query)
            total = total_result.scalar()
            
            # Apply sorting
            sort_column = getattr(Patient, sort_by, Patient.last_name)
            if sort_order.lower() == "desc":
                query = query.order_by(sort_column.desc())
            else:
                query = query.order_by(sort_column.asc())
            
            # Add secondary sort for consistency
            if sort_by != "last_name":
                query = query.order_by(Patient.last_name, Patient.first_name)
            
            # Apply pagination
            query = query.offset((page - 1) * per_page).limit(per_page)
            
            # Execute query
            result = await session.execute(query)
            patients = result.scalars().all()
            
            # Convert to response models
            patient_summaries = []
            for patient in patients:
                # Calculate age
                today = date.today()
                age = today.year - patient.date_of_birth.year - ((today.month, today.day) < (patient.date_of_birth.month, patient.date_of_birth.day))
                
                summary = PatientSummary(
                    patient_id=str(patient.patient_id),
                    medical_record_number=patient.medical_record_number,
                    first_name=patient.first_name,
                    last_name=patient.last_name,
                    date_of_birth=patient.date_of_birth,
                    gender=patient.gender,
                    status=patient.status,
                    primary_care_physician=patient.primary_care_physician,
                    created_at=patient.created_at,
                    age=age
                )
                patient_summaries.append(summary)
            
            # Calculate pagination info
            total_pages = (total + per_page - 1) // per_page
            
            return PatientListResponse(
                patients=patient_summaries,
                total=total,
                page=page,
                per_page=per_page,
                total_pages=total_pages
            )
            
    except Exception as e:
        logger.error(f"Error searching patients: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="PATIENT_SEARCH_ERROR",
                message="Failed to search patients"
            ).model_dump()
        )