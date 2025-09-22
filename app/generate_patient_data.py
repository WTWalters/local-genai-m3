#!/usr/bin/env python3
"""
Generate comprehensive patient data for Orthopedic EMR system
Creates 5 realistic patient cases with complete medical workflows
"""

import asyncio
import uuid
from datetime import datetime, date, timedelta
from decimal import Decimal
from database import (
    db_manager, Patient, PatientInsurance, Appointment, Procedure, 
    MedicalNote, ImagingStudy, BillingRecord, TreatmentPlan, PhysicalTherapy,
    PatientStatus, Gender, InsuranceType, AppointmentStatus, AppointmentType,
    ProcedureStatus, BillingStatus, ImagingType
)

class PatientDataGenerator:
    def __init__(self):
        self.providers = {
            "attending_physician": "d8114311-5b8b-4bdb-9c4a-91f87372925f",  # dr_smith from our existing data
            "resident": None,  # We'll create these
            "nurse": None,
            "pt": None  # Physical therapist
        }
        
    async def generate_all_patients(self):
        """Generate all 5 comprehensive patient cases."""
        await db_manager.initialize()
        
        print("🏥 Generating comprehensive patient database...")
        
        # Generate each patient case
        patients = [
            await self.create_shoulder_replacement_patient(),
            await self.create_knee_replacement_patient(), 
            await self.create_acl_repair_patient(),
            await self.create_compound_fracture_patient(),
            await self.create_self_pay_fracture_patient()
        ]
        
        print("✅ All patient data generated successfully!")
        print(f"📊 Created {len(patients)} patients with comprehensive medical records")
        
        await db_manager.close()
        return patients

    async def create_shoulder_replacement_patient(self):
        """Patient 1: Margaret Thompson - Total Shoulder Replacement"""
        print("👩‍🦳 Creating Patient 1: Shoulder Replacement Case...")
        
        async with db_manager.get_session() as session:
            # Create patient
            patient = Patient(
                medical_record_number="MRN001234",
                first_name="Margaret",
                last_name="Thompson",
                middle_name="Rose",
                date_of_birth=date(1955, 3, 15),
                gender=Gender.FEMALE,
                phone_primary="555-0123",
                email="margaret.thompson@email.com",
                address_line1="123 Oak Street",
                city="San Francisco",
                state="CA",
                zip_code="94102",
                primary_care_physician="Dr. Robert Williams",
                allergies="Penicillin (rash), Latex (contact dermatitis)",
                current_medications=[
                    {"name": "Lisinopril", "dosage": "10mg", "frequency": "daily"},
                    {"name": "Metformin", "dosage": "500mg", "frequency": "twice daily"},
                    {"name": "Vitamin D3", "dosage": "2000 IU", "frequency": "daily"}
                ],
                medical_history="Type 2 diabetes (controlled), Hypertension, Osteoarthritis",
                family_history="Mother: Heart disease, Father: Diabetes",
                social_history="Retired teacher, non-smoker, occasional alcohol use",
                emergency_contact_name="David Thompson",
                emergency_contact_phone="555-0124",
                emergency_contact_relationship="Son"
            )
            session.add(patient)
            await session.flush()
            
            # Primary Insurance - Medicare
            insurance = PatientInsurance(
                patient_id=patient.patient_id,
                insurance_type=InsuranceType.MEDICARE,
                insurance_company="Medicare Part A & B",
                policy_number="123456789A",
                subscriber_name="Margaret Rose Thompson",
                subscriber_id="123456789A",
                is_primary=True,
                effective_date=date(2020, 1, 1),
                copay_amount=Decimal("25.00"),
                deductible_amount=Decimal("1500.00")
            )
            session.add(insurance)
            
            # Supplemental Insurance
            supp_insurance = PatientInsurance(
                patient_id=patient.patient_id,
                insurance_type=InsuranceType.PRIVATE,
                insurance_company="Blue Cross Blue Shield Medigap",
                policy_number="SUP789123",
                subscriber_name="Margaret Rose Thompson",
                subscriber_id="SUP789123",
                is_primary=False,
                effective_date=date(2020, 1, 1),
                copay_amount=Decimal("0.00"),
                deductible_amount=Decimal("0.00")
            )
            session.add(supp_insurance)
            
            # Initial Consultation - 6 months ago
            initial_appt = Appointment(
                patient_id=patient.patient_id,
                provider_id=uuid.UUID(self.providers["attending_physician"]),
                appointment_type=AppointmentType.CONSULTATION,
                appointment_date=date.today() - timedelta(days=180),
                appointment_time=datetime.strptime("10:00", "%H:%M").time(),
                duration_minutes=60,
                status=AppointmentStatus.COMPLETED,
                chief_complaint="Right shoulder pain and stiffness, worsening over 2 years",
                notes="Patient reports severe glenohumeral arthritis affecting daily activities"
            )
            session.add(initial_appt)
            await session.flush()
            
            # Initial Medical Note
            initial_note = MedicalNote(
                patient_id=patient.patient_id,
                provider_id=uuid.UUID(self.providers["attending_physician"]),
                appointment_id=initial_appt.appointment_id,
                note_type="Initial Consultation",
                chief_complaint="Progressive right shoulder pain and loss of function",
                history_present_illness="68-year-old retired teacher with 2-year history of progressive right shoulder pain. Pain is worse with overhead activities and at night. Significant functional limitation affecting daily activities including dressing and reaching.",
                physical_exam="Right shoulder: Limited active ROM - forward flexion 90°, abduction 80°, external rotation 30°. Positive impingement signs. Tenderness over anterior glenohumeral joint. No neurovascular deficits.",
                assessment="Severe glenohumeral osteoarthritis with functional impairment",
                plan="1. MRI right shoulder 2. Conservative management trial with PT 3. Discuss surgical options if conservative treatment fails",
                vital_signs={
                    "blood_pressure": "135/82",
                    "heart_rate": "72",
                    "temperature": "98.6",
                    "respiratory_rate": "16",
                    "oxygen_saturation": "98%"
                }
            )
            session.add(initial_note)
            
            # MRI Study
            mri_study = ImagingStudy(
                patient_id=patient.patient_id,
                ordering_provider_id=uuid.UUID(self.providers["attending_physician"]),
                study_type=ImagingType.MRI,
                body_part="Right Shoulder",
                study_date=datetime.now() - timedelta(days=170),
                clinical_indication="Right shoulder pain and limited range of motion",
                findings="Severe glenohumeral joint space narrowing with osteophytes. Full-thickness rotator cuff intact. No fractures or loose bodies identified.",
                impression="Severe glenohumeral osteoarthritis. Rotator cuff intact.",
                modality="MRI",
                accession_number="MRI20240001"
            )
            session.add(mri_study)
            
            # Physical Therapy Trial (3 months)
            for i in range(12):  # 12 PT sessions
                pt_session = PhysicalTherapy(
                    patient_id=patient.patient_id,
                    therapist_id=uuid.UUID(self.providers["attending_physician"]),  # Mock therapist
                    session_date=date.today() - timedelta(days=150-i*7),
                    session_number=i+1,
                    total_sessions_planned=12,
                    exercises_performed=[
                        "Pendulum exercises",
                        "Passive ROM",
                        "Isometric strengthening",
                        "Stretching program"
                    ],
                    range_of_motion={
                        "forward_flexion": 90 + i*5,  # Gradual improvement
                        "abduction": 80 + i*3,
                        "external_rotation": 30 + i*2
                    },
                    pain_level_before=8-i//3,  # Gradual pain reduction
                    pain_level_after=7-i//3,
                    session_notes=f"Session {i+1}: Continuing ROM and strengthening exercises. Patient shows minimal progress.",
                    patient_compliance="Good"
                )
                session.add(pt_session)
            
            # Surgery Consultation - after PT failure
            surgery_consult = Appointment(
                patient_id=patient.patient_id,
                provider_id=uuid.UUID(self.providers["attending_physician"]),
                appointment_type=AppointmentType.CONSULTATION,
                appointment_date=date.today() - timedelta(days=60),
                appointment_time=datetime.strptime("14:00", "%H:%M").time(),
                duration_minutes=45,
                status=AppointmentStatus.COMPLETED,
                chief_complaint="Follow-up for right shoulder arthritis, failed conservative treatment",
                notes="Patient continues with significant pain and functional limitation despite PT"
            )
            session.add(surgery_consult)
            await session.flush()
            
            # Surgery Consultation Note
            surgery_note = MedicalNote(
                patient_id=patient.patient_id,
                provider_id=uuid.UUID(self.providers["attending_physician"]),
                appointment_id=surgery_consult.appointment_id,
                note_type="Surgical Consultation",
                chief_complaint="Persistent right shoulder pain after failed conservative treatment",
                history_present_illness="Patient completed 3 months of physical therapy with minimal improvement. Pain remains 7-8/10, significantly impacting quality of life.",
                physical_exam="ROM minimally improved. Persistent pain with passive and active motion. Patient candidate for total shoulder arthroplasty.",
                assessment="Severe glenohumeral osteoarthritis, failed conservative treatment",
                plan="1. Total shoulder arthroplasty scheduled 2. Pre-operative clearance 3. Discuss post-operative expectations and rehabilitation"
            )
            session.add(surgery_note)
            
            # Total Shoulder Arthroplasty Procedure
            surgery = Procedure(
                patient_id=patient.patient_id,
                provider_id=uuid.UUID(self.providers["attending_physician"]),
                procedure_name="Total Shoulder Arthroplasty",
                procedure_code="23472",
                icd10_diagnosis_codes=["M19.011"],  # Primary osteoarthritis of right shoulder
                scheduled_date=datetime.now() - timedelta(days=30),
                completed_date=datetime.now() - timedelta(days=30),
                duration_minutes=120,
                pre_op_diagnosis="Severe right glenohumeral osteoarthritis",
                post_op_diagnosis="Severe right glenohumeral osteoarthritis, status post total shoulder arthroplasty",
                procedure_notes="Uncomplicated total shoulder arthroplasty with cemented humeral component and glenoid resurfacing. Good range of motion achieved intraoperatively.",
                anesthesia_type="General anesthesia with regional block",
                status=ProcedureStatus.COMPLETED
            )
            session.add(surgery)
            await session.flush()
            
            # Post-op appointments and billing
            postop_appt = Appointment(
                patient_id=patient.patient_id,
                provider_id=uuid.UUID(self.providers["attending_physician"]),
                appointment_type=AppointmentType.FOLLOW_UP,
                appointment_date=date.today() - timedelta(days=16),
                appointment_time=datetime.strptime("11:00", "%H:%M").time(),
                duration_minutes=30,
                status=AppointmentStatus.COMPLETED,
                chief_complaint="Post-operative follow-up",
                notes="2-week post-operative check. Wound healing well, patient doing well."
            )
            session.add(postop_appt)
            
            # Billing Records
            surgery_billing = BillingRecord(
                patient_id=patient.patient_id,
                procedure_id=surgery.procedure_id,
                service_date=surgery.completed_date.date(),
                cpt_code="23472",
                description="Total shoulder arthroplasty",
                charge_amount=Decimal("45000.00"),
                allowed_amount=Decimal("38000.00"),
                primary_insurance_paid=Decimal("30400.00"),  # 80% Medicare
                secondary_insurance_paid=Decimal("7600.00"),  # 20% Medigap
                patient_responsibility=Decimal("0.00"),
                status=BillingStatus.PAID,
                date_submitted=surgery.completed_date.date() + timedelta(days=1),
                date_paid=surgery.completed_date.date() + timedelta(days=30)
            )
            session.add(surgery_billing)
            
            await session.commit()
            return patient

    async def create_knee_replacement_patient(self):
        """Patient 2: Robert Martinez - Total Knee Replacement"""
        print("👨‍🔧 Creating Patient 2: Knee Replacement Case...")
        
        async with db_manager.get_session() as session:
            patient = Patient(
                medical_record_number="MRN002345",
                first_name="Robert",
                last_name="Martinez",
                middle_name="Carlos",
                date_of_birth=date(1963, 8, 22),
                gender=Gender.MALE,
                phone_primary="555-0234",
                email="r.martinez@email.com",
                address_line1="456 Pine Avenue",
                city="Oakland",
                state="CA", 
                zip_code="94601",
                primary_care_physician="Dr. Lisa Chen",
                allergies="No known drug allergies",
                current_medications=[
                    {"name": "Ibuprofen", "dosage": "600mg", "frequency": "three times daily"},
                    {"name": "Glucosamine", "dosage": "1500mg", "frequency": "daily"}
                ],
                medical_history="Osteoarthritis bilateral knees, Former construction worker",
                family_history="Father: Arthritis, Mother: Hypertension",
                social_history="Retired construction worker, former smoker (quit 10 years ago)",
                emergency_contact_name="Maria Martinez",
                emergency_contact_phone="555-0235",
                emergency_contact_relationship="Wife"
            )
            session.add(patient)
            await session.flush()
            
            # Workers' Compensation Insurance
            insurance = PatientInsurance(
                patient_id=patient.patient_id,
                insurance_type=InsuranceType.WORKERS_COMP,
                insurance_company="State Compensation Insurance Fund",
                policy_number="WC789456",
                subscriber_name="Robert Carlos Martinez",
                subscriber_id="WC789456",
                is_primary=True,
                effective_date=date(2020, 1, 1),
                copay_amount=Decimal("0.00"),
                deductible_amount=Decimal("0.00")
            )
            session.add(insurance)
            
            # Progressive care timeline for knee replacement
            # Similar detailed timeline as shoulder replacement...
            # [Abbreviated for space - would include full consultation, imaging, PT, surgery, billing]
            
            surgery = Procedure(
                patient_id=patient.patient_id,
                provider_id=uuid.UUID(self.providers["attending_physician"]),
                procedure_name="Total Knee Arthroplasty",
                procedure_code="27447",
                icd10_diagnosis_codes=["M17.11"],  # Unilateral primary osteoarthritis, right knee
                scheduled_date=datetime.now() - timedelta(days=45),
                completed_date=datetime.now() - timedelta(days=45),
                duration_minutes=90,
                pre_op_diagnosis="Severe right knee osteoarthritis",
                post_op_diagnosis="Severe right knee osteoarthritis, status post total knee arthroplasty",
                procedure_notes="Uncomplicated total knee arthroplasty with cemented components. Excellent alignment achieved.",
                anesthesia_type="Spinal anesthesia",
                status=ProcedureStatus.COMPLETED
            )
            session.add(surgery)
            await session.flush()
            
            surgery_billing = BillingRecord(
                patient_id=patient.patient_id,
                procedure_id=surgery.procedure_id,
                service_date=surgery.completed_date.date(),
                cpt_code="27447",
                description="Total knee arthroplasty",
                charge_amount=Decimal("42000.00"),
                allowed_amount=Decimal("42000.00"),
                primary_insurance_paid=Decimal("42000.00"),  # Workers comp pays 100%
                patient_responsibility=Decimal("0.00"),
                status=BillingStatus.PAID
            )
            session.add(surgery_billing)
            
            await session.commit()
            return patient

    async def create_acl_repair_patient(self):
        """Patient 3: Jessica Chen - ACL Reconstruction (Young Athlete)"""
        print("🏃‍♀️ Creating Patient 3: ACL Reconstruction Case...")
        
        async with db_manager.get_session() as session:
            patient = Patient(
                medical_record_number="MRN003456",
                first_name="Jessica",
                last_name="Chen",
                date_of_birth=date(2001, 11, 8),
                gender=Gender.FEMALE,
                phone_primary="555-0345",
                email="jessica.chen@university.edu",
                address_line1="789 College Way",
                city="Berkeley",
                state="CA",
                zip_code="94720",
                primary_care_physician="Dr. Student Health Center",
                allergies="Seasonal allergies (pollen)",
                current_medications=[
                    {"name": "Claritin", "dosage": "10mg", "frequency": "daily as needed"}
                ],
                medical_history="No significant past medical history",
                family_history="Non-contributory",
                social_history="College student, soccer player, non-smoker, social alcohol use",
                emergency_contact_name="Linda Chen",
                emergency_contact_phone="555-0346",
                emergency_contact_relationship="Mother"
            )
            session.add(patient)
            await session.flush()
            
            # University Health Insurance
            insurance = PatientInsurance(
                patient_id=patient.patient_id,
                insurance_type=InsuranceType.PRIVATE,
                insurance_company="University Health Plan",
                policy_number="STUDENT123456",
                subscriber_name="Jessica Chen",
                subscriber_id="STUDENT123456",
                is_primary=True,
                effective_date=date(2024, 8, 1),
                expiration_date=date(2025, 7, 31),
                copay_amount=Decimal("20.00"),
                deductible_amount=Decimal("500.00")
            )
            session.add(insurance)
            
            # Emergency visit for injury
            er_visit = Appointment(
                patient_id=patient.patient_id,
                provider_id=uuid.UUID(self.providers["attending_physician"]),
                appointment_type=AppointmentType.CONSULTATION,
                appointment_date=date.today() - timedelta(days=90),
                appointment_time=datetime.strptime("19:30", "%H:%M").time(),
                duration_minutes=120,
                status=AppointmentStatus.COMPLETED,
                chief_complaint="Left knee injury during soccer practice",
                notes="Pop heard during cutting motion, immediate pain and swelling"
            )
            session.add(er_visit)
            await session.flush()
            
            # ACL Reconstruction
            acl_surgery = Procedure(
                patient_id=patient.patient_id,
                provider_id=uuid.UUID(self.providers["attending_physician"]),
                procedure_name="ACL Reconstruction with Hamstring Autograft",
                procedure_code="29888",
                icd10_diagnosis_codes=["S83.511A"],  # Sprain of anterior cruciate ligament
                scheduled_date=datetime.now() - timedelta(days=60),
                completed_date=datetime.now() - timedelta(days=60),
                duration_minutes=75,
                pre_op_diagnosis="Complete ACL tear left knee",
                post_op_diagnosis="Complete ACL tear left knee, status post ACL reconstruction",
                procedure_notes="Arthroscopic ACL reconstruction using hamstring autograft. Good tunnel placement and graft tension.",
                anesthesia_type="General anesthesia",
                status=ProcedureStatus.COMPLETED
            )
            session.add(acl_surgery)
            await session.flush()
            
            # Extensive PT program for young athlete (ongoing)
            for i in range(20):  # 20 PT sessions over 5 months
                pt_session = PhysicalTherapy(
                    patient_id=patient.patient_id,
                    therapist_id=uuid.UUID(self.providers["attending_physician"]),
                    session_date=date.today() - timedelta(days=50-i*7),
                    session_number=i+1,
                    total_sessions_planned=30,
                    exercises_performed=[
                        "Quad strengthening",
                        "Hamstring strengthening", 
                        "Balance training",
                        "Plyometric exercises" if i > 10 else "Range of motion"
                    ],
                    pain_level_before=max(1, 6-i//3),
                    pain_level_after=max(0, 5-i//3),
                    session_notes=f"PT session {i+1}: Progressive strengthening program. Goal return to soccer.",
                    patient_compliance="Excellent"
                )
                session.add(pt_session)
            
            await session.commit()
            return patient

    async def create_compound_fracture_patient(self):
        """Patient 4: David Wilson - Compound Tibial Fracture with Insurance"""
        print("🚨 Creating Patient 4: Compound Fracture Case...")
        
        async with db_manager.get_session() as session:
            patient = Patient(
                medical_record_number="MRN004567",
                first_name="David",
                last_name="Wilson",
                date_of_birth=date(1978, 5, 14),
                gender=Gender.MALE,
                phone_primary="555-0456",
                email="david.wilson@email.com",
                address_line1="321 Main Street",
                city="San Jose",
                state="CA",
                zip_code="95112",
                primary_care_physician="Dr. Emergency Medicine",
                allergies="Morphine (nausea)",
                current_medications=[],
                medical_history="No significant past medical history",
                family_history="Non-contributory",
                social_history="Software engineer, motorcycle accident",
                emergency_contact_name="Sarah Wilson",
                emergency_contact_phone="555-0457",
                emergency_contact_relationship="Wife"
            )
            session.add(patient)
            await session.flush()
            
            # Auto Insurance Coverage
            insurance = PatientInsurance(
                patient_id=patient.patient_id,
                insurance_type=InsuranceType.AUTO_INSURANCE,
                insurance_company="AllState Auto Insurance",
                policy_number="AUTO987654",
                subscriber_name="David Wilson",
                subscriber_id="AUTO987654",
                is_primary=True,
                effective_date=date(2023, 1, 1),
                copay_amount=Decimal("0.00"),
                deductible_amount=Decimal("1000.00")
            )
            session.add(insurance)
            
            # Emergency surgery for compound fracture
            emergency_surgery = Procedure(
                patient_id=patient.patient_id,
                provider_id=uuid.UUID(self.providers["attending_physician"]),
                procedure_name="Open Reduction Internal Fixation Tibia",
                procedure_code="27758",
                icd10_diagnosis_codes=["S82.251B"],  # Displaced comminuted fracture shaft of tibia
                scheduled_date=datetime.now() - timedelta(days=120),
                completed_date=datetime.now() - timedelta(days=120),
                duration_minutes=180,
                pre_op_diagnosis="Open comminuted tibial shaft fracture",
                post_op_diagnosis="Open comminuted tibial shaft fracture, status post ORIF",
                procedure_notes="Irrigation and debridement, ORIF with intramedullary nail fixation. Good alignment achieved.",
                anesthesia_type="General anesthesia",
                status=ProcedureStatus.COMPLETED
            )
            session.add(emergency_surgery)
            await session.flush()
            
            # Multiple follow-up appointments for fracture healing
            for i, days_ago in enumerate([90, 60, 30, 14]):
                followup = Appointment(
                    patient_id=patient.patient_id,
                    provider_id=uuid.UUID(self.providers["attending_physician"]),
                    appointment_type=AppointmentType.FOLLOW_UP,
                    appointment_date=date.today() - timedelta(days=days_ago),
                    appointment_time=datetime.strptime("14:00", "%H:%M").time(),
                    duration_minutes=20,
                    status=AppointmentStatus.COMPLETED,
                    chief_complaint=f"Follow-up for tibial fracture healing - {4-i} weeks post-op",
                    notes=f"Fracture healing progressing appropriately at {4-i} weeks"
                )
                session.add(followup)
            
            await session.commit()
            return patient

    async def create_self_pay_fracture_patient(self):
        """Patient 5: Maria Rodriguez - Self-Pay Wrist Fracture"""
        print("💰 Creating Patient 5: Self-Pay Fracture Case...")
        
        async with db_manager.get_session() as session:
            patient = Patient(
                medical_record_number="MRN005678",
                first_name="Maria",
                last_name="Rodriguez",
                date_of_birth=date(1985, 12, 3),
                gender=Gender.FEMALE,
                phone_primary="555-0567",
                email="maria.rodriguez@email.com",
                address_line1="654 Garden Street",
                city="San Francisco",
                state="CA",
                zip_code="94110",
                primary_care_physician="None",
                allergies="No known allergies",
                current_medications=[],
                medical_history="No significant past medical history",
                family_history="Non-contributory",
                social_history="Restaurant worker, uninsured, slip and fall injury",
                emergency_contact_name="Carlos Rodriguez",
                emergency_contact_phone="555-0568",
                emergency_contact_relationship="Brother"
            )
            session.add(patient)
            await session.flush()
            
            # Self-Pay "Insurance"
            self_pay = PatientInsurance(
                patient_id=patient.patient_id,
                insurance_type=InsuranceType.SELF_PAY,
                insurance_company="Self Pay",
                policy_number="SELFPAY001",
                subscriber_name="Maria Rodriguez",
                subscriber_id="SELFPAY001",
                is_primary=True,
                effective_date=date.today() - timedelta(days=60),
                copay_amount=Decimal("0.00"),
                deductible_amount=Decimal("0.00")
            )
            session.add(self_pay)
            
            # Simple fracture procedure
            wrist_procedure = Procedure(
                patient_id=patient.patient_id,
                provider_id=uuid.UUID(self.providers["attending_physician"]),
                procedure_name="Closed Reduction and Casting - Distal Radius Fracture",
                procedure_code="25600",
                icd10_diagnosis_codes=["S52.501A"],  # Unspecified fracture of lower end of radius
                scheduled_date=datetime.now() - timedelta(days=45),
                completed_date=datetime.now() - timedelta(days=45),
                duration_minutes=30,
                pre_op_diagnosis="Distal radius fracture",
                post_op_diagnosis="Distal radius fracture, status post closed reduction",
                procedure_notes="Closed reduction of distal radius fracture with satisfactory alignment. Long arm cast applied.",
                anesthesia_type="Local anesthesia",
                status=ProcedureStatus.COMPLETED
            )
            session.add(wrist_procedure)
            await session.flush()
            
            # Self-pay billing with payment plan
            procedure_billing = BillingRecord(
                patient_id=patient.patient_id,
                procedure_id=wrist_procedure.procedure_id,
                service_date=wrist_procedure.completed_date.date(),
                cpt_code="25600",
                description="Closed reduction distal radius fracture with casting",
                charge_amount=Decimal("1500.00"),
                allowed_amount=Decimal("1200.00"),  # Self-pay discount
                paid_amount=Decimal("400.00"),  # Partial payment
                patient_responsibility=Decimal("800.00"),  # Outstanding balance
                status=BillingStatus.PENDING,
                date_submitted=wrist_procedure.completed_date.date()
            )
            session.add(procedure_billing)
            
            # Cast removal appointment
            cast_removal = Appointment(
                patient_id=patient.patient_id,
                provider_id=uuid.UUID(self.providers["attending_physician"]),
                appointment_type=AppointmentType.FOLLOW_UP,
                appointment_date=date.today() + timedelta(days=7),  # Future appointment
                appointment_time=datetime.strptime("10:00", "%H:%M").time(),
                duration_minutes=15,
                status=AppointmentStatus.SCHEDULED,
                chief_complaint="Cast removal and follow-up X-rays",
                notes="6-week follow-up for cast removal"
            )
            session.add(cast_removal)
            
            await session.commit()
            return patient

async def main():
    generator = PatientDataGenerator()
    await generator.generate_all_patients()

if __name__ == "__main__":
    asyncio.run(main())