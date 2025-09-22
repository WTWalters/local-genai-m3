"""
Generate vector embeddings for existing medical documents and patient data.
This script processes all medical text and creates semantic embeddings for search.
"""

import asyncio
import logging
from typing import List, Dict, Any
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import db_manager
from database.patient_models import Patient, MedicalNote, Procedure, ImagingStudy
from core.embeddings import (
    embedding_service, 
    embed_medical_note, 
    embed_procedure_note, 
    embed_imaging_study, 
    embed_patient_summary
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def generate_patient_embeddings(session: AsyncSession) -> int:
    """Generate embeddings for patient summaries."""
    
    logger.info("🔍 Processing patient summary embeddings...")
    
    # Get all patients without embeddings
    result = await session.execute(
        select(Patient).where(Patient.summary_embedding.is_(None))
    )
    patients = result.scalars().all()
    
    if not patients:
        logger.info("No patients need embedding generation")
        return 0
    
    logger.info(f"Found {len(patients)} patients to process")
    
    processed = 0
    for patient in patients:
        try:
            # Prepare patient data for embedding
            patient_data = {
                'allergies': patient.allergies,
                'medical_history': patient.medical_history,
                'family_history': patient.family_history,
                'social_history': patient.social_history
            }
            
            # Generate embedding
            embedding = await embed_patient_summary(patient_data)
            
            if embedding and any(embedding):  # Check if embedding is not all zeros
                # Update patient with embedding
                await session.execute(
                    update(Patient)
                    .where(Patient.patient_id == patient.patient_id)
                    .values(summary_embedding=embedding)
                )
                processed += 1
                
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{len(patients)} patients")
                    
        except Exception as e:
            logger.error(f"Failed to process patient {patient.patient_id}: {e}")
    
    await session.commit()
    logger.info(f"✅ Generated embeddings for {processed} patient summaries")
    return processed

async def generate_medical_note_embeddings(session: AsyncSession) -> int:
    """Generate embeddings for medical notes."""
    
    logger.info("🔍 Processing medical note embeddings...")
    
    # Get all medical notes without embeddings
    result = await session.execute(
        select(MedicalNote).where(MedicalNote.content_embedding.is_(None))
    )
    notes = result.scalars().all()
    
    if not notes:
        logger.info("No medical notes need embedding generation")
        return 0
    
    logger.info(f"Found {len(notes)} medical notes to process")
    
    processed = 0
    for note in notes:
        try:
            # Prepare note data for embedding
            note_data = {
                'chief_complaint': note.chief_complaint,
                'history_present_illness': note.history_present_illness,
                'assessment': note.assessment,
                'plan': note.plan,
                'physical_exam': note.physical_exam,
                'review_of_systems': note.review_of_systems
            }
            
            # Generate embedding
            embedding = await embed_medical_note(note_data)
            
            if embedding and any(embedding):  # Check if embedding is not all zeros
                # Update note with embedding
                await session.execute(
                    update(MedicalNote)
                    .where(MedicalNote.note_id == note.note_id)
                    .values(content_embedding=embedding)
                )
                processed += 1
                
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{len(notes)} notes")
                    
        except Exception as e:
            logger.error(f"Failed to process note {note.note_id}: {e}")
    
    await session.commit()
    logger.info(f"✅ Generated embeddings for {processed} medical notes")
    return processed

async def generate_procedure_embeddings(session: AsyncSession) -> int:
    """Generate embeddings for procedures."""
    
    logger.info("🔍 Processing procedure embeddings...")
    
    # Get all procedures without embeddings
    result = await session.execute(
        select(Procedure).where(Procedure.content_embedding.is_(None))
    )
    procedures = result.scalars().all()
    
    if not procedures:
        logger.info("No procedures need embedding generation")
        return 0
    
    logger.info(f"Found {len(procedures)} procedures to process")
    
    processed = 0
    for procedure in procedures:
        try:
            # Prepare procedure data for embedding
            procedure_data = {
                'procedure_name': procedure.procedure_name,
                'pre_op_diagnosis': procedure.pre_op_diagnosis,
                'post_op_diagnosis': procedure.post_op_diagnosis,
                'procedure_notes': procedure.procedure_notes,
                'complications': procedure.complications
            }
            
            # Generate embedding
            embedding = await embed_procedure_note(procedure_data)
            
            if embedding and any(embedding):  # Check if embedding is not all zeros
                # Update procedure with embedding
                await session.execute(
                    update(Procedure)
                    .where(Procedure.procedure_id == procedure.procedure_id)
                    .values(content_embedding=embedding)
                )
                processed += 1
                
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{len(procedures)} procedures")
                    
        except Exception as e:
            logger.error(f"Failed to process procedure {procedure.procedure_id}: {e}")
    
    await session.commit()
    logger.info(f"✅ Generated embeddings for {processed} procedures")
    return processed

async def generate_imaging_study_embeddings(session: AsyncSession) -> int:
    """Generate embeddings for imaging studies."""
    
    logger.info("🔍 Processing imaging study embeddings...")
    
    # Get all imaging studies without embeddings
    result = await session.execute(
        select(ImagingStudy).where(ImagingStudy.content_embedding.is_(None))
    )
    studies = result.scalars().all()
    
    if not studies:
        logger.info("No imaging studies need embedding generation")
        return 0
    
    logger.info(f"Found {len(studies)} imaging studies to process")
    
    processed = 0
    for study in studies:
        try:
            # Prepare imaging data for embedding
            imaging_data = {
                'clinical_indication': study.clinical_indication,
                'findings': study.findings,
                'impression': study.impression,
                'study_type': study.study_type,
                'body_part': study.body_part
            }
            
            # Generate embedding
            embedding = await embed_imaging_study(imaging_data)
            
            if embedding and any(embedding):  # Check if embedding is not all zeros
                # Update study with embedding
                await session.execute(
                    update(ImagingStudy)
                    .where(ImagingStudy.study_id == study.study_id)
                    .values(content_embedding=embedding)
                )
                processed += 1
                
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{len(studies)} studies")
                    
        except Exception as e:
            logger.error(f"Failed to process study {study.study_id}: {e}")
    
    await session.commit()
    logger.info(f"✅ Generated embeddings for {processed} imaging studies")
    return processed

async def collect_all_texts(session: AsyncSession) -> List[str]:
    """Collect all medical texts for corpus fitting."""
    
    logger.info("🔍 Collecting all medical texts for corpus fitting...")
    all_texts = []
    
    # Collect patient texts
    result = await session.execute(select(Patient))
    patients = result.scalars().all()
    
    for patient in patients:
        patient_text = await embedding_service.combine_medical_text_fields(
            allergies=patient.allergies,
            medical_history=patient.medical_history,
            family_history=patient.family_history,
            social_history=patient.social_history
        )
        if patient_text.strip():
            all_texts.append(patient_text)
    
    # Collect medical note texts
    result = await session.execute(select(MedicalNote))
    notes = result.scalars().all()
    
    for note in notes:
        note_text = await embedding_service.combine_medical_text_fields(
            chief_complaint=note.chief_complaint,
            history_present_illness=note.history_present_illness,
            assessment=note.assessment,
            plan=note.plan,
            physical_exam=note.physical_exam,
            review_of_systems=note.review_of_systems
        )
        if note_text.strip():
            all_texts.append(note_text)
    
    # Collect procedure texts
    result = await session.execute(select(Procedure))
    procedures = result.scalars().all()
    
    for procedure in procedures:
        procedure_text = await embedding_service.combine_medical_text_fields(
            procedure_name=procedure.procedure_name,
            pre_op_diagnosis=procedure.pre_op_diagnosis,
            post_op_diagnosis=procedure.post_op_diagnosis,
            procedure_notes=procedure.procedure_notes,
            complications=procedure.complications
        )
        if procedure_text.strip():
            all_texts.append(procedure_text)
    
    # Collect imaging study texts
    result = await session.execute(select(ImagingStudy))
    studies = result.scalars().all()
    
    for study in studies:
        imaging_text = await embedding_service.combine_medical_text_fields(
            clinical_indication=study.clinical_indication,
            findings=study.findings,
            impression=study.impression,
            study_type=study.study_type,
            body_part=study.body_part
        )
        if imaging_text.strip():
            all_texts.append(imaging_text)
    
    logger.info(f"Collected {len(all_texts)} medical texts for corpus")
    return all_texts

async def main():
    """Main function to generate all embeddings."""
    
    logger.info("🚀 Starting medical document embedding generation...")
    
    try:
        # Initialize embedding service
        await embedding_service.initialize()
        
        # Initialize database
        await db_manager.initialize()
        
        async with db_manager.get_session() as session:
            # Collect all texts and fit corpus once
            all_texts = await collect_all_texts(session)
            
            if all_texts:
                logger.info("🔧 Fitting embedding service on complete corpus...")
                await embedding_service.fit_corpus(all_texts)
            else:
                logger.warning("No medical texts found to process")
                return
        
        total_processed = 0
        
        async with db_manager.get_session() as session:
            # Generate patient summary embeddings
            patient_count = await generate_patient_embeddings(session)
            total_processed += patient_count
            
            # Generate medical note embeddings
            note_count = await generate_medical_note_embeddings(session)
            total_processed += note_count
            
            # Generate procedure embeddings
            procedure_count = await generate_procedure_embeddings(session)
            total_processed += procedure_count
            
            # Generate imaging study embeddings
            imaging_count = await generate_imaging_study_embeddings(session)
            total_processed += imaging_count
        
        logger.info(f"🎉 Embedding generation completed!")
        logger.info(f"📊 Summary:")
        logger.info(f"   - Patient summaries: {patient_count}")
        logger.info(f"   - Medical notes: {note_count}")
        logger.info(f"   - Procedures: {procedure_count}")
        logger.info(f"   - Imaging studies: {imaging_count}")
        logger.info(f"   - Total processed: {total_processed}")
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise
    
    finally:
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(main())