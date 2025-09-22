"""
Vector embedding service for medical text using simple TF-IDF approach.
HIPAA-compliant local embedding generation for semantic search.
"""

import logging
from typing import List, Optional, Dict, Any
import asyncio
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re
import string

logger = logging.getLogger(__name__)

class MedicalEmbeddingService:
    """
    Medical text embedding service using TF-IDF with SVD dimensionality reduction.
    Designed for HIPAA-compliant local processing.
    """
    
    def __init__(self, embedding_dimension: int = 384):
        """
        Initialize the embedding service.
        
        Args:
            embedding_dimension: Target dimension for embeddings
        """
        self.embedding_dimension = embedding_dimension
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.is_fitted = False
        self.training_corpus = []  # Store corpus for fitting
        
        # Medical stopwords and common abbreviations
        self.medical_stopwords = {
            'patient', 'history', 'noted', 'report', 'findings', 'impression',
            'clinical', 'examination', 'assessment', 'plan', 'cc', 'hpi',
            'ros', 'pe', 'ap', 'the', 'and', 'or', 'in', 'on', 'at', 'to',
            'is', 'was', 'will', 'has', 'have', 'had', 'with', 'without'
        }
        
    async def initialize(self):
        """Initialize the embedding model."""
        try:
            logger.info("Initializing TF-IDF embedding service...")
            
            # Initialize TF-IDF vectorizer with medical-specific settings
            self.vectorizer = TfidfVectorizer(
                max_features=1000,  # Smaller vocabulary for faster processing
                stop_words=list(self.medical_stopwords),
                ngram_range=(1, 1),  # Only unigrams for small datasets
                min_df=1,  # Include all terms for small datasets
                max_df=1.0,  # Include all terms (no filtering by frequency)
                lowercase=True,
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only alphabetic tokens
            )
            
            # Initialize SVD for dimensionality reduction
            target_components = min(self.embedding_dimension, 384)  # Cap at 384
            self.svd = TruncatedSVD(
                n_components=target_components,
                random_state=42
            )
            
            logger.info(f"TF-IDF embedding service initialized. Target dimension: {target_components}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise
    
    def _clean_medical_text(self, text: str) -> str:
        """Clean and normalize medical text."""
        if not text:
            return ""
            
        # Basic cleaning
        text = text.strip().lower()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation except periods (for sentence structure)
        text = re.sub(r'[^\w\s\.]', ' ', text)
        
        return text
    
    async def fit_corpus(self, texts: List[str]):
        """Fit the vectorizer and SVD on a corpus of texts."""
        if not texts:
            logger.warning("No texts provided for fitting corpus")
            return
            
        try:
            # Clean texts
            cleaned_texts = [self._clean_medical_text(text) for text in texts if text and text.strip()]
            
            if not cleaned_texts:
                logger.warning("No valid texts after cleaning")
                return
                
            logger.info(f"Fitting TF-IDF on {len(cleaned_texts)} documents...")
            
            # Fit TF-IDF vectorizer
            tfidf_matrix = self.vectorizer.fit_transform(cleaned_texts)
            
            # Fit SVD if we have enough components
            if tfidf_matrix.shape[1] >= self.svd.n_components:
                self.svd.fit(tfidf_matrix)
                self.is_fitted = True
                logger.info(f"Fitted SVD with {self.svd.n_components} components")
            else:
                logger.warning(f"Insufficient features ({tfidf_matrix.shape[1]}) for SVD ({self.svd.n_components} components)")
                # Reduce SVD components to match available features
                self.svd.n_components = min(tfidf_matrix.shape[1], 50)
                self.svd.fit(tfidf_matrix)
                self.is_fitted = True
                
        except Exception as e:
            logger.error(f"Failed to fit corpus: {e}")
            self.is_fitted = False
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.svd.n_components if self.is_fitted else [0.0] * self.embedding_dimension
            
        try:
            # Clean text
            cleaned_text = self._clean_medical_text(text)
            
            if not self.is_fitted:
                # If not fitted, try to fit on this single text
                await self.fit_corpus([cleaned_text])
                
            if not self.is_fitted:
                # Still not fitted, return zero vector
                return [0.0] * self.embedding_dimension
            
            # Transform text to TF-IDF
            tfidf_vector = self.vectorizer.transform([cleaned_text])
            
            # Apply SVD transformation
            embedding = self.svd.transform(tfidf_vector)
            
            # Pad or truncate to target dimension
            result = embedding[0].tolist()
            
            # Ensure we have the right dimension
            if len(result) < self.embedding_dimension:
                result.extend([0.0] * (self.embedding_dimension - len(result)))
            elif len(result) > self.embedding_dimension:
                result = result[:self.embedding_dimension]
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            # Return zero vector on error
            return [0.0] * self.embedding_dimension
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        if not texts:
            return []
            
        try:
            # Clean texts
            cleaned_texts = [self._clean_medical_text(text) for text in texts]
            
            # Fit if not already fitted
            if not self.is_fitted:
                await self.fit_corpus(cleaned_texts)
                
            if not self.is_fitted:
                # Return zero vectors if still not fitted
                return [[0.0] * self.embedding_dimension] * len(texts)
            
            # Transform all texts
            tfidf_matrix = self.vectorizer.transform(cleaned_texts)
            embeddings = self.svd.transform(tfidf_matrix)
            
            # Convert to list format and ensure correct dimensions
            results = []
            for embedding in embeddings:
                result = embedding.tolist()
                
                # Ensure we have the right dimension
                if len(result) < self.embedding_dimension:
                    result.extend([0.0] * (self.embedding_dimension - len(result)))
                elif len(result) > self.embedding_dimension:
                    result = result[:self.embedding_dimension]
                    
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            # Return zero vectors on error
            return [[0.0] * self.embedding_dimension] * len(texts)
    
    async def combine_medical_text_fields(self, **fields) -> str:
        """Combine multiple medical text fields into a single searchable text."""
        combined_parts = []
        
        # Define field priorities and labels
        field_labels = {
            'chief_complaint': 'Chief Complaint',
            'history_present_illness': 'History of Present Illness',
            'assessment': 'Assessment',
            'plan': 'Plan',
            'physical_exam': 'Physical Exam',
            'review_of_systems': 'Review of Systems',
            'procedure_notes': 'Procedure Notes',
            'diagnosis': 'Diagnosis',
            'clinical_indication': 'Clinical Indication',
            'findings': 'Findings',
            'impression': 'Impression',
            'session_notes': 'Session Notes',
            'allergies': 'Allergies',
            'medical_history': 'Medical History'
        }
        
        # Combine fields in logical order
        for field_name, field_value in fields.items():
            if field_value and field_value.strip():
                label = field_labels.get(field_name, field_name.replace('_', ' ').title())
                combined_parts.append(f"{label}: {field_value.strip()}")
        
        return ". ".join(combined_parts)
    
    async def search_similarity(self, query_embedding: List[float], candidate_embeddings: List[List[float]], top_k: int = 10) -> List[tuple]:
        """Find most similar embeddings to a query."""
        if not candidate_embeddings:
            return []
            
        try:
            # Convert to numpy arrays for efficient computation
            query_vec = np.array(query_embedding)
            candidate_vecs = np.array(candidate_embeddings)
            
            # Compute cosine similarity
            # Normalize vectors
            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                return []
                
            query_normalized = query_vec / query_norm
            
            candidate_norms = np.linalg.norm(candidate_vecs, axis=1)
            # Avoid division by zero
            non_zero_mask = candidate_norms > 0
            similarities = np.zeros(len(candidate_vecs))
            
            if np.any(non_zero_mask):
                candidate_normalized = candidate_vecs[non_zero_mask] / candidate_norms[non_zero_mask].reshape(-1, 1)
                similarities[non_zero_mask] = np.dot(candidate_normalized, query_normalized)
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Return results with scores
            results = [(int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0.01]
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to compute similarity search: {e}")
            return []

# Global embedding service instance
embedding_service = MedicalEmbeddingService()

# Convenience functions for common operations
async def embed_medical_note(note_data: Dict[str, Any]) -> List[float]:
    """Generate embedding for a medical note."""
    combined_text = await embedding_service.combine_medical_text_fields(
        chief_complaint=note_data.get('chief_complaint'),
        history_present_illness=note_data.get('history_present_illness'),
        assessment=note_data.get('assessment'),
        plan=note_data.get('plan'),
        physical_exam=note_data.get('physical_exam'),
        review_of_systems=note_data.get('review_of_systems')
    )
    return await embedding_service.embed_text(combined_text)

async def embed_procedure_note(procedure_data: Dict[str, Any]) -> List[float]:
    """Generate embedding for a procedure note."""
    combined_text = await embedding_service.combine_medical_text_fields(
        procedure_name=procedure_data.get('procedure_name'),
        pre_op_diagnosis=procedure_data.get('pre_op_diagnosis'),
        post_op_diagnosis=procedure_data.get('post_op_diagnosis'),
        procedure_notes=procedure_data.get('procedure_notes'),
        complications=procedure_data.get('complications')
    )
    return await embedding_service.embed_text(combined_text)

async def embed_imaging_study(imaging_data: Dict[str, Any]) -> List[float]:
    """Generate embedding for an imaging study."""
    combined_text = await embedding_service.combine_medical_text_fields(
        clinical_indication=imaging_data.get('clinical_indication'),
        findings=imaging_data.get('findings'),
        impression=imaging_data.get('impression'),
        study_type=imaging_data.get('study_type'),
        body_part=imaging_data.get('body_part')
    )
    return await embedding_service.embed_text(combined_text)

async def embed_patient_summary(patient_data: Dict[str, Any]) -> List[float]:
    """Generate embedding for patient summary."""
    combined_text = await embedding_service.combine_medical_text_fields(
        allergies=patient_data.get('allergies'),
        medical_history=patient_data.get('medical_history'),
        family_history=patient_data.get('family_history'),
        social_history=patient_data.get('social_history')
    )
    return await embedding_service.embed_text(combined_text)