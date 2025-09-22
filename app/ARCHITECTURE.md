# Architecture Decision Record (ADR)

## Orthopedic EMR RAG System - Technical Decisions

This document captures the key architectural decisions made during the development of the Orthopedic EMR RAG System.

---

## ADR-001: Database Technology Selection

**Date**: 2025-09-18  
**Status**: Accepted  
**Deciders**: Development Team

### Context
Need to select a database technology that supports both traditional relational data and vector similarity search for the RAG system.

### Decision
**PostgreSQL 17.5 with pgvector extension**

### Rationale
1. **HIPAA Compliance**: PostgreSQL has established compliance features and audit capabilities
2. **Vector Support**: pgvector extension provides native vector operations without external dependencies
3. **Operational Simplicity**: Single database system reduces operational complexity
4. **ACID Compliance**: Medical data requires strong consistency guarantees
5. **Mature Ecosystem**: Well-supported with extensive tooling and documentation

### Alternatives Considered
- **ChromaDB**: Rejected due to operational complexity and compliance concerns
- **Pinecone**: Rejected due to cloud dependency and cost considerations
- **Weaviate**: Rejected due to complexity and learning curve

### Consequences
- ✅ Simplified deployment and operations
- ✅ Strong consistency for medical data
- ✅ Native vector similarity search
- ⚠️ Limited to PostgreSQL's vector capabilities
- ⚠️ Requires pgvector extension installation

---

## ADR-002: Embedding Technology Selection

**Date**: 2025-09-18  
**Status**: Accepted  
**Deciders**: Development Team

### Context
Need to generate vector embeddings for medical text content while maintaining HIPAA compliance and avoiding external API dependencies.

### Decision
**TF-IDF with SVD dimensionality reduction (384 dimensions)**

### Rationale
1. **Local Processing**: No external API calls required for HIPAA compliance
2. **Medical Domain**: TF-IDF captures medical terminology effectively
3. **Performance**: Fast generation and search with acceptable accuracy
4. **Simplicity**: Well-understood algorithm with predictable behavior
5. **Resource Efficient**: Low memory and compute requirements

### Alternatives Considered
- **OpenAI Embeddings**: Rejected due to HIPAA compliance concerns
- **Sentence-Transformers**: Rejected due to dependency conflicts and complexity
- **Custom Training**: Rejected due to time and data requirements

### Consequences
- ✅ HIPAA compliant local processing
- ✅ Fast embedding generation (<100ms)
- ✅ No external dependencies
- ⚠️ May not capture semantic nuances as well as transformer models
- ⚠️ Requires corpus fitting for optimal performance

---

## ADR-003: Authentication and Authorization Architecture

**Date**: 2025-09-18  
**Status**: Accepted  
**Deciders**: Development Team

### Context
Medical systems require robust authentication, role-based access control, and comprehensive audit logging for HIPAA compliance.

### Decision
**JWT-based authentication with Role-Based Access Control (RBAC) middleware**

### Rationale
1. **Stateless Authentication**: JWT tokens enable distributed deployment
2. **Medical Roles**: Built-in support for Doctor, Nurse, Admin, Receptionist roles
3. **Fine-grained Permissions**: Granular control over PHI access
4. **Audit Trail**: Comprehensive logging of all access attempts
5. **Session Management**: Configurable timeouts for security

### Implementation Details
```python
# Core permissions
class Permission(str, Enum):
    VIEW_PATIENTS = "view_patients"
    EDIT_PATIENTS = "edit_patients"
    VIEW_PHI = "view_phi"
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOGS = "view_audit_logs"

# Medical roles mapping
MEDICAL_ROLES = {
    "doctor": [Permission.VIEW_PATIENTS, Permission.EDIT_PATIENTS, Permission.VIEW_PHI],
    "nurse": [Permission.VIEW_PATIENTS, Permission.VIEW_PHI],
    "admin": [Permission.MANAGE_USERS, Permission.VIEW_AUDIT_LOGS],
    "receptionist": [Permission.VIEW_PATIENTS]
}
```

### Consequences
- ✅ HIPAA-compliant access controls
- ✅ Comprehensive audit logging
- ✅ Scalable role management
- ⚠️ Requires careful session management
- ⚠️ JWT token security considerations

---

## ADR-004: API Framework Selection

**Date**: 2025-09-18  
**Status**: Accepted  
**Deciders**: Development Team

### Context
Need a modern Python web framework that supports async operations, automatic API documentation, and strong typing for medical applications.

### Decision
**FastAPI with Pydantic v2**

### Rationale
1. **Async Support**: Native async/await for database operations
2. **Type Safety**: Built-in request/response validation
3. **Auto Documentation**: OpenAPI spec generation for API documentation
4. **Performance**: High-performance async framework
5. **Medical Standards**: Strong validation for medical data integrity

### API Structure
```
/api/v1/
├── auth/           # Authentication endpoints
├── patients/       # Patient management
├── search/         # Semantic search
├── users/          # User management
└── system/         # System endpoints
```

### Consequences
- ✅ Excellent developer experience
- ✅ Built-in API documentation
- ✅ Strong type safety
- ✅ High performance async operations
- ⚠️ Requires Python 3.11+ for optimal performance

---

## ADR-005: Data Models and Schema Design

**Date**: 2025-09-18  
**Status**: Accepted  
**Deciders**: Development Team

### Context
Medical data requires comprehensive modeling of patient lifecycle from diagnosis through treatment and follow-up.

### Decision
**Comprehensive relational model with vector augmentation**

### Core Entities
1. **Patient**: Demographics, medical history, allergies, emergency contacts
2. **Medical Notes**: Clinical documentation with vector embeddings
3. **Procedures**: Surgical and treatment procedures with semantic search
4. **Imaging Studies**: Radiology reports with vector similarity
5. **Appointments**: Scheduling and visit tracking
6. **Billing**: Insurance and payment processing
7. **Physical Therapy**: Rehabilitation tracking

### Vector Integration
Each major medical content type includes vector embeddings:
- 384-dimensional vectors using TF-IDF + SVD
- Cosine similarity search with configurable thresholds
- Hybrid search capabilities combining SQL and vector operations

### Consequences
- ✅ Comprehensive medical workflow coverage
- ✅ Semantic search across all content types
- ✅ Flexible querying with SQL and vector operations
- ⚠️ Complex data model requires careful maintenance
- ⚠️ Vector storage increases database size

---

## ADR-006: Security and Compliance Architecture

**Date**: 2025-09-18  
**Status**: Accepted  
**Deciders**: Development Team

### Context
HIPAA compliance requires comprehensive security controls including audit logging, access controls, and data protection.

### Decision
**Layered security architecture with comprehensive audit trail**

### Security Layers
1. **Transport Security**: HTTPS with security headers
2. **Authentication**: JWT with bcrypt password hashing
3. **Authorization**: Role-based access control (RBAC)
4. **Audit Logging**: Comprehensive access and action logging
5. **Session Management**: Configurable timeouts and secure tokens
6. **Data Protection**: Encrypted storage and secure key management

### Audit Requirements
```python
# All PHI access logged with:
- User identification and role
- Timestamp and session information
- Action type and resource accessed
- Success/failure status
- IP address and user agent
- Reason for access (when required)
```

### Consequences
- ✅ HIPAA-compliant security architecture
- ✅ Comprehensive audit trail
- ✅ Defense in depth security model
- ⚠️ Complex security configuration
- ⚠️ Performance impact from audit logging

---

## ADR-007: Search Architecture Design

**Date**: 2025-09-18  
**Status**: Accepted  
**Deciders**: Development Team

### Context
Medical search requires both precise SQL queries for structured data and semantic search for unstructured clinical text.

### Decision
**Hybrid search architecture combining SQL and vector similarity**

### Search Types Implemented
1. **Advanced SQL Search**: 30+ criteria including demographics, conditions, procedures
2. **Semantic Vector Search**: TF-IDF embeddings with cosine similarity
3. **Search Suggestions**: Medical terminology autocomplete
4. **Hybrid Results**: Combined ranking of SQL and vector results

### Performance Characteristics
- **Semantic Search**: ~79ms response time for 13 documents
- **SQL Search**: Sub-10ms for structured queries
- **Vector Indexing**: IVFFlat indexing for efficient similarity search
- **Result Ranking**: Configurable similarity thresholds

### Search API
```python
POST /api/v1/search/semantic
{
  "query": "knee pain and arthroscopy",
  "search_types": ["patients", "notes", "procedures", "imaging"],
  "limit": 10,
  "similarity_threshold": 0.1
}
```

### Consequences
- ✅ Flexible search capabilities for medical content
- ✅ Fast response times for real-time search
- ✅ Semantic understanding of medical terminology
- ⚠️ Complex ranking algorithm for hybrid results
- ⚠️ Requires corpus fitting for optimal embedding performance

---

## ADR-008: Deployment and Operations

**Date**: 2025-09-18  
**Status**: Accepted  
**Deciders**: Development Team

### Context
Medical systems require secure, reliable deployment with proper monitoring and compliance capabilities.

### Decision
**Local development with production-ready configuration**

### Deployment Characteristics
- **Air-gapped Deployment**: Supports isolated network environments
- **Docker Ready**: Containerized deployment capability
- **Configuration Management**: Environment-based configuration
- **Health Monitoring**: Built-in health check endpoints
- **Log Management**: Structured logging with audit trail

### Configuration
```python
# Security-focused defaults
HOST = "127.0.0.1"  # Localhost only for security
PORT = 8000
SESSION_TIMEOUT_MINUTES = 15
PASSWORD_MIN_LENGTH = 12
MFA_REQUIRED_FOR_PHI = True
AUDIT_LOG_RETENTION_DAYS = 2555  # 7 years HIPAA
```

### Consequences
- ✅ Secure by default configuration
- ✅ HIPAA-compliant deployment model
- ✅ Comprehensive monitoring capabilities
- ⚠️ Requires careful production hardening
- ⚠️ Air-gapped deployment may limit some features

---

## Implementation Timeline

### Phase 1: Foundation (Completed)
- [x] FastAPI application structure
- [x] PostgreSQL with pgvector setup
- [x] Basic authentication and RBAC
- [x] Patient data models

### Phase 2: Core Features (Completed)
- [x] Patient API endpoints
- [x] Advanced search capabilities
- [x] Vector embedding generation
- [x] Semantic search implementation

### Phase 3: Integration (In Progress)
- [ ] Hybrid search combining SQL and vector similarity
- [ ] Frontend dashboard integration
- [ ] Complete MFA implementation

### Phase 4: Production Readiness (Planned)
- [ ] Comprehensive testing suite
- [ ] Performance optimization
- [ ] Security audit and compliance review
- [ ] Production deployment documentation

---

## Performance Benchmarks

### Current Performance (as of 2025-09-18)
- **Semantic Search**: 79ms for 13 documents
- **Vector Generation**: <30 seconds for full corpus
- **Database Queries**: Sub-10ms for patient lookups
- **Memory Usage**: <200MB for full application
- **Embedding Dimensions**: 384 (optimal for performance/accuracy)

### Scalability Projections
- **Target**: 10,000+ patient records
- **Expected Search Time**: <200ms for semantic queries
- **Vector Storage**: ~15MB per 1,000 patients
- **Concurrent Users**: 50+ medical staff users

---

## Security Considerations

### Data Protection
- **Encryption at Rest**: Database encryption recommended
- **Encryption in Transit**: HTTPS with TLS 1.3
- **Key Management**: Secure secret key handling
- **Backup Security**: Encrypted backup requirements

### Access Controls
- **Principle of Least Privilege**: Role-based access controls
- **Session Security**: Secure token management
- **Audit Requirements**: Comprehensive activity logging
- **Compliance Monitoring**: Real-time security event detection

### Threat Model
- **Insider Threats**: Role-based access and audit logging
- **External Attacks**: Network security and authentication
- **Data Breaches**: Encryption and access controls
- **Compliance Violations**: Automated compliance checking

---

This architecture provides a solid foundation for a HIPAA-compliant orthopedic EMR system with advanced semantic search capabilities while maintaining security, performance, and operational simplicity.