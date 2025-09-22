# Orthopedic EMR RAG System

A HIPAA-compliant AI-powered medical information system for orthopedic practice with semantic search capabilities.

## 🏥 System Overview

This system provides a comprehensive Electronic Medical Records (EMR) solution specifically designed for orthopedic practices, featuring:

- **Semantic Search**: Vector-based similarity search across medical documents
- **HIPAA Compliance**: Enterprise-grade security and audit logging
- **Role-Based Access Control**: Medical staff permission management
- **Comprehensive Patient Management**: Full lifecycle from diagnosis to follow-up
- **Real-time Search**: Sub-100ms semantic search across medical records

## 🏗️ Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   PostgreSQL    │    │   Vector Store  │
│   Web Server    │◄──►│   Database      │◄──►│   (pgvector)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RBAC          │    │   Embedding     │    │   Search        │
│   Middleware    │    │   Service       │    │   Engine        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack

- **Backend**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL 17.5 with pgvector 0.8.1
- **Embeddings**: TF-IDF with SVD dimensionality reduction (384 dimensions)
- **Authentication**: JWT with bcrypt password hashing
- **Security**: CORS, trusted hosts, security headers, audit logging

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 17.5
- pgvector extension

### Installation

1. **Clone and setup environment**:
   ```bash
   cd app
   pip install fastapi uvicorn sqlalchemy asyncpg pydantic-settings
   pip install scikit-learn pgvector
   ```

2. **Configure database**:
   ```bash
   # Install pgvector
   brew install pgvector
   
   # Enable extension in PostgreSQL
   psql -d ortho_emr_security -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

3. **Initialize database schema**:
   ```bash
   python create_patient_tables.py
   python generate_patient_data.py
   python add_vector_columns.py
   python generate_embeddings.py
   ```

4. **Start the server**:
   ```bash
   python main.py
   ```

The server will start at `http://localhost:8000` with API documentation at `/docs`.

## 📊 Database Schema

### Core Tables

- **`patients`**: Patient demographics and medical history with vector embeddings
- **`medical_notes`**: Clinical notes with semantic embeddings  
- **`procedures`**: Surgical procedures and treatments with embeddings
- **`imaging_studies`**: Radiology reports with semantic search
- **`users`**: Medical staff with role-based permissions
- **`audit_logs`**: HIPAA-compliant access logging

### Vector Columns

All major medical content includes 384-dimensional vector embeddings:
- `patients.summary_embedding` - Patient summary semantics
- `medical_notes.content_embedding` - Clinical note content
- `procedures.content_embedding` - Procedure documentation
- `imaging_studies.content_embedding` - Radiology findings

## 🔍 API Endpoints

### Authentication
- `POST /api/v1/auth/login` - User authentication
- `POST /api/v1/auth/logout` - Session termination

### Patient Management
- `GET /api/v1/patients/` - List patients with pagination
- `GET /api/v1/patients/{id}` - Get patient details
- `POST /api/v1/patients/search` - Advanced patient search
- `GET /api/v1/patients/{id}/comprehensive` - Full patient record

### Semantic Search
- `POST /api/v1/search/semantic` - Vector similarity search
- `GET /api/v1/search/suggestions` - Search term suggestions

### System
- `GET /health` - Health check endpoint
- `GET /api` - API information

## 🔒 Security Features

### HIPAA Compliance
- **Audit Logging**: All PHI access logged with user attribution
- **Session Management**: 15-minute timeout with secure tokens
- **Access Controls**: Role-based permissions (Doctor, Nurse, Admin, etc.)
- **Data Encryption**: Secure password hashing and session management

### Role-Based Access Control (RBAC)
```python
# Available roles and permissions
MEDICAL_ROLES = {
    "doctor": [Permission.VIEW_PATIENTS, Permission.EDIT_PATIENTS, Permission.VIEW_PHI],
    "nurse": [Permission.VIEW_PATIENTS, Permission.VIEW_PHI],
    "admin": [Permission.MANAGE_USERS, Permission.VIEW_AUDIT_LOGS],
    "receptionist": [Permission.VIEW_PATIENTS]
}
```

## 🧠 Semantic Search Implementation

### Embedding Generation
- **Algorithm**: TF-IDF vectorization with SVD dimensionality reduction
- **Dimensions**: 384-dimensional vectors for optimal performance
- **Medical Text Processing**: Custom stopwords and medical terminology handling
- **Corpus**: 13 medical documents processed and embedded

### Search Performance
- **Response Time**: ~79ms for semantic queries
- **Similarity Scoring**: Cosine similarity with configurable thresholds
- **Result Types**: Patients, medical notes, procedures, imaging studies

### Example Search Results
```json
{
  "query": "knee pain and arthroscopy",
  "total_results": 13,
  "search_time_ms": 79,
  "results": [
    {
      "type": "patient",
      "score": 0.641,
      "title": "Maria Rodriguez",
      "content": "Medical History: No significant past medical history..."
    }
  ]
}
```

## 📋 Medical Data Model

### Patient Lifecycle
1. **Initial Consultation**: Chief complaint, history, physical exam
2. **Diagnostic Phase**: Imaging studies, lab results
3. **Treatment Planning**: Surgical procedures, conservative treatment
4. **Procedure Execution**: Operative notes, complications
5. **Recovery**: Physical therapy, follow-up appointments
6. **Billing**: Insurance processing, payment tracking

### Sample Patient Cases
The system includes 5 realistic orthopedic patient cases:
- **Robert Johnson**: Knee arthroscopy for meniscal tear
- **Maria Rodriguez**: Shoulder rotator cuff repair
- **David Wilson**: Hip replacement surgery
- **Sarah Brown**: ACL reconstruction
- **Jessica Chen**: Ankle fracture repair

## 🛠️ Development

### Project Structure
```
app/
├── api/                 # FastAPI route modules
│   ├── auth.py         # Authentication endpoints
│   ├── patients.py     # Patient management
│   ├── search.py       # Semantic search
│   └── router.py       # Main API router
├── core/               # Core application logic
│   ├── config.py       # Configuration management
│   ├── embeddings.py   # Vector embedding service
│   └── security.py     # Security middleware
├── database/           # Database components
│   ├── connection.py   # Database connection management
│   ├── models.py       # SQLAlchemy security models
│   └── patient_models.py # Medical data models
├── middleware/         # Custom middleware
│   ├── audit.py        # HIPAA audit logging
│   ├── rbac.py         # Role-based access control
│   └── session.py      # Session management
├── schemas/            # Pydantic data models
│   └── patients.py     # Patient API schemas
├── static/             # Frontend files
│   ├── index.html      # Medical dashboard
│   └── styles.css      # Dashboard styling
└── main.py            # Application entry point
```

### Testing
```bash
# Test semantic search
python test_semantic_search.py

# Test patient API
python api/test_rbac.py
```

## 📈 Performance Metrics

### Database Performance
- **Patient Records**: 5 comprehensive cases
- **Medical Documents**: 13 embedded documents
- **Vector Index**: Cosine similarity with IVFFlat indexing
- **Query Performance**: Sub-100ms semantic search

### Embedding Statistics
- **Patients**: 5 summary embeddings generated
- **Medical Notes**: 2 clinical notes embedded
- **Procedures**: 5 surgical procedures embedded  
- **Imaging Studies**: 1 radiology report embedded
- **Total Processing Time**: <30 seconds for full corpus

## 🔄 Current Status

### ✅ Completed Features
- [x] FastAPI application with security middleware
- [x] HIPAA-compliant authentication system
- [x] Role-based access control (RBAC) middleware
- [x] Comprehensive patient database with realistic cases
- [x] Patient data models and database schema
- [x] Patient API endpoints with RBAC protection
- [x] Advanced patient search with multiple criteria
- [x] pgvector extension installation and configuration
- [x] Vector embeddings for medical documents
- [x] Semantic search API endpoints

### 🚧 In Progress
- [ ] Hybrid search combining SQL and vector similarity

### 📋 Planned Features
- [ ] Patient search by insurance and billing status
- [ ] Medical records API endpoints
- [ ] Frontend dashboard integration
- [ ] Session management and security monitoring
- [ ] Health check and system status endpoints
- [ ] Complete MFA and authentication integration

## 🏥 Medical Compliance

### HIPAA Requirements
- **Administrative Safeguards**: Access controls, audit logs, user training
- **Physical Safeguards**: Secure deployment environment
- **Technical Safeguards**: Encryption, access controls, audit logs

### Audit Trail
All PHI access is logged with:
- User identification
- Timestamp of access
- Type of action performed
- Patient record accessed
- Success/failure status
- IP address and session information

## 🔧 Configuration

### Environment Variables
```bash
# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/ortho_emr_security
DB_ECHO=false

# Security Settings
SECRET_KEY=your-secret-key-here
ENVIRONMENT=development

# Application Settings
HOST=127.0.0.1
PORT=8000
```

### Medical Settings
```python
# Session and Security
SESSION_TIMEOUT_MINUTES=15
PASSWORD_MIN_LENGTH=12
MAX_LOGIN_ATTEMPTS=5
MFA_REQUIRED_FOR_PHI=True

# Audit and Compliance
AUDIT_LOG_RETENTION_DAYS=2555  # 7 years HIPAA requirement
REQUIRE_REASON_FOR_ACCESS=True
```

## 📚 References

- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

## 📞 Support

For technical support or medical workflow questions, please refer to the API documentation at `/docs` when the server is running.

---

**Note**: This system is designed for educational and development purposes. For production deployment in a medical environment, additional security reviews, compliance audits, and testing are required.