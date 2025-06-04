# Initial Setup

This document describes the initial setup process for the Nurturing Assistant Backend.

## Overview

The initial setup process allows administrators to:
1. Import initial data (students, engagements, and content)
2. Train the recommendation model
3. Purge all data if needed

## Environment Variables

The following environment variables are required:

### Production Mode
- `GCS_BUCKET_NAME`: The name of the Google Cloud Storage bucket containing the initial data files

## Data Files

### Required Files
The following CSV files are required for the initial setup:

1. `students.csv`:
   - Required columns:
     - `student_id`: Unique identifier for each student
     - `demographic_features`: JSON string containing demographic information
     - `application_status`: Current application status

2. `engagements.csv`:
   - Required columns:
     - `student_id`: Reference to student
     - `engagement_type`: Type of engagement
     - `timestamp`: When the engagement occurred

3. `content.csv`:
   - Required columns:
     - `content_id`: Unique identifier for content
     - `content_type`: Type of content
     - `metadata`: JSON string containing content metadata

### File Locations

#### Testing Mode
Files should be placed in the `data/initial/` directory:
```
data/initial/
  ├── students.csv
  ├── engagements.csv
  └── content.csv
```

#### Production Mode
Files should be uploaded to the GCS bucket in the `initial/` directory:
```
initial/
  ├── students.csv
  ├── engagements.csv
  └── content.csv
```

## API Endpoints

### Setup Process
```http
POST /initial-setup/setup
Content-Type: application/json

{
  "mode": "testing" | "production"
}
```

### Check Status
```http
GET /initial-setup/status
```

### Purge Data
```http
POST /initial-setup/purge
```

## Setup Process Flow

1. **Validation**
   - Checks for required environment variables
   - Validates CSV file formats and required columns
   - Reports any validation errors

2. **Data Import**
   - Downloads files from GCS (production mode) or uses local files (testing mode)
   - Imports data into the database
   - Trains the recommendation model

3. **Progress Tracking**
   - Real-time progress updates
   - Detailed error reporting
   - Status monitoring endpoint

## Error Handling

The setup process includes comprehensive error handling:
- File validation errors
- Database errors
- GCS access errors
- Model training errors

All errors are reported through the status endpoint with detailed messages.

## Security

- All endpoints require admin authentication
- GCS credentials are managed through environment variables
- No sensitive data is exposed through the API

## Frontend Interface

The frontend provides a user-friendly interface for:
- Selecting setup mode (testing/production)
- Starting the setup process
- Monitoring progress
- Viewing errors
- Purging data

## Troubleshooting

Common issues and solutions:

1. **Missing Environment Variables**
   - Ensure `GCS_BUCKET_NAME` is set in production mode
   - Check environment variable configuration

2. **File Validation Errors**
   - Verify CSV file formats
   - Check for required columns
   - Ensure data types are correct

3. **GCS Access Issues**
   - Verify GCS credentials
   - Check bucket permissions
   - Ensure bucket exists

4. **Database Errors**
   - Check database connection
   - Verify table structure
   - Ensure sufficient permissions 