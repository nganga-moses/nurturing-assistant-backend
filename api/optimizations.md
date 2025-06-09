-- Database Indexes for Risk Assessment Performance
-- Add these indexes to your tables

-- StudentProfile indexes
CREATE INDEX idx_student_funnel_stage ON student_profiles(funnel_stage);
CREATE INDEX idx_student_application_status ON student_profiles(application_status);
CREATE INDEX idx_student_id_lookup ON student_profiles(student_id);

-- EngagementHistory indexes (CRITICAL for performance)
CREATE INDEX idx_engagement_student_id ON engagement_history(student_id);
CREATE INDEX idx_engagement_timestamp ON engagement_history(timestamp);
CREATE INDEX idx_engagement_student_timestamp ON engagement_history(student_id, timestamp DESC);

-- Composite index for latest engagement queries
CREATE INDEX idx_engagement_latest ON engagement_history(student_id, timestamp DESC);

-- Optional: Materialized view for risk assessment (PostgreSQL)
CREATE MATERIALIZED VIEW risk_assessment_mv AS
WITH latest_engagements AS (
    SELECT 
        student_id,
        MAX(timestamp) as last_engagement
    FROM engagement_history 
    GROUP BY student_id
),
risk_scores AS (
    SELECT 
        s.student_id,
        s.funnel_stage,
        le.last_engagement,
        CASE 
            WHEN le.last_engagement IS NULL THEN 0.8
            WHEN le.last_engagement < NOW() - INTERVAL '30 days' THEN 0.9
            WHEN le.last_engagement < NOW() - INTERVAL '14 days' THEN 0.7
            WHEN le.last_engagement < NOW() - INTERVAL '7 days' THEN 0.5
            ELSE 0.2
        END * 
        CASE LOWER(s.funnel_stage)
            WHEN 'awareness' THEN 1.2
            WHEN 'interest' THEN 1.1
            WHEN 'consideration' THEN 1.0
            WHEN 'decision' THEN 0.9
            WHEN 'application' THEN 0.8
            ELSE 1.0
        END as risk_score
    FROM student_profiles s
    LEFT JOIN latest_engagements le ON s.student_id = le.student_id
)
SELECT 
    COUNT(*) FILTER (WHERE risk_score >= 0.7) as high_risk_count,
    COUNT(*) FILTER (WHERE risk_score >= 0.4 AND risk_score < 0.7) as medium_risk_count,
    COUNT(*) FILTER (WHERE risk_score < 0.4) as low_risk_count,
    COUNT(*) as total_students
FROM risk_scores;

-- Refresh every 15 minutes for real-time dashboard
-- REFRESH MATERIALIZED VIEW risk_assessment_mv;