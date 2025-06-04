import os
import sys
import json
import random
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.models import StudentProfile, EngagementHistory, EngagementContent
from database.session import get_db


def generate_sample_data(num_students=100, num_engagements_per_student=10, num_content_items=50):
    """
    Generate sample data for the student engagement recommender system.
    
    Args:
        num_students: Number of student profiles to generate
        num_engagements_per_student: Average number of engagements per student
        num_content_items: Number of content items to generate
    """
    # Get session
    session = get_db()
    
    # Generate content items
    content_items = []
    
    # Check if content items already exist
    existing_content = session.query(EngagementContent).all()
    existing_content_ids = set(item.content_id for item in existing_content)
    
    if existing_content:
        print(f"Found {len(existing_content)} existing content items. Using them.")
        content_items = existing_content
    else:
        print("Generating new content items...")
        for i in range(num_content_items):
            content_id = f"C{i+1:05d}"
            
            # Skip if this content_id already exists
            if content_id in existing_content_ids:
                continue
                
            engagement_type = random.choice(['Email', 'SMS', 'Visit'])
            target_funnel_stage = random.choice(['Awareness', 'Interest', 'Consideration', 'Decision'])
            
            # Generate random embedding vector
            embedding = [round(random.uniform(-1, 1), 2) for _ in range(10)]
            
            content_item = EngagementContent(
                content_id=content_id,
                engagement_type=engagement_type,
                content_category=random.choice(['Program Information', 'Application Help', 'Campus Life', 'Financial Aid']),
                content_description=f"{engagement_type} content about {random.choice(['programs', 'campus', 'admissions', 'scholarships'])}",
                content_features={
                    'topics': random.sample(['curriculum', 'faculty', 'research', 'admissions', 'deadlines'], 2),
                    'embedding': embedding
                },
                success_rate=round(random.uniform(0.3, 0.9), 2),
                target_funnel_stage=target_funnel_stage,
                appropriate_for_risk_level=random.choice(['low', 'medium', 'high'])
            )
            
            try:
                session.add(content_item)
                session.flush()  # Try to flush to catch any constraint violations early
                content_items.append(content_item)
            except Exception as e:
                session.rollback()
                print(f"Error adding content item {content_id}: {e}")
                continue
    
    # Generate student profiles
    students = []
    
    # Check if students already exist
    existing_students = session.query(StudentProfile).all()
    existing_student_ids = set(student.student_id for student in existing_students)
    
    if existing_students:
        print(f"Found {len(existing_students)} existing student profiles. Using them.")
        students = existing_students
    else:
        print("Generating new student profiles...")
        for i in range(num_students):
            student_id = f"S{i+1:05d}"
            
            # Skip if this student_id already exists
            if student_id in existing_student_ids:
                continue
                
            # Generate random demographic features
            demographic_features = {
                'first_name': f"FirstName{i+1}",
                'last_name': f"LastName{i+1}",
                'high_school_gpa': round(random.uniform(2.5, 4.0), 2),
                'location': {
                    'city': f"City{random.randint(1, 20)}",
                    'state': f"State{random.randint(1, 10)}",
                    'zip_code': f"{random.randint(10000, 99999)}"
                },
                'academic_scores': {
                    'SAT': random.randint(1000, 1600),
                    'ACT': random.randint(20, 36),
                    'GPA': round(random.uniform(2.5, 4.0), 2)
                },
                'interests': random.sample(['science', 'arts', 'engineering', 'business', 'medicine', 'law'], 2),
                'extracurricular_activities': random.sample(['sports', 'music', 'volunteering', 'leadership', 'clubs'], 2)
            }
            
            # Generate application status
            application_status = {
                'started': random.random() > 0.2,
                'submitted': random.random() > 0.6,
                'completed': random.random() > 0.8
            }
            
            # Determine funnel stage based on application status
            if not application_status['started']:
                funnel_stage = random.choice(['Awareness', 'Interest'])
            elif not application_status['submitted']:
                funnel_stage = random.choice(['Interest', 'Consideration'])
            elif not application_status['completed']:
                funnel_stage = random.choice(['Consideration', 'Decision'])
            else:
                funnel_stage = 'Application'
            
            # Generate risk scores
            dropout_risk_score = round(random.uniform(0.1, 0.9), 2)
            application_likelihood_score = round(random.uniform(0.1, 0.9), 2)
            
            student = StudentProfile(
                student_id=student_id,
                demographic_features=demographic_features,
                application_status=application_status,
                funnel_stage=funnel_stage,
                last_interaction_date=(datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                dropout_risk_score=dropout_risk_score,
                application_likelihood_score=application_likelihood_score
            )
            
            try:
                session.add(student)
                session.flush()  # Try to flush to catch any constraint violations early
                students.append(student)
                existing_student_ids.add(student_id)  # Add to our tracking set
            except Exception as e:
                session.rollback()
                print(f"Error adding student {student_id}: {e}")
                continue
    
    # Generate engagement history
    engagement_counter = 1
    
    # Check if engagements already exist
    existing_engagements = session.query(EngagementHistory).all()
    existing_engagement_ids = set(engagement.engagement_id for engagement in existing_engagements)
    
    if existing_engagements:
        print(f"Found {len(existing_engagements)} existing engagement records. Skipping engagement generation.")
        engagement_counter = len(existing_engagements) + 1
    else:
        print("Generating new engagement history...")
        for student in students:
            # Determine number of engagements for this student
            num_engagements = random.randint(1, num_engagements_per_student * 2)
            
            for _ in range(num_engagements):
                # Create unique engagement ID
                engagement_id = f"E{engagement_counter:06d}"
                
                # Skip if this engagement_id already exists
                if engagement_id in existing_engagement_ids:
                    engagement_counter += 1
                    continue
                
                # Select a random content item
                content_item = random.choice(content_items)
                
                # Generate engagement timestamp
                timestamp = (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat()
                
                # Determine engagement response
                response_options = ['ignored', 'opened', 'clicked', 'responded']
                engagement_response = random.choices(
                    response_options, 
                    weights=[0.2, 0.4, 0.3, 0.1]
                )[0]
                
                # Determine funnel stage transition
                funnel_stages = ['Awareness', 'Interest', 'Consideration', 'Decision', 'Application']
                current_stage_index = funnel_stages.index(student.funnel_stage) if student.funnel_stage in funnel_stages else 0
                
                # Determine if there was a stage progression
                stage_progression = random.random() < 0.2  # 20% chance of stage progression
                
                funnel_stage_before = student.funnel_stage
                funnel_stage_after = student.funnel_stage
                
                if stage_progression and current_stage_index < len(funnel_stages) - 1:
                    funnel_stage_after = funnel_stages[current_stage_index + 1]
                
                # Create engagement record
                engagement = EngagementHistory(
                    engagement_id=engagement_id,
                    student_id=student.student_id,
                    content_id=content_item.content_id,
                    timestamp=timestamp,
                    engagement_response=engagement_response,
                    engagement_metrics={
                        'metrics': generate_metrics_for_type(content_item.engagement_type, engagement_response)
                    },
                    funnel_stage_before=funnel_stage_before,
                    funnel_stage_after=funnel_stage_after
                )
                
                try:
                    session.add(engagement)
                    session.flush()  # Try to flush to catch any constraint violations early
                    existing_engagement_ids.add(engagement_id)  # Add to our tracking set
                    engagement_counter += 1
                except Exception as e:
                    session.rollback()
                    print(f"Error adding engagement {engagement_id}: {e}")
                    engagement_counter += 1
                    continue
    
    # Commit changes
    try:
        session.commit()
        print(f"Successfully committed all data to the database.")
    except Exception as e:
        session.rollback()
        print(f"Error committing data: {e}")
        print("Rolling back and trying to commit in smaller batches...")
        
        # Try to commit in smaller batches
        try:
            # Commit content items
            session.commit()
            print("Database setup completed with some data.")
        except Exception as e:
            session.rollback()
            print(f"Failed to commit any data: {e}")
    
    # Count actual records in database
    student_count = session.query(StudentProfile).count()
    content_count = session.query(EngagementContent).count()
    engagement_count = session.query(EngagementHistory).count()
    
    print(f"Database contains {student_count} students, {content_count} content items, and {engagement_count} engagements.")


def generate_metrics_for_type(engagement_type, response):
    """Helper method to generate type-specific metrics"""
    if engagement_type == 'Email':
        if response in ['opened', 'clicked', 'responded']:
            return {
                'open_time': (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                'click_through': response in ['clicked', 'responded'],
                'time_spent': random.randint(10, 300) if response in ['clicked', 'responded'] else 0
            }
        else:
            return {
                'delivered': response != 'bounced',
                'bounced': response == 'bounced'
            }
    elif engagement_type == 'SMS':
        return {
            'delivered': response != 'bounced',
            'response_time': random.randint(1, 1440) if response == 'responded' else None,
            'response_length': random.randint(10, 200) if response == 'responded' else 0
        }
    elif engagement_type == 'Visit':
        if response in ['attended', 'completed']:
            return {
                'duration_minutes': random.randint(30, 180),
                'activities_participated': random.randint(1, 5),
                'staff_interactions': random.randint(0, 3)
            }
        else:
            return {
                'registered': True,
                'attended': False,
                'reason_for_absence': random.choice(['Schedule Conflict', 'Transportation Issues', 'Changed Mind', 'Unknown'])
            }
    else:
        return {}


if __name__ == "__main__":
    # Generate sample data
    generate_sample_data()
