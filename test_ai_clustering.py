"""Test AI-powered clustering with semantic similarity."""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from apps.subjects.models import Subject
from apps.analytics.clustering import analyze_subject_topics

# Get Disaster Management subject
subject = Subject.objects.filter(name__icontains='DISASTER').first()

if not subject:
    print("❌ No Disaster Management subject found!")
    print("Please upload papers first.")
    exit(1)

print(f"\n{'='*80}")
print(f"🤖 Testing AI-Powered Clustering")
print(f"{'='*80}\n")
print(f"Subject: {subject.name}")
print(f"Papers: {subject.papers.count()}")
print(f"Questions: {subject.papers.first().questions.count() if subject.papers.exists() else 0}")

print(f"\n{'='*80}")
print(f"Running AI semantic similarity clustering...")
print(f"{'='*80}\n")

# Run clustering with AI
result = analyze_subject_topics(
    subject,
    similarity_threshold=0.75,  # High threshold for semantic matching
    tier_1_threshold=4,  # 4+ times = TOP PRIORITY
    tier_2_threshold=3,  # 3 times = HIGH PRIORITY
    tier_3_threshold=2   # 2 times = MEDIUM PRIORITY
)

print(f"\n✅ Clustering complete!")
print(f"   Clusters created: {result['clusters_created']}")
print(f"   Questions clustered: {result['questions_clustered']}")

# Show results by module and priority
from apps.analytics.models import TopicCluster

for module_num in range(1, 6):
    clusters = TopicCluster.objects.filter(
        subject=subject,
        module__number=module_num
    ).order_by('-frequency_count', 'topic_name')
    
    if not clusters.exists():
        continue
    
    print(f"\n{'='*80}")
    print(f"MODULE {module_num}")
    print(f"{'='*80}")
    
    # Group by priority
    tier1 = clusters.filter(priority_tier=TopicCluster.PriorityTier.TIER_1)
    tier2 = clusters.filter(priority_tier=TopicCluster.PriorityTier.TIER_2)
    tier3 = clusters.filter(priority_tier=TopicCluster.PriorityTier.TIER_3)
    tier4 = clusters.filter(priority_tier=TopicCluster.PriorityTier.TIER_4)
    
    if tier1.exists():
        print(f"\n🔥🔥🔥 TOP PRIORITY — Repeated 4+ Times ({tier1.count()} topics)")
        for cluster in tier1:
            print(f"   • {cluster.topic_name}")
            print(f"     Appears: {', '.join(cluster.years_appeared)}")
    
    if tier2.exists():
        print(f"\n🔥🔥 HIGH PRIORITY — Repeated 3 Times ({tier2.count()} topics)")
        for cluster in tier2:
            print(f"   • {cluster.topic_name}")
            print(f"     Appears: {', '.join(cluster.years_appeared)}")
    
    if tier3.exists():
        print(f"\n🔥 MEDIUM PRIORITY — Repeated 2 Times ({tier3.count()} topics)")
        for cluster in tier3:
            print(f"   • {cluster.topic_name}")
            print(f"     Appears: {', '.join(cluster.years_appeared)}")
    
    if tier4.exists():
        print(f"\n✓ LOW PRIORITY — Appears Once ({tier4.count()} topics)")
        # Show first 3 only
        for cluster in tier4[:3]:
            print(f"   • {cluster.topic_name}")
        if tier4.count() > 3:
            print(f"   ... and {tier4.count() - 3} more")

print(f"\n{'='*80}")
print(f"✅ AI Clustering Test Complete!")
print(f"{'='*80}\n")
