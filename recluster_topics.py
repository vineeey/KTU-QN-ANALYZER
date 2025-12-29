#!/usr/bin/env python
"""
Re-cluster questions with fixed canonical topic logic.
"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from apps.subjects.models import Subject
from apps.analytics.clustering import TopicClusteringService
from apps.analytics.models import TopicCluster

print("=" * 60)
print("RE-CLUSTERING WITH CANONICAL TOPIC LOGIC")
print("=" * 60)

# Get first subject (should be DISASTER MANAGEMENT or MOS)
subject = Subject.objects.first()
if not subject:
    print("❌ No subjects found!")
    exit(1)

print(f"\n📚 Subject: {subject.name}")
print(f"   University: {subject.university}")
print(f"   Papers: {subject.papers.count()}")

# Show before stats
before_clusters = TopicCluster.objects.filter(subject=subject).count()
print(f"\n📊 Before: {before_clusters} clusters")

# Re-cluster with new logic
print("\n🔄 Re-clustering with canonical topic extraction...")
print("   ✓ Lower similarity threshold (0.55)")
print("   ✓ Aggressive action verb removal")
print("   ✓ Core concept extraction only")
print("   ✓ Part A + Part B merged for frequency")
print("   ✓ Year-based counting (not question count)")

service = TopicClusteringService(subject)
result = service.analyze_subject()

print(f"\n✅ DONE!")
print(f"   Created: {result['clusters_created']} clusters")
print(f"   Questions clustered: {result['questions_clustered']}")

# Show tier distribution
tier_1 = TopicCluster.objects.filter(subject=subject, priority_tier=1).count()
tier_2 = TopicCluster.objects.filter(subject=subject, priority_tier=2).count()
tier_3 = TopicCluster.objects.filter(subject=subject, priority_tier=3).count()
tier_4 = TopicCluster.objects.filter(subject=subject, priority_tier=4).count()

print(f"\n🎯 Priority Distribution:")
print(f"   🔥🔥🔥 TIER 1 (4+ times): {tier_1} topics")
print(f"   🔥🔥 TIER 2 (3 times): {tier_2} topics")
print(f"   🔥 TIER 3 (2 times): {tier_3} topics")
print(f"   ✓ TIER 4 (1 time): {tier_4} topics")

# Show sample Tier 1 topics
if tier_1 > 0:
    print(f"\n📋 Sample TIER 1 Topics (TOP PRIORITY):")
    top_topics = TopicCluster.objects.filter(
        subject=subject, 
        priority_tier=1
    ).order_by('-frequency_count')[:5]
    
    for i, topic in enumerate(top_topics, 1):
        years = ', '.join(str(y) for y in topic.years_appeared) if topic.years_appeared else 'N/A'
        print(f"   {i}. {topic.topic_name}")
        print(f"      → {topic.frequency_count} times | Years: {years}")

print("\n" + "=" * 60)
print("✅ Re-clustering complete!")
print("=" * 60)
