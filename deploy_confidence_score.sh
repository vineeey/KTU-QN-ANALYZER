#!/bin/bash
# Quick deployment script for confidence score feature

echo "🚀 Deploying Confidence Score Feature"
echo "======================================"

# Step 1: Apply database migration
echo "📊 Step 1: Applying database migration..."
python manage.py migrate analytics

# Step 2: Re-cluster existing data (optional but recommended)
echo "🔄 Step 2: Re-clustering existing data to calculate confidence scores..."
python recluster_topics.py

# Step 3: Verify installation
echo "✅ Step 3: Verification"
python manage.py shell -c "
from apps.analytics.models import TopicCluster
from django.db.models import Avg

# Check if confidence_score field exists
cluster = TopicCluster.objects.first()
if cluster:
    print(f'✓ Sample cluster: {cluster.topic_name}')
    print(f'  - Confidence: {cluster.confidence_score}%')
    print(f'  - Part A count: {cluster.part_a_count}')
    print(f'  - Part B count: {cluster.part_b_count}')
    
    avg_confidence = TopicCluster.objects.aggregate(Avg('confidence_score'))['confidence_score__avg']
    print(f'✓ Average confidence across all topics: {avg_confidence:.1f}%')
else:
    print('⚠ No topic clusters found. Upload and analyze papers first.')
"

echo ""
echo "✨ Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Visit analytics dashboard to see confidence scores"
echo "2. Generate module PDF reports to verify PDF output"
echo "3. Check that Part A/Part B counts are accurate"
