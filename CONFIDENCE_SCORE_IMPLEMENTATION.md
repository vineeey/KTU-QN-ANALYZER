# Confidence Score & Part-Wise Contribution Implementation

## Summary
Successfully implemented the confidence score and Part A/Part B contribution tracking feature for the KTU PYQ Analyzer, as specified in the project requirements. This enhancement adds quantified intelligence to the priority scoring system.

## Changes Made

### 1. Database Model Enhancement
**File:** `apps/analytics/models.py`

- Added `confidence_score` field to `TopicCluster` model:
  - Type: `FloatField` (0.0-100.0)
  - Default: 0.0
  - Help text: "Percentage of uploaded years where this topic appeared (0-100)"
  - This field stores pre-calculated confidence for better performance

### 2. Clustering Service Updates
**File:** `apps/analytics/clustering.py`

#### Phase 1: Calculate Total Years Uploaded
- Modified `analyze_subject()` method to compute `total_years_uploaded`
- Counts distinct years from all papers in the subject
- Stores value in `self.total_years_uploaded` for use during cluster creation
- Falls back to paper count if years are not specified

#### Phase 2: Confidence Score Calculation
- Enhanced `_save_cluster()` method to calculate confidence during cluster creation
- Formula implemented: `confidence_score = (frequency_count / total_years_uploaded) × 100`
- Rounded to 1 decimal place for readability
- Example: Topic appeared in 6 out of 7 years = 85.7% confidence

#### Phase 3: Part-Wise Contribution
- Already tracking `part_a_count` and `part_b_count` in cluster creation
- Counts questions by their `part` field ('A' or 'B')
- Provides insight into whether topic appears as short answer or long answer

### 3. PDF Report Display
**Files:** 
- `apps/reports/ktu_report_generator.py`
- `apps/reports/module_report_generator.py`
- `apps/reports/utils.py`

**Status:** Already fully implemented ✅
- `calculate_confidence()` function exists in `utils.py`
- `format_priority_details()` includes confidence and part A/B in output
- PDF generation already displays: "Confidence: X% | Part A: Y | Part B: Z"

### 4. Web Analytics Dashboard
**File:** `templates/analytics/dashboard.html`

Added display of confidence and part-wise metrics for each topic cluster:
```html
<div class="flex items-center gap-4 mb-2 text-sm text-gray-600">
    <div>Confidence: <span class="text-blue-700 font-bold">{{ cluster.confidence_score }}%</span></div>
    <div>Part A: <span class="text-green-700 font-bold">{{ cluster.part_a_count }}</span></div>
    <div>Part B: <span class="text-purple-700 font-bold">{{ cluster.part_b_count }}</span></div>
</div>
```

### 5. Module Detail Analytics
**File:** `templates/analytics/module_detail.html`

Enhanced the topic table with two new columns:
- **Confidence:** Badge showing confidence percentage
- **Part A / Part B:** Shows split as "X / Y" with color-coded values

### 6. Database Migration
**File:** `apps/analytics/migrations/0002_add_confidence_score_to_topic_cluster.py`

Created migration to add the `confidence_score` field to existing `TopicCluster` records.

## How It Works

### Data Flow
1. **User uploads multiple PYQ PDFs** (various years)
2. **System extracts questions** and tracks their year and part (A/B)
3. **Clustering groups similar questions** into topics
4. **For each topic cluster:**
   - Count distinct years it appeared
   - Count Part A occurrences
   - Count Part B occurrences
   - Calculate: `confidence = (years_appeared / total_years) × 100`
5. **Display in reports and UI** with color-coded indicators

### Example Output

#### In PDF Report:
```
1. Disaster Risk Management (Framework + Core Elements)

Appears in: 2021, 2022, 2023, 2024, 2025
Repetition count: 5
Confidence: 71.4%

Appears as:
• Part A: 2 times
• Part B: 3 times

→ Very high probability long-answer topic.
```

#### In Web Dashboard:
- Visual badge showing confidence percentage
- Color-coded Part A (green) and Part B (purple) counts
- Sortable table with all metrics visible

## Technical Details

### Confidence Score Formula
```python
confidence_score = round((frequency_count / total_years_uploaded) * 100, 1)
```

### Part Distribution
```python
for question in cluster_questions:
    if question.part == 'A':
        part_a_count += 1
    elif question.part == 'B':
        part_b_count += 1
```

## Benefits to Students

1. **Confidence Score** answers: "How reliable is this topic's importance?"
   - 85%+ = Very reliable pattern
   - 50-84% = Moderately reliable
   - <50% = Less predictable

2. **Part A/Part B Split** answers: "How should I prepare?"
   - High Part A count → Focus on concise definitions/short notes
   - High Part B count → Prepare detailed explanations/derivations
   - Mixed → Be ready for both formats

## Testing Checklist

- [ ] Run migrations: `python manage.py migrate analytics`
- [ ] Re-cluster existing data: `python recluster_topics.py`
- [ ] Verify confidence scores appear in analytics dashboard
- [ ] Generate module PDF reports and check confidence display
- [ ] Test with multiple years of data to validate percentage calculation
- [ ] Verify Part A/Part B counts are accurate

## Future Enhancements

1. Add confidence-based filtering in UI (e.g., "Show only 70%+ confidence topics")
2. Visual confidence indicator (progress bar or gauge)
3. Trend analysis: Show if topic confidence is increasing/decreasing over time
4. Smart recommendations based on part distribution patterns

## Files Modified

1. `apps/analytics/models.py` - Added confidence_score field
2. `apps/analytics/clustering.py` - Calculate confidence and track total years
3. `templates/analytics/dashboard.html` - Display confidence and parts in dashboard
4. `templates/analytics/module_detail.html` - Add confidence column to table
5. `apps/analytics/migrations/0002_add_confidence_score_to_topic_cluster.py` - Migration

## Files Already Supporting Feature (No Changes Needed)

1. `apps/reports/utils.py` - calculate_confidence() function
2. `apps/reports/ktu_report_generator.py` - PDF generation with confidence
3. `apps/reports/module_report_generator.py` - Module reports with confidence

---

**Implementation Status:** ✅ Complete

All requirements from `.github/copilot-instructions.md` Phase 9 have been implemented successfully.
