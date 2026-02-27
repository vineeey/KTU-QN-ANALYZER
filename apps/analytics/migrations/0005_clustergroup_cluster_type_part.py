"""
Migration: Add cluster_type and part fields to ClusterGroup.

cluster_type: discriminates between 'strict_repetition' (LLM-confirmed,
    used for study priority) and 'concept_similarity' (informational only).

part: 'A' or 'B' — enforces the module+part isolation contract.
    Each cluster belongs to exactly one part; cross-part clustering is
    forbidden by the ClusteringService.
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analytics', '0004_production_models'),
    ]

    operations = [
        # ── cluster_type field ─────────────────────────────────────────────
        migrations.AddField(
            model_name='clustergroup',
            name='cluster_type',
            field=models.CharField(
                choices=[
                    ('strict_repetition', 'Strict Repetition (LLM Confirmed)'),
                    ('concept_similarity', 'Concept Similarity (Informational)'),
                ],
                default='strict_repetition',
                db_index=True,
                max_length=30,
                help_text=(
                    'strict_repetition: verified repeated question (used for priority). '
                    'concept_similarity: topically related but not confirmed same (informational).'
                ),
            ),
        ),
        # ── part field ────────────────────────────────────────────────────
        migrations.AddField(
            model_name='clustergroup',
            name='part',
            field=models.CharField(
                default='A',
                db_index=True,
                max_length=1,
                help_text='Part A or Part B — enforced by ClusteringService isolation rules.',
            ),
        ),
        # ── composite index (subject, module, part) for fast lookups ──────
        migrations.AddIndex(
            model_name='clustergroup',
            index=models.Index(
                fields=['subject', 'module', 'part'],
                name='analytics_clustergroup_subject_module_part_idx',
            ),
        ),
        # ── index on cluster_type for filtered queries ─────────────────────
        migrations.AddIndex(
            model_name='clustergroup',
            index=models.Index(
                fields=['cluster_type'],
                name='analytics_clustergroup_cluster_type_idx',
            ),
        ),
    ]
