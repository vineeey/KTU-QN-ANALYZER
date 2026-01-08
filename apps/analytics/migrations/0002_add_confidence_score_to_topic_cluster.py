# Generated migration for adding confidence_score field to TopicCluster model

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analytics', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='topiccluster',
            name='confidence_score',
            field=models.FloatField(default=0.0, help_text='Percentage of uploaded years where this topic appeared (0-100)'),
        ),
    ]
