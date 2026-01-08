"""
Management command for automatic cleanup of expired jobs.

Run this as a cron job:
    python manage.py cleanup_expired_jobs

Or use Django-Q scheduled task:
    from django_q.tasks import schedule
    schedule('apps.analysis.management.commands.cleanup_expired_jobs.Command',
             schedule_type=Schedule.HOURLY)
"""
from django.core.management.base import BaseCommand
from apps.analysis.pipeline_13phases import Phase13_Cleanup


class Command(BaseCommand):
    help = 'Clean up expired analysis jobs and their data'
    
    def handle(self, *args, **options):
        self.stdout.write('Starting cleanup of expired jobs...')
        
        count = Phase13_Cleanup.cleanup_expired_jobs()
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully cleaned up {count} expired jobs')
        )
