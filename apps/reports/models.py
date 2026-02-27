"""
GeneratedReport model for tracking PDF report files.
"""
from django.db import models
from apps.core.models import BaseModel


class GeneratedReport(BaseModel):
    """
    Tracks a PDF report generated for a subject's module.

    Each record points to the file on disk (media/reports/).
    Reports can be regenerated; the old file will be replaced.
    """

    class ReportType(models.TextChoices):
        MODULE = "module", "Module Report"
        SUBJECT = "subject", "Full Subject Report"

    subject = models.ForeignKey(
        "subjects.Subject",
        on_delete=models.CASCADE,
        related_name="generated_reports",
    )
    module = models.ForeignKey(
        "subjects.Module",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="generated_reports",
        help_text="Null for full-subject reports.",
    )
    report_type = models.CharField(
        max_length=20,
        choices=ReportType.choices,
        default=ReportType.MODULE,
    )
    file = models.FileField(
        upload_to="reports/",
        null=True,
        blank=True,
        help_text="Generated PDF file.",
    )
    file_size_bytes = models.PositiveIntegerField(default=0)
    page_count = models.PositiveIntegerField(default=0)

    # Snapshot of generation parameters for reproducibility
    generation_meta = models.JSONField(
        default=dict,
        blank=True,
        help_text="Snapshot of settings used during generation.",
    )

    class Meta:
        verbose_name = "Generated Report"
        verbose_name_plural = "Generated Reports"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["subject", "module"]),
            models.Index(fields=["-created_at"]),
        ]

    def __str__(self) -> str:
        if self.module:
            return f"Report: {self.subject.name} – Module {self.module.number}"
        return f"Report: {self.subject.name} (full)"

    @property
    def download_url(self) -> str:
        """Return the media URL for downloading the report, or empty string."""
        try:
            return self.file.url if self.file else ""
        except ValueError:
            return ""
