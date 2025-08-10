from django.db import models

class ChaiResult(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    device_uuid = models.CharField(max_length=128, blank=True, null=True)
    chai_teaspoons = models.FloatField(null=True)
    bubble_count = models.IntegerField(null=True)
    raw_data = models.JSONField(null=True, blank=True)  # store raw metrics if needed

    def __str__(self):
        return f"ChaiResult {self.id} tsp={self.chai_teaspoons} bubbles={self.bubble_count}"
