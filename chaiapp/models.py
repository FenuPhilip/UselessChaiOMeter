from django.db import models

class ChaiResult(models.Model):
    device_uuid = models.CharField(max_length=100)
    chai_height = models.FloatField()
    froth_height = models.FloatField()
    chai_to_froth_ratio = models.FloatField()
    bubble_count = models.IntegerField()
    roast = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Result from {self.device_uuid} at {self.created_at}"
