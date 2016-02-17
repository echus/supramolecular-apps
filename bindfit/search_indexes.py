from haystack import indexes
from . import models

class FitIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True, use_template=True)
    #author = indexes.CharField(model_attr='user')

    def get_model(self):
        return models.Fit 

    def index_queryset(self, using=None):
        """Used when the entire index for model is updated."""
        return self.get_model().objects.all()
