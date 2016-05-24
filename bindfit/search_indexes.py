from haystack import indexes
from . import models

class FitIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.EdgeNgramField(document=True, use_template=True)
    searchable = indexes.CharField(model_attr='meta_options_searchable')

    fitter_name = indexes.CharField(model_attr='fitter_name')

    options_dilute = indexes.BooleanField(model_attr='options_dilute')
    options_normalise = indexes.BooleanField(model_attr='options_normalise')

    options_method = indexes.CharField(model_attr='options_method')
    options_flavour = indexes.CharField(model_attr='options_flavour')

    def get_model(self):
        return models.Fit 

    def index_queryset(self, using=None):
        """Used when the entire index for model is updated."""
        return self.get_model().objects.filter(meta_options_searchable=True)
