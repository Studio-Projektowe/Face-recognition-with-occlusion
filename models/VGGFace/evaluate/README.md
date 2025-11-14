https://stackoverflow.com/questions/51337558/how-to-import-keras-engine-topology-in-tensorflow

> ![NOTE]
> In the keras_vggface/models.py file, change the import from:
> from keras.engine.topology import get_source_inputs
> to:
> from keras.utils.layer_utils import get_source_inputs

https://stackoverflow.com/questions/77680911/import-error-in-google-colab-with-keras-utils